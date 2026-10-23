"""Extract prompt and solution SAE activations from Parquet shards.

Store per-feature mean, sum, occurrence count, and maximum for each span.
Mean pooling includes all valid tokens in the span, including feature zeros.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from saerl.core.cli_utils import atomic_write_json, ensure_dir, setup_logging
from saerl.sae_analysis.extract import _process_batch
from saerl.sae_analysis.sae.opensae_adapter import OpenSaeRunner

logger = logging.getLogger(__name__)


def _dtype_from_str(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def _build_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("difficulty", pa.float64()),
            ("prompt_len_tok", pa.int64()),
            ("sol_len_tok", pa.int64()),
            ("full_len_tok", pa.int64()),
            ("cut", pa.int64()),
            ("was_truncated", pa.bool_()),
            ("latex_ratio", pa.float64()),
            ("digit_ratio", pa.float64()),
            ("sym_ratio", pa.float64()),
            ("prompt_token_count", pa.int64()),
            ("prompt_feat_ids", pa.list_(pa.int64())),
            ("prompt_feat_vals", pa.list_(pa.float64())),
            ("prompt_feat_sum", pa.list_(pa.float64())),
            ("prompt_feat_count", pa.list_(pa.int64())),
            ("prompt_feat_max", pa.list_(pa.float64())),
            ("sol_token_count", pa.int64()),
            ("sol_feat_ids", pa.list_(pa.int64())),
            ("sol_feat_vals", pa.list_(pa.float64())),
            ("sol_feat_sum", pa.list_(pa.float64())),
            ("sol_feat_count", pa.list_(pa.int64())),
            ("sol_feat_max", pa.list_(pa.float64())),
        ]
    )


def _list_parquet_files(input_root: Path) -> List[Path]:
    if input_root.is_file():
        if input_root.suffix.lower() != ".parquet":
            raise ValueError(f"input_root is a file but not .parquet: {input_root}")
        return [input_root]
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    paths = sorted([p for p in input_root.rglob("*.parquet") if p.is_file()])
    if not paths:
        raise ValueError(f"No parquet files found under: {input_root}")
    return paths


def _parquet_total_rows(parquet_paths: Sequence[Path]) -> int:
    total = 0
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        if meta is None:
            total += int(pf.read().num_rows)
        else:
            total += int(meta.num_rows)
    return int(total)


def _parquet_row_counts(parquet_paths: Sequence[Path]) -> List[int]:
    counts: List[int] = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        meta = pf.metadata
        if meta is None:
            counts.append(int(pf.read().num_rows))
        else:
            counts.append(int(meta.num_rows))
    return counts


def _prefix_offsets(counts: Sequence[int]) -> List[int]:
    offsets: List[int] = []
    total = 0
    for count in counts:
        offsets.append(total)
        total += int(count)
    return offsets


def _select_files(
    parquet_paths: Sequence[Path],
    row_counts: Sequence[int],
    offsets: Sequence[int],
    *,
    file_mod: int,
    file_rem: int,
    max_rows: int | None,
) -> List[Tuple[Path, int, int]]:
    selected: List[Tuple[Path, int, int]] = []
    for idx, (path, offset, count) in enumerate(zip(parquet_paths, offsets, row_counts)):
        if idx % file_mod != file_rem:
            continue
        eff_count = int(count)
        if max_rows is not None:
            remaining = int(max_rows) - int(offset)
            if remaining <= 0:
                eff_count = 0
            else:
                eff_count = min(eff_count, remaining)
        if eff_count <= 0:
            continue
        selected.append((path, int(offset), int(eff_count)))
    return selected


def _initial_done(selected_files: Sequence[Tuple[Path, int, int]], last_index: int) -> int:
    done = 0
    for _path, offset, count in selected_files:
        if last_index < offset:
            continue
        end = offset + count - 1
        if last_index >= end:
            done += count
        else:
            done += last_index - offset + 1
    return int(done)


def _first_nonempty(row: Dict[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return ""


def _sample_id_selected(sample_id: int, *, sample_id_mod: int, sample_id_rem: int) -> bool:
    return int(sample_id) % int(sample_id_mod) == int(sample_id_rem)


def _count_selected_rows(
    files: Sequence[Tuple[Path, int, int]],
    *,
    sample_id_field: str | None,
    read_batch_size: int,
    sample_id_mod: int,
    sample_id_rem: int,
    max_sample_id: int | None = None,
) -> int:
    total = 0
    columns = [sample_id_field] if sample_id_field else None

    for path, base_offset, max_rows_in_file in files:
        pf = pq.ParquetFile(path)
        row_in_file = 0
        for batch in pf.iter_batches(batch_size=read_batch_size, columns=columns):
            n = batch.num_rows
            if row_in_file >= int(max_rows_in_file):
                break
            batch_end = min(row_in_file + n, int(max_rows_in_file))
            end_idx = batch_end - row_in_file
            if sample_id_field:
                sample_ids = batch.column(0).to_pylist()
                for idx in range(end_idx):
                    raw = sample_ids[idx]
                    sample_id = int(base_offset) + row_in_file + idx if raw is None else int(raw)
                    if max_sample_id is not None and sample_id > int(max_sample_id):
                        continue
                    if _sample_id_selected(sample_id, sample_id_mod=sample_id_mod, sample_id_rem=sample_id_rem):
                        total += 1
            else:
                for idx in range(end_idx):
                    sample_id = int(base_offset) + row_in_file + idx
                    if max_sample_id is not None and sample_id > int(max_sample_id):
                        continue
                    if _sample_id_selected(sample_id, sample_id_mod=sample_id_mod, sample_id_rem=sample_id_rem):
                        total += 1
            row_in_file += n
    return int(total)


def _iter_rows_from_parquet_files(
    files: Sequence[Tuple[Path, int, int]],
    *,
    question_field: str,
    solution_fields: Sequence[str],
    difficulty_field: str,
    sample_id_field: str | None,
    read_batch_size: int,
    start_sample_id: int,
    sample_id_mod: int,
    sample_id_rem: int,
) -> Iterable[Dict[str, object]]:
    columns = [question_field, difficulty_field, *solution_fields]
    if sample_id_field and sample_id_field not in columns:
        columns.append(sample_id_field)

    for path, base_offset, max_rows_in_file in files:
        pf = pq.ParquetFile(path)
        start_in_file = 0 if sample_id_field else max(int(start_sample_id) - int(base_offset), 0)
        if not sample_id_field and start_in_file >= int(max_rows_in_file):
            continue
        row_in_file = 0
        for batch in pf.iter_batches(batch_size=read_batch_size, columns=columns):
            data = batch.to_pydict()
            n = len(data.get(question_field, []))
            if row_in_file >= int(max_rows_in_file):
                break
            batch_end = min(row_in_file + n, int(max_rows_in_file))
            if batch_end <= start_in_file:
                row_in_file += n
                continue
            start_idx = max(start_in_file - row_in_file, 0)
            end_idx = batch_end - row_in_file
            for idx in range(start_idx, end_idx):
                row = {col: (data.get(col) or [None] * n)[idx] for col in columns}
                if sample_id_field:
                    sample_id_raw = row.get(sample_id_field)
                    if sample_id_raw is None:
                        sample_id = int(base_offset) + row_in_file + idx
                    else:
                        sample_id = int(sample_id_raw)
                    if sample_id < int(start_sample_id):
                        continue
                else:
                    sample_id = int(base_offset) + row_in_file + idx
                    if sample_id < int(start_sample_id):
                        continue
                if not _sample_id_selected(sample_id, sample_id_mod=sample_id_mod, sample_id_rem=sample_id_rem):
                    continue
                question = str(row.get(question_field) or "")
                solution = _first_nonempty(row, solution_fields)
                difficulty_raw = row.get(difficulty_field)
                difficulty = float("nan") if difficulty_raw is None else float(difficulty_raw)

                yield {
                    "sample_id": sample_id,
                    "problem": question,
                    "solution": solution,
                    "difficulty": difficulty,
                }
            row_in_file += n


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SAE activations from Parquet shards.")
    parser.add_argument("--input-root", required=True, help="Parquet root or file")
    parser.add_argument("--out-dir", default="./runs/deepmath103k", help="Output run directory")
    parser.add_argument("--model-name-or-path", required=True, help="HF model name or local path")
    parser.add_argument("--sae-path", required=True, help="OpenSAE checkpoint path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=2, help="Model forward batch size")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug limit on total rows scanned")
    parser.add_argument("--read-batch-size", type=int, default=256, help="Parquet read batch size")
    parser.add_argument("--file-mod", type=int, default=1, help="Process only files where (index %% mod) == rem")
    parser.add_argument("--file-rem", type=int, default=0, help="Remainder for --file-mod selection")
    parser.add_argument(
        "--sample-id-mod",
        type=int,
        default=1,
        help="Process only rows where (sample_id %% mod) == rem; works with row-order ids or --sample-id-field.",
    )
    parser.add_argument("--sample-id-rem", type=int, default=0, help="Remainder for --sample-id-mod selection")

    parser.add_argument("--question-field", default="question")
    parser.add_argument(
        "--solution-fields",
        nargs="+",
        default=["r1_solution_1", "r1_solution_2", "r1_solution_3"],
        help="Try these fields in order; first non-empty is used as solution",
    )
    parser.add_argument("--difficulty-field", default="difficulty")
    parser.add_argument(
        "--sample-id-field",
        default=None,
        help="Optional parquet field containing the original sample_id; if set, it overrides row-order ids.",
    )

    args = parser.parse_args()

    setup_logging()
    out_dir = Path(args.out_dir)
    activations_dir = ensure_dir(out_dir / "artifacts" / "activations")
    progress_path = activations_dir / "progress.json"
    errors_path = activations_dir / "errors.jsonl"
    parquet_path = activations_dir / "samples.parquet"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    dtype = _dtype_from_str(args.dtype)
    runner = OpenSaeRunner(args.model_name_or_path, args.sae_path, device=args.device, dtype=dtype)

    last_index = -1
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as handle:
            last_index = int(json.load(handle).get("last_index", -1))
        logger.info("Resuming from sample_id=%s", last_index)

    schema = _build_schema()

    resume_mode = progress_path.exists() and parquet_path.exists()
    target_path = parquet_path

    writer: pq.ParquetWriter
    if resume_mode:
        target_path = parquet_path.with_suffix(".parquet.tmp")
        logger.info("Resuming with existing parquet; copying to %s", target_path)
        parquet_file = pq.ParquetFile(parquet_path)
        if parquet_file.schema_arrow != schema:
            backup_path = parquet_path.with_suffix(".parquet.bak")
            logger.warning(
                "Existing parquet schema differs from expected; backing up to %s and restarting extraction.",
                backup_path,
            )
            parquet_path.replace(backup_path)
            progress_path.replace(progress_path.with_suffix(".json.bak"))
            last_index = -1
            resume_mode = False
            target_path = parquet_path
        else:
            writer = pq.ParquetWriter(target_path, schema=schema, compression="snappy", use_dictionary=True)
            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_idx)
                writer.write_table(table)

    if not resume_mode:
        if parquet_path.exists():
            parquet_path.unlink()
        writer = pq.ParquetWriter(target_path, schema=schema, compression="snappy", use_dictionary=True)

    input_root = Path(args.input_root)
    parquet_paths = _list_parquet_files(input_root)
    logger.info("Found %s parquet shards under %s", len(parquet_paths), input_root)

    if int(args.file_mod) < 1:
        raise ValueError("--file-mod must be >= 1")
    if not (0 <= int(args.file_rem) < int(args.file_mod)):
        raise ValueError("--file-rem must satisfy: 0 <= rem < mod")
    if int(args.sample_id_mod) < 1:
        raise ValueError("--sample-id-mod must be >= 1")
    if not (0 <= int(args.sample_id_rem) < int(args.sample_id_mod)):
        raise ValueError("--sample-id-rem must satisfy: 0 <= rem < mod")

    total_rows = _parquet_total_rows(parquet_paths)
    if args.max_rows is not None:
        total_rows = min(total_rows, int(args.max_rows))
    if total_rows <= 0:
        raise ValueError("No rows found in parquet dataset.")

    row_counts = _parquet_row_counts(parquet_paths)
    offsets = _prefix_offsets(row_counts)
    selected_files = _select_files(
        parquet_paths,
        row_counts,
        offsets,
        file_mod=int(args.file_mod),
        file_rem=int(args.file_rem),
        max_rows=args.max_rows,
    )
    if not selected_files:
        logger.info("No shards selected with file_mod=%s file_rem=%s", args.file_mod, args.file_rem)
        return

    start_sample_id = max(last_index + 1, 0)
    if int(args.sample_id_mod) == 1:
        selected_total_rows = sum(count for _path, _offset, count in selected_files)
        initial = _initial_done(selected_files, last_index)
    else:
        selected_total_rows = _count_selected_rows(
            selected_files,
            sample_id_field=args.sample_id_field,
            read_batch_size=int(args.read_batch_size),
            sample_id_mod=int(args.sample_id_mod),
            sample_id_rem=int(args.sample_id_rem),
        )
        initial = 0
        if last_index >= 0:
            initial = _count_selected_rows(
                selected_files,
                sample_id_field=args.sample_id_field,
                read_batch_size=int(args.read_batch_size),
                sample_id_mod=int(args.sample_id_mod),
                sample_id_rem=int(args.sample_id_rem),
                max_sample_id=last_index,
            )
    if initial >= selected_total_rows:
        logger.info("Nothing to do: already processed %s/%s selected rows", initial, selected_total_rows)
        return

    data_iter = _iter_rows_from_parquet_files(
        selected_files,
        question_field=args.question_field,
        solution_fields=args.solution_fields,
        difficulty_field=args.difficulty_field,
        sample_id_field=args.sample_id_field,
        read_batch_size=args.read_batch_size,
        start_sample_id=start_sample_id,
        sample_id_mod=int(args.sample_id_mod),
        sample_id_rem=int(args.sample_id_rem),
    )

    batch: List[Dict[str, object]] = []
    start_time = time.time()

    for sample in tqdm(
        data_iter,
        total=selected_total_rows,
        initial=initial,
        desc="Extracting",
        unit="sample",
    ):
        batch.append(sample)
        if len(batch) < int(args.batch_size):
            continue
        _process_batch(
            batch,
            tokenizer,
            runner,
            int(pad_id),
            int(args.max_seq_len),
            schema,
            writer,
            errors_path,
        )
        last_index = int(batch[-1].get("sample_id", last_index))
        atomic_write_json(progress_path, {"last_index": last_index})
        batch = []

    if batch:
        _process_batch(
            batch,
            tokenizer,
            runner,
            int(pad_id),
            int(args.max_seq_len),
            schema,
            writer,
            errors_path,
        )
        last_index = int(batch[-1].get("sample_id", last_index))
        atomic_write_json(progress_path, {"last_index": last_index})

    writer.close()
    if resume_mode:
        target_path.replace(parquet_path)

    elapsed = time.time() - start_time
    logger.info("Parquet extraction completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
