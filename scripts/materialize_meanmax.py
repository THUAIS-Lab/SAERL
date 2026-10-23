"""Merge SAE activation shards into the curriculum representation table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _list_parquet_files(input_root: Path) -> list[Path]:
    if input_root.is_file():
        return [input_root]
    return sorted(p for p in input_root.rglob("*.parquet") if p.is_file())


def _row_counts(paths: list[Path]) -> list[int]:
    counts: list[int] = []
    for path in paths:
        counts.append(int(pq.ParquetFile(path).metadata.num_rows))
    return counts


class RowStream:
    def __init__(self, path: Path, batch_size: int = 256) -> None:
        self.path = path
        self.parquet = pq.ParquetFile(path)
        self.schema = self.parquet.schema_arrow
        self.iterator = self.parquet.iter_batches(batch_size=batch_size)
        self.batch: dict[str, list] | None = None
        self.index = 0
        self._load_next_batch()

    def _load_next_batch(self) -> None:
        try:
            next_batch = next(self.iterator)
        except StopIteration:
            self.batch = None
            self.index = 0
            return
        self.batch = next_batch.to_pydict()
        self.index = 0

    def exhausted(self) -> bool:
        return self.batch is None

    def current_sample_id(self) -> int:
        if self.batch is None:
            raise RuntimeError(f"stream exhausted: {self.path}")
        return int(self.batch["sample_id"][self.index])

    def pop(self) -> dict[str, object]:
        if self.batch is None:
            raise RuntimeError(f"stream exhausted: {self.path}")
        row = {key: values[self.index] for key, values in self.batch.items()}
        self.index += 1
        if self.index >= len(self.batch["sample_id"]):
            self._load_next_batch()
        return row


def _write_buffer(writer: pq.ParquetWriter, schema: pa.Schema, rows: list[dict[str, object]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def _open_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    return pq.ParquetWriter(path, schema=schema, compression="snappy", use_dictionary=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--part-path", action="append", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--sae-path", required=True)
    parser.add_argument("--conda-env", default=None, help="Optional extraction environment name")
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--read-batch-size", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--gpus", nargs="+", default=[], help="Optional extraction GPU IDs")
    parser.add_argument("--merge-batch-size", type=int, default=256)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    part_paths = [Path(p).expanduser().resolve() for p in args.part_path]
    missing = [str(path) for path in part_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing part parquet(s): {missing}")

    streams = [RowStream(path, batch_size=int(args.merge_batch_size)) for path in part_paths]
    schemas = [stream.schema for stream in streams]
    schema = schemas[0]
    for path, other_schema in zip(part_paths[1:], schemas[1:]):
        if other_schema != schema:
            raise ValueError(f"schema mismatch: {path}")

    samples_meanmax_path = output_dir / "samples_meanmax.parquet"
    samples_meanmax_writer = _open_writer(samples_meanmax_path, schema)

    source_files = _list_parquet_files(input_root)
    source_counts = _row_counts(source_files)
    source_total = sum(source_counts)
    expected_rows = int(args.expected_rows)
    if source_total != expected_rows:
        raise ValueError(f"source row count mismatch: got {source_total}, expected {expected_rows}")

    merged_rows = 0
    next_expected_sample_id = 0
    sample_meanmax_buffer: list[dict[str, object]] = []
    part_row_counts: dict[str, int] = {}
    part_sample_id_ranges: dict[str, dict[str, int]] = {}

    try:
        for part_path in part_paths:
            meta = pq.ParquetFile(part_path).metadata
            part_row_counts[str(part_path)] = int(meta.num_rows)

        while True:
            active = [stream for stream in streams if not stream.exhausted()]
            if not active:
                break
            stream = min(active, key=lambda s: s.current_sample_id())
            row = stream.pop()
            sample_id = int(row["sample_id"])

            if sample_id != next_expected_sample_id:
                raise ValueError(
                    f"sample_id order mismatch: expected {next_expected_sample_id}, got {sample_id}"
                )

            source_key = str(stream.path)
            if source_key not in part_sample_id_ranges:
                part_sample_id_ranges[source_key] = {"min": sample_id, "max": sample_id}
            else:
                part_sample_id_ranges[source_key]["max"] = sample_id

            # Each row stores both span means and maxima.
            sample_meanmax_buffer.append(row)
            merged_rows += 1
            next_expected_sample_id += 1

            if len(sample_meanmax_buffer) >= int(args.merge_batch_size):
                _write_buffer(samples_meanmax_writer, schema, sample_meanmax_buffer)
                sample_meanmax_buffer = []

        _write_buffer(samples_meanmax_writer, schema, sample_meanmax_buffer)
    finally:
        samples_meanmax_writer.close()

    if merged_rows != expected_rows:
        raise ValueError(f"merged row count mismatch: got {merged_rows}, expected {expected_rows}")

    source_payload = [
        {"path": str(path), "rows": int(count)}
        for path, count in zip(source_files, source_counts)
    ]

    manifest = {
        "source_root": str(input_root),
        "source_files_sorted": source_payload,
        "row_count_per_file": {str(path): int(count) for path, count in zip(source_files, source_counts)},
        "source_total_rows": int(source_total),
        "sample_id_policy": "sample_id = sorted_shards_prefix_sum + row_in_file",
        "model_name_or_path": str(Path(args.model_name_or_path).expanduser().resolve()),
        "sae_path": str(Path(args.sae_path).expanduser().resolve()),
        "conda_env": args.conda_env,
        "gpus": [int(x) for x in args.gpus],
        "dtype": args.dtype,
        "batch_size": int(args.batch_size),
        "read_batch_size": int(args.read_batch_size),
        "max_seq_len": int(args.max_seq_len),
        "final_rows": int(merged_rows),
        "sample_id_min": 0,
        "sample_id_max": int(expected_rows - 1),
        "is_sorted_by_sample_id": True,
        "part_row_counts": part_row_counts,
        "part_sample_id_ranges": part_sample_id_ranges,
        "storage_layout": {
            "samples_meanmax.parquet": "single physical Parquet artifact",
        },
        "variant_semantics": {
            "samples_meanmax.parquet": {
                "prompt_feat_vals": "mean = prompt_feat_sum / max(prompt_token_count, 1)",
                "sol_feat_vals": "mean = sol_feat_sum / max(sol_token_count, 1)",
                "prompt_feat_max": "retained alongside mean values",
                "sol_feat_max": "retained alongside mean values",
            },
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    print(f"wrote: {samples_meanmax_path}")
    print(f"wrote: {manifest_path}")
    print(f"rows: {merged_rows}")


if __name__ == "__main__":
    main()
