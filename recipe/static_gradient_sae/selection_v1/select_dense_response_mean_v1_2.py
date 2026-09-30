#!/usr/bin/env python3
"""Select a non-redundant core set from dense response-mean vectors.

This is the dense-response-mean variant of ``selection_v1.2.md``. It reads
only ``sample_id``, ``dense_grad_feat_vals``, and ``zero_signal_dense``. From
the fixed ``[prompt_mean || prompt_signed_peak || response_mean || response_signed_peak]``
layout it selects ``dense_grad_feat_vals[2F:3F]``.

CUDA is used automatically when available and when the normalized dense matrix
fits in free GPU memory. Otherwise the same exact selection runs on CPU.

Example:
    python select_dense_response_mean_v1_2.py \
        --run-dir SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461 \
        --cosine-threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .select_topk_response_mean_v1_2 import (
        LoadedRepresentations,
        _primitive_numpy,
        _ragged_numpy,
        _select_with_device,
        _write_outputs,
    )
except ImportError:
    from recipe.static_gradient_sae.selection_v1.select_topk_response_mean_v1_2 import (
        LoadedRepresentations,
        _primitive_numpy,
        _ragged_numpy,
        _select_with_device,
        _write_outputs,
    )

INPUT_COLUMNS = (
    "sample_id",
    "dense_grad_feat_vals",
    "zero_signal_dense",
)
EXPECTED_TYPES = {
    "sample_id": pa.int64(),
    "dense_grad_feat_vals": pa.list_(pa.float32()),
    "zero_signal_dense": pa.bool_(),
}
REPRESENTATION_LAYOUT = (
    "prompt_mean",
    "prompt_signed_peak",
    "response_mean",
    "response_signed_peak",
)


def _load_dense_manifest(run_dir: Path) -> tuple[int, float]:
    path = run_dir / "representation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    feature_size = int(manifest.get("sae_feature_size", 0))
    if feature_size < 1:
        raise ValueError(f"Invalid sae_feature_size in {path}: {feature_size}")
    if tuple(manifest.get("representation_layout", ())) != REPRESENTATION_LAYOUT:
        raise ValueError(
            f"Unsupported representation_layout in {path}: "
            f"{manifest.get('representation_layout')!r}; expected {list(REPRESENTATION_LAYOUT)!r}"
        )
    expected_size = 4 * feature_size
    if int(manifest.get("dense_representation_size", -1)) != expected_size:
        raise ValueError(
            f"dense_representation_size in {path} must be 4F={expected_size}, "
            f"got {manifest.get('dense_representation_size')!r}"
        )
    zero_eps = float(manifest.get("gradient_zero_eps", 1e-12))
    if not math.isfinite(zero_eps) or zero_eps < 0.0:
        raise ValueError(f"Invalid gradient_zero_eps in {path}: {zero_eps}")
    return feature_size, zero_eps


def _validate_dense_schema(schema: pa.Schema, path: Path) -> None:
    for name, expected in EXPECTED_TYPES.items():
        index = schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"Missing required column {name!r} in {path}")
        actual = schema.field(index).type
        if actual != expected:
            raise TypeError(f"Column {name!r} in {path} has type {actual}, expected {expected}")


def _discover_dense_shards(run_dir: Path, input_glob: str) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(
        path for path in run_dir.glob(input_glob) if path.is_file() and not path.name.endswith(".inprogress")
    )
    if not paths:
        raise FileNotFoundError(f"No input shards match {input_glob!r} under {run_dir}")
    first_schema = pq.ParquetFile(paths[0]).schema_arrow
    _validate_dense_schema(first_schema, paths[0])
    metadata: list[dict[str, Any]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(first_schema):
            raise ValueError(f"Parquet schema mismatch: {path}")
        metadata.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "rows": int(parquet.metadata.num_rows),
                "row_groups": int(parquet.metadata.num_row_groups),
            }
        )
    return paths, metadata


def _count_dense_candidates(paths: list[Path], read_batch_size: int) -> tuple[int, int]:
    """Count rows and the maximum active capacity without reading dense values."""
    total = 0
    candidate_capacity = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=["zero_signal_dense"],
            use_threads=True,
        ):
            flags = _primitive_numpy(batch.column(0), np.bool_)
            total += int(flags.size)
            candidate_capacity += int(np.count_nonzero(~flags))
    return total, candidate_capacity


def _load_dense_response_means(
    paths: list[Path],
    shard_metadata: list[dict[str, Any]],
    *,
    feature_size: int,
    zero_eps: float,
    read_batch_size: int,
) -> LoadedRepresentations:
    total_rows, candidate_capacity = _count_dense_candidates(paths, read_batch_size)
    normalized_storage = np.empty((candidate_capacity, feature_size), dtype=np.float32)
    sample_ids = np.empty(total_rows, dtype=np.int64)
    signal_strengths = np.empty(total_rows, dtype=np.float64)
    zero_signal_flags = np.empty(total_rows, dtype=np.bool_)
    active_all_rows = np.empty(candidate_capacity, dtype=np.int64)
    active_sample_ids = np.empty(candidate_capacity, dtype=np.int64)
    active_signal_strengths = np.empty(candidate_capacity, dtype=np.float64)
    seen_sample_ids: set[int] = set()
    all_count = 0
    active_count = 0
    expected_dense_size = 4 * feature_size
    response_offset = 2 * feature_size

    for shard_number, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=list(INPUT_COLUMNS),
            use_threads=True,
        ):
            batch_sample_ids = _primitive_numpy(batch.column(0), np.int64)
            dense_offsets, dense_values = _ragged_numpy(batch.column(1), np.float32)
            batch_zero_flags = _primitive_numpy(batch.column(2), np.bool_)
            lengths = np.diff(dense_offsets)
            if np.any(lengths != expected_dense_size):
                bad_row = int(np.flatnonzero(lengths != expected_dense_size)[0])
                raise ValueError(
                    f"{path} row {bad_row} has dense length {int(lengths[bad_row])}, expected 4F={expected_dense_size}"
                )

            for row in range(batch.num_rows):
                sample_id = int(batch_sample_ids[row])
                if sample_id in seen_sample_ids:
                    raise ValueError(f"Duplicate sample_id across extraction shards: {sample_id}")
                seen_sample_ids.add(sample_id)
                start = int(dense_offsets[row]) + response_offset
                response_mean = dense_values[start : start + feature_size]
                if not np.all(np.isfinite(response_mean)):
                    raise ValueError(f"sample_id={sample_id} dense response mean contains NaN or infinity")
                response64 = response_mean.astype(np.float64, copy=False)
                norm = math.sqrt(max(float(np.dot(response64, response64)), 0.0))
                zero_flag = bool(batch_zero_flags[row])

                sample_ids[all_count] = sample_id
                signal_strengths[all_count] = norm
                zero_signal_flags[all_count] = zero_flag
                if not zero_flag and norm > zero_eps:
                    normalized_storage[active_count] = response_mean / norm
                    active_all_rows[active_count] = all_count
                    active_sample_ids[active_count] = sample_id
                    active_signal_strengths[active_count] = norm
                    active_count += 1
                all_count += 1

        print(
            f"Loaded dense shard {shard_number}/{len(paths)}: {path.name} ({all_count:,} total samples)",
            flush=True,
        )

    if all_count != total_rows:
        raise ValueError(f"Dense row-count mismatch: expected={total_rows}, loaded={all_count}")
    if active_count > candidate_capacity:
        raise AssertionError("Active dense row count exceeded precomputed capacity")

    return LoadedRepresentations(
        feature_size=feature_size,
        zero_eps=zero_eps,
        sample_ids=sample_ids,
        signal_strengths=signal_strengths,
        zero_signal_flags=zero_signal_flags,
        active_all_rows=active_all_rows[:active_count],
        active_sample_ids=active_sample_ids[:active_count],
        active_signal_strengths=active_signal_strengths[:active_count],
        normalized_active=normalized_storage[:active_count],
        shard_metadata=shard_metadata,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Gradient-SAE extraction run directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: RUN_DIR/selection_dense_response_mean_v1_2)",
    )
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device; auto uses CUDA when available and CPU otherwise",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=4,
        help="Dense representation rows read from Parquet together",
    )
    parser.add_argument(
        "--similarity-block-size",
        type=int,
        default=64,
        help="Rows per exact cosine block",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not -1.0 <= float(args.cosine_threshold) <= 1.0:
        raise ValueError("--cosine-threshold must be in [-1, 1]")
    if int(args.read_batch_size) < 1 or int(args.similarity_block_size) < 1:
        raise ValueError("--read-batch-size and --similarity-block-size must be positive")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "selection_dense_response_mean_v1_2"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    started = time.monotonic()
    feature_size, zero_eps = _load_dense_manifest(run_dir)
    paths, shard_metadata = _discover_dense_shards(run_dir, str(args.input_glob))
    loaded = _load_dense_response_means(
        paths,
        shard_metadata,
        feature_size=feature_size,
        zero_eps=zero_eps,
        read_batch_size=int(args.read_batch_size),
    )
    result, compute = _select_with_device(
        loaded.normalized_active,
        loaded.active_sample_ids,
        loaded.active_signal_strengths,
        cosine_threshold=float(args.cosine_threshold),
        block_size=int(args.similarity_block_size),
        requested_device=str(args.device),
    )
    elapsed = time.monotonic() - started
    manifest = _write_outputs(
        output_dir,
        loaded,
        result,
        run_dir=run_dir,
        input_glob=str(args.input_glob),
        cosine_threshold=float(args.cosine_threshold),
        block_size=int(args.similarity_block_size),
        compute=compute,
        method="selection_dense_response_mean_v1.2",
        input_columns=INPUT_COLUMNS,
        representation="sample-level Dense response mean",
        zero_signal_field="zero_signal_dense",
        elapsed_seconds=elapsed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
