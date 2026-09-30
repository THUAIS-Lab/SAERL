#!/usr/bin/env python3
"""Select samples from calibrated Token-TopK response mean and signed peak.

This implements ``selection_v1.md``. For every sample it constructs

    z = [response_mean / (s_mean + eps) || response_signed_peak / (s_peak + eps)]

where each branch scale is its RMS L2 norm over samples not marked
``zero_signal_token_topk``. Samples are processed by descending ``||z||_2``
and pruned using exact signed cosine against the selected set.

CUDA is used automatically when available and when the dense normalized
matrix fits in free GPU memory. Otherwise the exact sparse CPU path is used.

Example:
    python select_topk_response_mean_peak_v1.py \
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
import scipy.sparse as sp

try:
    from .select_topk_response_mean_v1_2 import (
        LoadedRepresentations,
        _primitive_numpy,
        _ragged_numpy,
        _select_with_device,
        _write_outputs,
    )
except ImportError:
    from select_topk_response_mean_v1_2 import (
        LoadedRepresentations,
        _primitive_numpy,
        _ragged_numpy,
        _select_with_device,
        _write_outputs,
    )

INPUT_COLUMNS = (
    "sample_id",
    "topk_response_feat_ids",
    "topk_response_feat_mean",
    "topk_response_feat_signed_peak",
    "zero_signal_token_topk",
)
EXPECTED_TYPES = {
    "sample_id": pa.int64(),
    "topk_response_feat_ids": pa.list_(pa.int64()),
    "topk_response_feat_mean": pa.list_(pa.float32()),
    "topk_response_feat_signed_peak": pa.list_(pa.float32()),
    "zero_signal_token_topk": pa.bool_(),
}
REPRESENTATION_LAYOUT = (
    "prompt_mean",
    "prompt_signed_peak",
    "response_mean",
    "response_signed_peak",
)


def _load_manifest(run_dir: Path) -> tuple[int, float]:
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
    if int(manifest.get("topk_representation_size", -1)) != expected_size:
        raise ValueError(
            f"topk_representation_size in {path} must be 4F={expected_size}, "
            f"got {manifest.get('topk_representation_size')!r}"
        )
    zero_eps = float(manifest.get("gradient_zero_eps", 1e-12))
    if not math.isfinite(zero_eps) or zero_eps < 0.0:
        raise ValueError(f"Invalid gradient_zero_eps in {path}: {zero_eps}")
    return feature_size, zero_eps


def _validate_schema(schema: pa.Schema, path: Path) -> None:
    for name, expected in EXPECTED_TYPES.items():
        index = schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"Missing required column {name!r} in {path}")
        actual = schema.field(index).type
        if actual != expected:
            raise TypeError(f"Column {name!r} in {path} has type {actual}, expected {expected}")


def _discover_shards(run_dir: Path, input_glob: str) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(
        path for path in run_dir.glob(input_glob) if path.is_file() and not path.name.endswith(".inprogress")
    )
    if not paths:
        raise FileNotFoundError(f"No input shards match {input_glob!r} under {run_dir}")

    first_schema = pq.ParquetFile(paths[0]).schema_arrow
    _validate_schema(first_schema, paths[0])
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


def _load_mean_peak_representations(
    paths: list[Path],
    shard_metadata: list[dict[str, Any]],
    *,
    feature_size: int,
    zero_eps: float,
    read_batch_size: int,
) -> tuple[LoadedRepresentations, dict[str, Any]]:
    sample_ids: list[int] = []
    zero_signal_flags: list[bool] = []
    mean_squared_norms: list[float] = []
    peak_squared_norms: list[float] = []
    candidate_all_rows: list[int] = []
    candidate_sample_ids: list[int] = []
    candidate_indices: list[np.ndarray] = []
    candidate_values: list[np.ndarray] = []
    candidate_indptr = [0]
    seen_sample_ids: set[int] = set()
    calibration_count = 0
    mean_squared_norm_sum = 0.0
    peak_squared_norm_sum = 0.0

    for shard_number, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=list(INPUT_COLUMNS),
            use_threads=True,
        ):
            sample_id_array = batch.column(batch.schema.get_field_index("sample_id"))
            ids_array = batch.column(batch.schema.get_field_index("topk_response_feat_ids"))
            mean_array = batch.column(batch.schema.get_field_index("topk_response_feat_mean"))
            peak_array = batch.column(batch.schema.get_field_index("topk_response_feat_signed_peak"))
            zero_array = batch.column(batch.schema.get_field_index("zero_signal_token_topk"))

            batch_sample_ids = _primitive_numpy(sample_id_array, np.int64)
            id_offsets, feature_ids = _ragged_numpy(ids_array, np.int64)
            mean_offsets, mean_values = _ragged_numpy(mean_array, np.float32)
            peak_offsets, peak_values = _ragged_numpy(peak_array, np.float32)
            batch_zero_flags = _primitive_numpy(zero_array, np.bool_)

            for row in range(batch.num_rows):
                sample_id = int(batch_sample_ids[row])
                if sample_id in seen_sample_ids:
                    raise ValueError(f"Duplicate sample_id across extraction shards: {sample_id}")
                seen_sample_ids.add(sample_id)

                ids = feature_ids[int(id_offsets[row]) : int(id_offsets[row + 1])]
                means = mean_values[int(mean_offsets[row]) : int(mean_offsets[row + 1])]
                peaks = peak_values[int(peak_offsets[row]) : int(peak_offsets[row + 1])]
                if ids.size != means.size or ids.size != peaks.size:
                    raise ValueError(
                        f"sample_id={sample_id} has inconsistent response arrays: "
                        f"ids={ids.size}, mean={means.size}, signed_peak={peaks.size}"
                    )
                if ids.size and (ids[0] < 0 or ids[-1] >= feature_size):
                    raise ValueError(f"sample_id={sample_id} response feature ids fall outside [0, {feature_size})")
                if ids.size > 1 and np.any(ids[1:] <= ids[:-1]):
                    raise ValueError(f"sample_id={sample_id} response feature ids must be strictly increasing")
                if not np.all(np.isfinite(means)) or not np.all(np.isfinite(peaks)):
                    raise ValueError(f"sample_id={sample_id} response mean or signed peak contains NaN or infinity")

                means64 = means.astype(np.float64, copy=False)
                peaks64 = peaks.astype(np.float64, copy=False)
                mean_squared_norm = max(float(np.dot(means64, means64)), 0.0)
                peak_squared_norm = max(float(np.dot(peaks64, peaks64)), 0.0)
                zero_flag = bool(batch_zero_flags[row])
                all_row = len(sample_ids)
                sample_ids.append(sample_id)
                zero_signal_flags.append(zero_flag)
                mean_squared_norms.append(mean_squared_norm)
                peak_squared_norms.append(peak_squared_norm)

                if zero_flag:
                    continue

                calibration_count += 1
                mean_squared_norm_sum += mean_squared_norm
                peak_squared_norm_sum += peak_squared_norm
                mean_nonzero = means != 0.0
                peak_nonzero = peaks != 0.0
                row_indices = np.concatenate(
                    (
                        ids[mean_nonzero],
                        ids[peak_nonzero] + feature_size,
                    )
                ).astype(np.int64, copy=False)
                row_values = np.concatenate((means[mean_nonzero], peaks[peak_nonzero])).astype(np.float32, copy=False)
                candidate_all_rows.append(all_row)
                candidate_sample_ids.append(sample_id)
                candidate_indices.append(row_indices.copy())
                candidate_values.append(row_values.copy())
                candidate_indptr.append(candidate_indptr[-1] + int(row_indices.size))

        print(
            f"Loaded shard {shard_number}/{len(paths)}: {path.name} ({len(sample_ids):,} total samples)",
            flush=True,
        )

    if not sample_ids:
        raise ValueError("No representation rows were found")

    mean_rms_norm = math.sqrt(mean_squared_norm_sum / calibration_count) if calibration_count else 0.0
    peak_rms_norm = math.sqrt(peak_squared_norm_sum / calibration_count) if calibration_count else 0.0
    calibration_epsilon = zero_eps if zero_eps > 0.0 else float(np.finfo(np.float64).eps)
    mean_denominator = mean_rms_norm + calibration_epsilon
    peak_denominator = peak_rms_norm + calibration_epsilon

    all_mean_squared = np.asarray(mean_squared_norms, dtype=np.float64)
    all_peak_squared = np.asarray(peak_squared_norms, dtype=np.float64)
    signal_strengths = np.hypot(
        np.sqrt(all_mean_squared) / mean_denominator,
        np.sqrt(all_peak_squared) / peak_denominator,
    )
    if not np.all(np.isfinite(signal_strengths)):
        raise ValueError("Calibrated mean+peak signal strengths contain NaN or infinity")

    if candidate_indices:
        indices = np.concatenate(candidate_indices)
        values = np.concatenate(candidate_values)
    else:
        indices = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=np.float32)
    mean_branch = indices < feature_size
    values[mean_branch] /= mean_denominator
    values[~mean_branch] /= peak_denominator
    del mean_branch
    candidates = sp.csr_matrix(
        (values, indices, np.asarray(candidate_indptr, dtype=np.int64)),
        shape=(len(candidate_sample_ids), 2 * feature_size),
        dtype=np.float32,
    )
    candidates.sort_indices()
    del candidate_indices, candidate_values, indices, values

    candidate_all_rows_array = np.asarray(candidate_all_rows, dtype=np.int64)
    candidate_strengths = signal_strengths[candidate_all_rows_array]
    active_mask = candidate_strengths > zero_eps
    normalized = candidates[active_mask].tocsr()
    del candidates
    active_strengths = candidate_strengths[active_mask]
    for row, strength in enumerate(active_strengths):
        start = int(normalized.indptr[row])
        stop = int(normalized.indptr[row + 1])
        normalized.data[start:stop] *= np.float32(1.0 / strength)
    normalized.eliminate_zeros()
    normalized.sort_indices()

    loaded = LoadedRepresentations(
        feature_size=feature_size,
        zero_eps=zero_eps,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        signal_strengths=signal_strengths,
        zero_signal_flags=np.asarray(zero_signal_flags, dtype=np.bool_),
        active_all_rows=candidate_all_rows_array[active_mask],
        active_sample_ids=np.asarray(candidate_sample_ids, dtype=np.int64)[active_mask],
        active_signal_strengths=active_strengths,
        normalized_active=normalized,
        shard_metadata=shard_metadata,
    )
    calibration = {
        "sample_count": calibration_count,
        "epsilon": calibration_epsilon,
        "response_mean_rms_norm": mean_rms_norm,
        "response_signed_peak_rms_norm": peak_rms_norm,
        "response_mean_denominator": mean_denominator,
        "response_signed_peak_denominator": peak_denominator,
    }
    print(
        "Branch calibration: "
        f"N={calibration_count:,}, response_mean RMS={mean_rms_norm:.8g}, "
        f"response_signed_peak RMS={peak_rms_norm:.8g}",
        flush=True,
    )
    return loaded, calibration


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Gradient-SAE extraction run directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: RUN_DIR/selection_topk_response_mean_peak_v1)",
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
        help="Rows read from Parquet together",
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
        else run_dir / "selection_topk_response_mean_peak_v1"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    started = time.monotonic()
    feature_size, zero_eps = _load_manifest(run_dir)
    paths, shard_metadata = _discover_shards(run_dir, str(args.input_glob))
    loaded, calibration = _load_mean_peak_representations(
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
        method="selection_topk_response_mean_peak_v1",
        input_columns=INPUT_COLUMNS,
        representation="sample-level Token-TopK calibrated response mean + signed peak",
        zero_signal_field="zero_signal_token_topk",
        elapsed_seconds=elapsed,
        signal_strength_description="L2 norm after response mean/peak branch RMS calibration",
        similarity_description="exact signed cosine on calibrated response mean + signed peak",
        silent_zero_reason="calibrated_response_mean_peak_zero",
        extra_manifest={"branch_calibration": calibration},
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
