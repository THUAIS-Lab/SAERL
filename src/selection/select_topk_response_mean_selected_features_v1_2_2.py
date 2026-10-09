#!/usr/bin/env python3
"""Select samples using Token-TopK response mean projected to chosen features.

The selected SAE feature indices are loaded from a one-dimensional integer
``.npy`` file. Coordinate ``k`` of the projected vector corresponds exactly
to ``feature_indices[k]``; matching response-mean values are unchanged and
missing features are zero. Selection then follows ``selection_v1.2.2.md``.

CUDA is used automatically when available. Otherwise the same exact signed-
cosine greedy selection runs on CPU.

Example:
    python select_topk_response_mean_selected_features_v1_2_2.py \
        --run-dir SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461 \
        --feature-indices RUN_DIR/feature_selection/energy_feature_indices.npy \
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
import pyarrow.parquet as pq

try:
    from .select_topk_response_mean_v1_2 import (
        INPUT_COLUMNS,
        LoadedRepresentations,
        _discover_shards,
        _hash_int64,
        _load_manifest,
        _primitive_numpy,
        _ragged_numpy,
        _select_with_device,
        _write_outputs,
    )
except ImportError:
    from saerl.selection.select_topk_response_mean_v1_2 import (
        INPUT_COLUMNS,
        LoadedRepresentations,
        _discover_shards,
        _hash_int64,
        _load_manifest,
        _primitive_numpy,
        _ragged_numpy,
        _select_with_device,
        _write_outputs,
    )


def _should_log_progress(current: int, total: int) -> bool:
    interval = max(1, total // 20)
    return current == 1 or current == total or current % interval == 0


def _load_feature_indices(path: Path, source_feature_size: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    indices = np.load(path, allow_pickle=False)
    if indices.ndim != 1:
        raise ValueError(f"Feature-index array must be one-dimensional, got shape={indices.shape}")
    if indices.size == 0:
        raise ValueError("Feature-index array must not be empty")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"Feature-index array must have an integer dtype, got {indices.dtype}")
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= source_feature_size):
        invalid = indices[(indices < 0) | (indices >= source_feature_size)][0]
        raise ValueError(f"Feature index {int(invalid)} falls outside source feature range [0, {source_feature_size})")
    if np.unique(indices).size != indices.size:
        raise ValueError("Feature-index array contains duplicate indices")
    return np.ascontiguousarray(indices)


def _count_rows_and_candidates(paths: list[Path], read_batch_size: int) -> tuple[int, int]:
    total_rows = 0
    candidate_capacity = 0
    for shard_number, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=["zero_signal_token_topk"],
            use_threads=True,
        ):
            flags = _primitive_numpy(batch.column(0), np.bool_)
            total_rows += int(flags.size)
            candidate_capacity += int(np.count_nonzero(~flags))
        if _should_log_progress(shard_number, len(paths)):
            print(
                f"[count] shards={shard_number:,}/{len(paths):,} "
                f"rows={total_rows:,} nonzero-flag candidates={candidate_capacity:,}",
                flush=True,
            )
    return total_rows, candidate_capacity


def _load_projected_representations(
    paths: list[Path],
    shard_metadata: list[dict[str, Any]],
    *,
    source_feature_size: int,
    selected_feature_indices: np.ndarray,
    zero_eps: float,
    read_batch_size: int,
) -> tuple[LoadedRepresentations, dict[str, int]]:
    total_rows, candidate_capacity = _count_rows_and_candidates(paths, read_batch_size)
    selected_feature_size = int(selected_feature_indices.size)
    feature_to_coordinate = np.full(source_feature_size, -1, dtype=np.int64)
    feature_to_coordinate[selected_feature_indices] = np.arange(selected_feature_size, dtype=np.int64)

    normalized_storage = np.zeros((candidate_capacity, selected_feature_size), dtype=np.float32)
    sample_ids = np.empty(total_rows, dtype=np.int64)
    signal_strengths = np.empty(total_rows, dtype=np.float64)
    zero_signal_flags = np.empty(total_rows, dtype=np.bool_)
    active_all_rows = np.empty(candidate_capacity, dtype=np.int64)
    active_sample_ids = np.empty(candidate_capacity, dtype=np.int64)
    active_signal_strengths = np.empty(candidate_capacity, dtype=np.float64)
    observed_selected_features = np.zeros(selected_feature_size, dtype=np.bool_)
    seen_sample_ids: set[int] = set()

    all_count = 0
    active_count = 0
    source_zero_count = 0
    projected_zero_count = 0
    matched_value_count = 0

    for shard_number, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=list(INPUT_COLUMNS),
            use_threads=True,
        ):
            ids_array = batch.column(batch.schema.get_field_index("topk_response_feat_ids"))
            mean_array = batch.column(batch.schema.get_field_index("topk_response_feat_mean"))
            id_offsets, feature_ids = _ragged_numpy(ids_array, np.int64)
            mean_offsets, mean_values = _ragged_numpy(mean_array, np.float32)
            batch_sample_ids = _primitive_numpy(batch.column(batch.schema.get_field_index("sample_id")), np.int64)
            batch_zero_flags = _primitive_numpy(
                batch.column(batch.schema.get_field_index("zero_signal_token_topk")), np.bool_
            )

            for row in range(batch.num_rows):
                sample_id = int(batch_sample_ids[row])
                if sample_id in seen_sample_ids:
                    raise ValueError(f"Duplicate sample_id across extraction shards: {sample_id}")
                seen_sample_ids.add(sample_id)

                ids = feature_ids[int(id_offsets[row]) : int(id_offsets[row + 1])]
                values = mean_values[int(mean_offsets[row]) : int(mean_offsets[row + 1])]
                if ids.size != values.size:
                    raise ValueError(
                        f"sample_id={sample_id} has inconsistent response arrays: ids={ids.size}, mean={values.size}"
                    )
                if ids.size and (ids[0] < 0 or ids[-1] >= source_feature_size):
                    raise ValueError(
                        f"sample_id={sample_id} response feature ids fall outside [0, {source_feature_size})"
                    )
                if ids.size > 1 and np.any(ids[1:] <= ids[:-1]):
                    raise ValueError(f"sample_id={sample_id} response feature ids must be strictly increasing")
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"sample_id={sample_id} response mean contains NaN or infinity")

                coordinates = feature_to_coordinate[ids]
                matched = coordinates >= 0
                selected_coordinates = coordinates[matched]
                selected_values = values[matched]
                selected_values64 = selected_values.astype(np.float64, copy=False)
                norm = math.sqrt(max(float(np.dot(selected_values64, selected_values64)), 0.0))
                zero_flag = bool(batch_zero_flags[row])

                sample_ids[all_count] = sample_id
                signal_strengths[all_count] = norm
                zero_signal_flags[all_count] = zero_flag
                matched_value_count += int(selected_values.size)
                if selected_values.size:
                    observed_selected_features[selected_coordinates[selected_values != 0.0]] = True

                if zero_flag:
                    source_zero_count += 1
                elif norm <= zero_eps:
                    projected_zero_count += 1
                else:
                    nonzero = selected_values != 0.0
                    normalized_storage[active_count, selected_coordinates[nonzero]] = selected_values[nonzero] / norm
                    active_all_rows[active_count] = all_count
                    active_sample_ids[active_count] = sample_id
                    active_signal_strengths[active_count] = norm
                    active_count += 1
                all_count += 1

        if _should_log_progress(shard_number, len(paths)):
            print(
                f"[project] shards={shard_number:,}/{len(paths):,} rows={all_count:,}/{total_rows:,} "
                f"active={active_count:,} source-silent={source_zero_count:,} "
                f"projected-zero={projected_zero_count:,}",
                flush=True,
            )

    if all_count != total_rows:
        raise ValueError(f"Row-count mismatch: expected={total_rows}, loaded={all_count}")
    if active_count > candidate_capacity:
        raise AssertionError("Active row count exceeded precomputed capacity")

    loaded = LoadedRepresentations(
        feature_size=selected_feature_size,
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
    projection_stats = {
        "source_zero_signal_samples": source_zero_count,
        "projected_zero_samples": projected_zero_count,
        "active_samples": active_count,
        "matched_stored_values": matched_value_count,
        "observed_nonzero_selected_features": int(np.count_nonzero(observed_selected_features)),
    }
    print(
        f"[project] complete: selected_feature_size={selected_feature_size:,}, "
        f"active={active_count:,}, source-silent={source_zero_count:,}, "
        f"projected-zero={projected_zero_count:,}, "
        f"observed-features={projection_stats['observed_nonzero_selected_features']:,}",
        flush=True,
    )
    return loaded, projection_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Gradient-SAE extraction run directory")
    parser.add_argument(
        "--feature-indices",
        required=True,
        help="One-dimensional integer .npy file containing selected SAE feature indices",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: RUN_DIR/selection_v1_2_2_FEATURE_FILE_STEM)",
    )
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device; auto uses CUDA when available and CPU otherwise",
    )
    parser.add_argument("--read-batch-size", type=int, default=4)
    parser.add_argument("--similarity-block-size", type=int, default=64)
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
    feature_indices_path = Path(args.feature_indices).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / f"selection_v1_2_2_{feature_indices_path.stem}"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    started = time.monotonic()
    _source_manifest, source_feature_size, zero_eps = _load_manifest(run_dir)
    selected_feature_indices = _load_feature_indices(feature_indices_path, source_feature_size)
    print(
        "[config] "
        f"run_dir={run_dir} feature_indices={feature_indices_path} "
        f"source_feature_size={source_feature_size:,} "
        f"selected_feature_size={selected_feature_indices.size:,} "
        f"threshold={float(args.cosine_threshold):.6g} device={args.device}",
        flush=True,
    )
    print(
        f"[features] min={int(selected_feature_indices.min()):,} "
        f"max={int(selected_feature_indices.max()):,} "
        f"sha256={_hash_int64(selected_feature_indices)}",
        flush=True,
    )

    paths, shard_metadata = _discover_shards(run_dir, str(args.input_glob))
    total_rows = sum(int(item["rows"]) for item in shard_metadata)
    print(f"[discover] shards={len(paths):,} rows={total_rows:,}", flush=True)
    loaded, projection_stats = _load_projected_representations(
        paths,
        shard_metadata,
        source_feature_size=source_feature_size,
        selected_feature_indices=selected_feature_indices,
        zero_eps=zero_eps,
        read_batch_size=int(args.read_batch_size),
    )

    print(
        f"[select] starting exact signed-cosine selection for {loaded.active_sample_ids.size:,} active samples",
        flush=True,
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
    feature_hash = _hash_int64(selected_feature_indices)
    manifest = _write_outputs(
        output_dir,
        loaded,
        result,
        run_dir=run_dir,
        input_glob=str(args.input_glob),
        cosine_threshold=float(args.cosine_threshold),
        block_size=int(args.similarity_block_size),
        compute=compute,
        method="selection_v1.2.2",
        input_columns=INPUT_COLUMNS,
        representation="Token-TopK response mean projected to selected features",
        zero_signal_field="zero_signal_token_topk",
        elapsed_seconds=elapsed,
        signal_strength_description="L2 norm of selected-feature response-mean vector",
        similarity_description="exact signed cosine in selected-feature response-mean space",
        silent_zero_reason="selected_feature_response_mean_zero",
        extra_manifest={
            "source_sae_feature_size": source_feature_size,
            "selected_feature_size": int(selected_feature_indices.size),
            "feature_indices_file": str(feature_indices_path),
            "feature_indices_sha256": feature_hash,
            "selected_feature_indices": selected_feature_indices.tolist(),
            "projection": projection_stats,
        },
    )
    summary = {
        "method": manifest["method"],
        "output_dir": str(output_dir),
        "selected_feature_size": int(selected_feature_indices.size),
        "counts": manifest["counts"],
        "retention_ratio_of_active": manifest["retention_ratio_of_active"],
        "compute": manifest["compute"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }
    print("[done] selection completed", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
