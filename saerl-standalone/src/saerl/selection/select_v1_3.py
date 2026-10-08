#!/usr/bin/env python3
"""V1.3: Dense response mean -> per-sample abs-TopK -> signed cosine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp

from saerl.selection.select_dense_response_mean_v1_2 import (
    INPUT_COLUMNS,
    _discover_dense_shards,
    _load_dense_manifest,
    _primitive_numpy,
    _ragged_numpy,
)
from saerl.selection.select_topk_response_mean_v1_2 import (
    LoadedRepresentations,
    _greedy_select_cpu,
    _write_outputs,
)


def _abs_topk(values: np.ndarray, k: int) -> np.ndarray:
    """Return feature-sorted abs-TopK ids; ties prefer smaller ids."""
    if k >= values.size:
        ids = np.arange(values.size)
    else:
        magnitude = np.abs(values)
        cutoff = np.partition(magnitude, values.size - k)[values.size - k]
        larger = np.flatnonzero(magnitude > cutoff)
        ids = np.concatenate((larger, np.flatnonzero(magnitude == cutoff)[: k - larger.size]))
    return np.sort(ids[values[ids] != 0.0])


def _load(run_dir: Path, input_glob: str, batch_size: int, top_k: int) -> LoadedRepresentations:
    feature_size, zero_eps = _load_dense_manifest(run_dir)
    paths, shard_metadata = _discover_dense_shards(run_dir, input_glob)
    top_k = min(top_k, feature_size)
    total_rows = sum(int(shard["rows"]) for shard in shard_metadata)
    print(
        f"Loading {len(paths):,} shards / {total_rows:,} samples (feature_size={feature_size:,}, top_k={top_k:,})",
        flush=True,
    )

    sample_ids: list[int] = []
    strengths: list[float] = []
    zero_flags: list[bool] = []
    active_rows: list[int] = []
    active_ids: list[int] = []
    active_strengths: list[float] = []
    csr_ids: list[np.ndarray] = []
    csr_values: list[np.ndarray] = []
    indptr = [0]

    for shard_number, path in enumerate(paths, start=1):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(INPUT_COLUMNS)):
            ids = _primitive_numpy(batch.column(0), np.int64)
            offsets, dense = _ragged_numpy(batch.column(1), np.float32)
            flags = _primitive_numpy(batch.column(2), np.bool_)
            for row, sample_id in enumerate(ids):
                start, stop = int(offsets[row]), int(offsets[row + 1])
                if stop - start != 4 * feature_size:
                    raise ValueError(
                        f"sample_id={sample_id} has Dense length {stop - start}, expected {4 * feature_size}"
                    )
                mean = dense[start + 2 * feature_size : start + 3 * feature_size]
                if not np.all(np.isfinite(mean)):
                    raise ValueError(f"sample_id={sample_id} response_mean contains NaN or infinity")
                feature_ids = _abs_topk(mean, top_k)
                selected = mean[feature_ids]
                norm = float(np.linalg.norm(selected.astype(np.float64, copy=False)))
                all_row = len(sample_ids)
                sample_ids.append(int(sample_id))
                strengths.append(norm)
                zero_flags.append(bool(flags[row]))
                if flags[row] or norm <= zero_eps:
                    continue
                csr_ids.append(feature_ids.astype(np.int64, copy=False))
                csr_values.append((selected / norm).astype(np.float32, copy=False))
                indptr.append(indptr[-1] + feature_ids.size)
                active_rows.append(all_row)
                active_ids.append(int(sample_id))
                active_strengths.append(norm)
        print(
            f"Loaded shard {shard_number:,}/{len(paths):,}: {path.name}; "
            f"processed {len(sample_ids):,}/{total_rows:,}, active {len(active_ids):,}",
            flush=True,
        )

    ids_array = np.asarray(sample_ids, dtype=np.int64)
    if not ids_array.size:
        raise ValueError("No representation rows found")
    if np.unique(ids_array).size != ids_array.size:
        raise ValueError("Duplicate sample_id across input shards")
    indices = np.concatenate(csr_ids) if csr_ids else np.empty(0, dtype=np.int64)
    values = np.concatenate(csr_values) if csr_values else np.empty(0, dtype=np.float32)
    normalized = sp.csr_matrix(
        (values, indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(active_ids), feature_size),
        dtype=np.float32,
    )
    return LoadedRepresentations(
        feature_size=feature_size,
        zero_eps=zero_eps,
        sample_ids=ids_array,
        signal_strengths=np.asarray(strengths, dtype=np.float64),
        zero_signal_flags=np.asarray(zero_flags, dtype=np.bool_),
        active_all_rows=np.asarray(active_rows, dtype=np.int64),
        active_sample_ids=np.asarray(active_ids, dtype=np.int64),
        active_signal_strengths=np.asarray(active_strengths, dtype=np.float64),
        normalized_active=normalized,
        shard_metadata=shard_metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument("--read-batch-size", type=int, default=4)
    parser.add_argument("--similarity-block-size", type=int, default=64)
    args = parser.parse_args()
    if args.top_k < 1 or args.read_batch_size < 1 or args.similarity_block_size < 1:
        parser.error("top-k and block sizes must be positive")
    if not -1.0 <= args.cosine_threshold <= 1.0:
        parser.error("cosine-threshold must be in [-1,1]")

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "selection_v1_3"
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")

    started = time.monotonic()
    loaded = _load(run_dir, args.input_glob, args.read_batch_size, args.top_k)
    print(
        f"Starting cosine selection for {loaded.active_sample_ids.size:,} active samples "
        f"(threshold={args.cosine_threshold:g}, block_size={args.similarity_block_size:,})",
        flush=True,
    )
    result = _greedy_select_cpu(
        loaded.normalized_active,
        loaded.active_sample_ids,
        loaded.active_signal_strengths,
        cosine_threshold=args.cosine_threshold,
        block_size=args.similarity_block_size,
    )
    manifest = _write_outputs(
        output_dir,
        loaded,
        result,
        run_dir=run_dir,
        input_glob=args.input_glob,
        cosine_threshold=args.cosine_threshold,
        block_size=args.similarity_block_size,
        compute={"device": "cpu", "backend": "scipy_sparse"},
        method="selection_v1.3",
        input_columns=INPUT_COLUMNS,
        representation="sample-level Dense response mean, then per-sample abs-TopK",
        zero_signal_field="zero_signal_dense",
        signal_strength_description="L2 norm after per-sample abs-TopK",
        similarity_description="exact signed cosine after per-sample abs-TopK",
        silent_zero_reason="topk_response_mean_zero",
        elapsed_seconds=time.monotonic() - started,
        extra_manifest={
            "top_k": min(args.top_k, loaded.feature_size),
            "dense_source_slice": "[2F, 3F)",
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
