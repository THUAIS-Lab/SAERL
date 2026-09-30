#!/usr/bin/env python3
"""Select global raw response-mean dimensions by sample-normalized energy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recipe.static_gradient_sae.grad_baseline.raw_gradient_common import (  # noqa: E402
    _ragged_float32,
    _sample_ids,
    load_source,
)


def select(
    run_dir: Path,
    top_k: int,
    *,
    input_glob: str = "step-*.parquet",
    read_batch_size: int = 64,
    zero_eps: float | None = None,
) -> tuple[np.ndarray, int, int]:
    """Return sorted Top-K raw response-mean coordinates and sample statistics."""
    if read_batch_size < 1:
        raise ValueError("read_batch_size must be positive")
    source = load_source(run_dir, input_glob=input_glob, zero_eps=zero_eps)
    if not 1 <= top_k <= source.response_mean_size:
        raise ValueError(f"top_k must be in [1, {source.response_mean_size}]")

    score = np.zeros(source.response_mean_size, dtype=np.float64)
    sample_count = 0
    total_samples = sum(int(item["rows"]) for item in source.shard_metadata)
    processed = 0

    for path in source.paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=["sample_id", "raw_grad_vals"],
            use_threads=True,
        ):
            sample_ids = _sample_ids(batch.column(0))
            offsets, flat_values = _ragged_float32(batch.column(1))
            for row in range(batch.num_rows):
                values = flat_values[int(offsets[row]) : int(offsets[row + 1])]
                if values.size != source.raw_gradient_size:
                    raise ValueError(
                        f"sample_id={int(sample_ids[row])} has raw_grad_vals length {values.size}, "
                        f"expected {source.raw_gradient_size}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"sample_id={int(sample_ids[row])} raw_grad_vals contains NaN or infinity")
                values64 = values[source.response_mean_slice].astype(np.float64, copy=False)
                norm_squared = float(np.dot(values64, values64))
                if norm_squared <= source.zero_eps**2:
                    continue
                score += values64**2 / norm_squared
                sample_count += 1
            processed += batch.num_rows
            print(
                f"\rProcessed {processed}/{total_samples} samples ({processed / total_samples:.1%})",
                end="",
                flush=True,
            )
    print()

    if sample_count == 0:
        raise ValueError("No nonzero raw response-mean samples found")
    selected = np.argsort(-score, kind="stable")[:top_k]
    return np.sort(selected.astype(np.int64)), sample_count, source.response_mean_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--read-batch-size", type=int, default=64)
    parser.add_argument(
        "--zero-eps",
        type=float,
        help="Override gradient_zero_eps (default: manifest value, or 1e-12 without a manifest)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output = args.output or run_dir / "raw_gradient_feature_selection" / "response_mean_energy_feature_indices.npy"
    output = output.expanduser().resolve()
    selected, sample_count, response_mean_size = select(
        run_dir,
        args.top_k,
        input_glob=args.input_glob,
        read_batch_size=args.read_batch_size,
        zero_eps=args.zero_eps,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, selected, allow_pickle=False)
    summary = {
        "selected": int(selected.size),
        "samples": sample_count,
        "raw_response_mean_size": response_mean_size,
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
