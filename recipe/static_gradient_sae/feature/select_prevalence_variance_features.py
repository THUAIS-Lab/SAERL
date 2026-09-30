#!/usr/bin/env python3
"""Select prevalent global Token-TopK response-mean features by normalized variance."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

COLUMNS = (
    "topk_response_feat_ids",
    "topk_response_feat_mean",
    "zero_signal_token_topk",
)


def select(
    run_dir: Path,
    top_k: int,
    salient_top_l: int,
    min_prevalence: float,
) -> tuple[np.ndarray, int, int]:
    feature_size = int(json.loads((run_dir / "representation_manifest.json").read_text())["sae_feature_size"])
    value_sum = np.zeros(feature_size, dtype=np.float64)
    squared_sum = np.zeros(feature_size, dtype=np.float64)
    prevalence_count = np.zeros(feature_size, dtype=np.int64)
    sample_count = 0
    paths = sorted(run_dir.glob("step-*.parquet"))
    total_samples = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    processed = 0

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS):
            ids_array, values_array = batch.column(0), batch.column(1)
            id_offsets = ids_array.offsets.to_numpy(zero_copy_only=False)
            value_offsets = values_array.offsets.to_numpy(zero_copy_only=False)
            ids = ids_array.values.to_numpy(zero_copy_only=False)
            values = values_array.values.to_numpy(zero_copy_only=False)
            zero_signal = batch.column(2).to_numpy(zero_copy_only=False)

            for row in range(batch.num_rows):
                if zero_signal[row]:
                    continue
                row_ids = ids[id_offsets[row] : id_offsets[row + 1]]
                row_values = values[value_offsets[row] : value_offsets[row + 1]].astype(np.float64)
                norm = np.linalg.norm(row_values)
                if norm == 0.0:
                    continue
                normalized = row_values / norm
                value_sum[row_ids] += normalized
                squared_sum[row_ids] += normalized**2
                nonzero = np.flatnonzero(row_values)
                order = np.argsort(-np.abs(row_values[nonzero]), kind="stable")[:salient_top_l]
                prevalence_count[row_ids[nonzero[order]]] += 1
                sample_count += 1
            processed += batch.num_rows
            print(
                f"\rProcessed {processed}/{total_samples} samples ({processed / total_samples:.1%})",
                end="",
                flush=True,
            )
    print()

    if sample_count == 0:
        raise ValueError("No nonzero Token-TopK response-mean samples found")
    if top_k < 1 or salient_top_l < 1 or not 0.0 <= min_prevalence <= 1.0:
        raise ValueError("top_k and salient_top_l must be positive; min_prevalence must be in [0, 1]")

    variance = squared_sum / sample_count - (value_sum / sample_count) ** 2
    candidates = np.flatnonzero(prevalence_count >= min_prevalence * sample_count)
    if candidates.size < top_k:
        raise ValueError(f"Only {candidates.size} features satisfy min_prevalence, fewer than top_k={top_k}")
    ranked = candidates[np.argsort(-variance[candidates], kind="stable")]
    return np.sort(ranked[:top_k].astype(np.int64)), sample_count, candidates.size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--salient-top-l", type=int, default=128)
    parser.add_argument("--min-prevalence", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output = args.output or run_dir / "feature_selection" / "prevalence_variance_feature_indices.npy"
    output = output.expanduser().resolve()
    selected, sample_count, candidate_count = select(
        run_dir, args.top_k, args.salient_top_l, args.min_prevalence
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, selected, allow_pickle=False)
    print(
        f"selected={selected.size} candidates={candidate_count} "
        f"samples={sample_count} output={output}"
    )


if __name__ == "__main__":
    main()
