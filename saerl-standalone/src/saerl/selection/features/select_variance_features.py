#!/usr/bin/env python3
"""Select global Token-TopK response-mean features by normalized variance."""

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


def select(run_dir: Path, top_k: int) -> tuple[np.ndarray, int]:
    feature_size = int(json.loads((run_dir / "representation_manifest.json").read_text())["sae_feature_size"])
    value_sum = np.zeros(feature_size, dtype=np.float64)
    squared_sum = np.zeros(feature_size, dtype=np.float64)
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
    if not 1 <= top_k <= feature_size:
        raise ValueError(f"top_k must be in [1, {feature_size}]")
    variance = squared_sum / sample_count - (value_sum / sample_count) ** 2
    selected = np.argsort(-variance, kind="stable")[:top_k]
    return np.sort(selected.astype(np.int64)), sample_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output = args.output or run_dir / "feature_selection" / "variance_feature_indices.npy"
    output = output.expanduser().resolve()
    selected, sample_count = select(run_dir, args.top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, selected, allow_pickle=False)
    print(f"selected={selected.size} samples={sample_count} output={output}")


if __name__ == "__main__":
    main()
