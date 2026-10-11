from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

RESPONSE_COLUMNS = (
    "sample_id",
    "topk_response_feat_ids",
    "topk_response_feat_mean",
    "zero_signal_token_topk",
)


@dataclass(frozen=True)
class ExtractionRuns:
    directories: list[Path]
    shards: list[Path]
    feature_size: int
    zero_eps: float
    row_count: int


def load_runs(run_dirs: list[Path]) -> ExtractionRuns:
    directories = [path.expanduser().resolve() for path in run_dirs]
    manifests = [json.loads((path / "representation_manifest.json").read_text()) for path in directories]
    basis_keys = (
        "sae_feature_size", "sae_checkpoint", "hookpoint", "gradient_feature_topk",
        "decoder_normalization", "gradient_zero_eps",
    )
    if any(
        manifest.get(key) != manifests[0].get(key)
        for manifest in manifests[1:]
        for key in basis_keys
    ):
        raise ValueError("Extraction runs must share the same SAE and projection settings")
    shards = [path for directory in directories for path in sorted(directory.glob("step-*.parquet"))]
    return ExtractionRuns(
        directories=directories,
        shards=shards,
        feature_size=int(manifests[0]["sae_feature_size"]),
        zero_eps=float(manifests[0].get("gradient_zero_eps", 1e-12)),
        row_count=sum(pq.ParquetFile(path).metadata.num_rows for path in shards),
    )


def iter_response_means(runs: ExtractionRuns, batch_size: int = 1024) -> Iterator[tuple]:
    for path in runs.shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(RESPONSE_COLUMNS)):
            ids_array, means_array = batch.column(1), batch.column(2)
            ids_offsets = ids_array.offsets.to_numpy(zero_copy_only=False)
            means_offsets = means_array.offsets.to_numpy(zero_copy_only=False)
            ids = ids_array.values.to_numpy(zero_copy_only=False)
            means = means_array.values.to_numpy(zero_copy_only=False)
            sample_ids = batch.column(0).to_numpy(zero_copy_only=False)
            zero_flags = batch.column(3).to_numpy(zero_copy_only=False)
            for row in range(batch.num_rows):
                yield (
                    int(sample_ids[row]),
                    ids[ids_offsets[row]:ids_offsets[row + 1]],
                    means[means_offsets[row]:means_offsets[row + 1]],
                    bool(zero_flags[row]),
                )


def select_features(run_dirs: list[Path], top_k: int) -> tuple[np.ndarray, np.ndarray, int]:
    runs = load_runs(run_dirs)
    if not 1 <= top_k <= runs.feature_size:
        raise ValueError(f"top_k must be in [1, {runs.feature_size}]")
    energy = np.zeros(runs.feature_size, dtype=np.float64)
    sample_count = 0
    for _, ids, means, zero_flag in iter_response_means(runs):
        values = means.astype(np.float64)
        norm_squared = float(np.dot(values, values))
        if zero_flag or norm_squared <= runs.zero_eps**2:
            continue
        energy[ids] += values**2 / norm_squared
        sample_count += 1
    if not sample_count:
        raise ValueError("No nonzero response-mean samples found")
    indices = np.sort(np.argsort(-energy, kind="stable")[:top_k].astype(np.int64))
    return indices, energy, sample_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Select global shared SAE features with Normalized Energy Top-K.")
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dirs = [path.expanduser().resolve() for path in args.run_dir]
    output = (args.output or run_dirs[0] / "selection/feature_indices.npy").expanduser().resolve()
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    indices, _, sample_count = select_features(run_dirs, args.top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, indices, allow_pickle=False)
    print(f"selected={indices.size} samples={sample_count} output={output}")


if __name__ == "__main__":
    main()
