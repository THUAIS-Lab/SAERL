"""Select global shared SAE features with Normalized Energy Top-K."""

import argparse
import json
from pathlib import Path

import numpy as np

from saerl.selection.representations import iter_response_means, load_runs


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
    # Stable ties choose smaller feature IDs; all samples share this feature set.
    indices = np.sort(np.argsort(-energy, kind="stable")[:top_k].astype(np.int64))
    return indices, energy, sample_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dirs = [path.expanduser().resolve() for path in args.run_dir]
    output = (args.output or run_dirs[0] / "feature_selection/energy_feature_indices.npy").expanduser().resolve()
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    indices, energy, sample_count = select_features(run_dirs, args.top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, indices, allow_pickle=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "normalized_energy_top_k",
                "score": "sum_i g_ij**2 / sum_k g_ik**2 over nonzero response-mean samples",
                "run_dirs": [str(path) for path in run_dirs],
                "nonzero_samples": sample_count,
                "feature_indices": indices.tolist(),
                "normalized_energy": energy[indices].tolist(),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"selected={indices.size} samples={sample_count} output={output}")


if __name__ == "__main__":
    main()
