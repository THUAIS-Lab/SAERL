"""Choose one global SAE feature set with Normalized Energy Top-K."""

import argparse
import json
from pathlib import Path

import numpy as np

from saerl.selection.energy import select
from saerl.selection.io import hash_int64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dirs = [path.expanduser().resolve() for path in args.run_dir]
    output = (args.output or run_dirs[0] / "feature_selection" / "energy_feature_indices.npy").expanduser().resolve()
    if output.suffix != ".npy":
        parser.error("--output must end in .npy")
    metadata_path = output.with_suffix(".json")
    for path in (output, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite feature selection: {path}")
    indices, energy, sample_count = select(run_dirs, args.top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, indices, allow_pickle=False)
    metadata_path.write_text(
        json.dumps(
            {
                "method": "normalized_energy_top_k",
                "score": "sum_i g_ij**2 / sum_k g_ik**2 over nonzero response-mean samples",
                "run_dirs": [str(path) for path in run_dirs],
                "nonzero_samples": sample_count,
                "feature_indices": indices.tolist(),
                "normalized_energy": energy[indices].tolist(),
                "feature_indices_sha256": hash_int64(indices),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selected={indices.size} samples={sample_count} output={output}")


if __name__ == "__main__":
    main()
