"""Compare old and installed selection outputs on the same extraction artifacts."""

import argparse
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy-root", "run-dir", "feature-indices", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--cosine-threshold", default=0.7, type=float)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    legacy = (
        args.legacy_root.resolve()
        / "recipe/static_gradient_sae/selection_v1/select_topk_response_mean_abs_selected_features_v1_2_3.py"
    )
    common = [
        "--run-dir",
        str(args.run_dir.resolve()),
        "--feature-indices",
        str(args.feature_indices.resolve()),
        "--cosine-threshold",
        str(args.cosine_threshold),
        "--device",
        "cpu",
    ]
    subprocess.run(
        [sys.executable, str(legacy), *common, "--output-dir", str(output / "legacy")], check=True, cwd=output
    )
    subprocess.run(
        [sys.executable, "-m", "saerl.select", *common, "--output-dir", str(output / "standalone")],
        check=True,
        cwd=output,
    )
    for name in ("selected_sample_ids.parquet", "silent_sample_ids.parquet", "selection_audit.parquet"):
        old, new = (pq.read_table(output / side / name) for side in ("legacy", "standalone"))
        if not old.equals(new, check_metadata=True):
            raise AssertionError(f"Output differs: {name}")
    print("PASS: selected IDs, silent IDs, representatives, scores and audit schemas are identical")


if __name__ == "__main__":
    main()
