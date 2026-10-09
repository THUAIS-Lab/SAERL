"""Build a gradient core set with absolute response-mean exact cosine greedy selection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from saerl.selection.exact_cosine import select_with_device
from saerl.selection.projection import load_projection, persist_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--feature-indices", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--read-batch-size", type=int, default=1024)
    parser.add_argument("--similarity-block-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.cosine_threshold <= 1:
        parser.error("--cosine-threshold must be in [0, 1]")
    if args.read_batch_size < 1 or args.similarity_block_size < 1:
        parser.error("batch and block sizes must be positive")
    output_dir = (args.output_dir or args.run_dir[0] / "selection").expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")

    started = time.monotonic()
    projection = load_projection(args.run_dir, args.feature_indices, args.read_batch_size)
    active = projection.active_rows
    normalized = np.abs(projection.values[active])
    np.divide(normalized, projection.strengths[active, None], out=normalized)
    result, compute = select_with_device(
        normalized,
        projection.sample_ids[active],
        projection.strengths[active],
        cosine_threshold=args.cosine_threshold,
        block_size=args.similarity_block_size,
        requested_device=args.device,
    )
    manifest = persist_selection(
        output_dir, projection, result, args.cosine_threshold, compute, time.monotonic() - started
    )
    print(json.dumps({"counts": manifest["counts"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
