#!/usr/bin/env python3
"""V1.2.4 selection on absolute Token-TopK response-mean patterns."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from .select_topk_response_mean_v1_2 import (
        INPUT_COLUMNS,
        _discover_shards,
        _load_manifest,
        _load_representations,
        _select_with_device,
        _write_outputs,
    )
except ImportError:
    from saerl.selection.select_topk_response_mean_v1_2 import (
        INPUT_COLUMNS,
        _discover_shards,
        _load_manifest,
        _load_representations,
        _select_with_device,
        _write_outputs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--read-batch-size", type=int, default=4)
    parser.add_argument("--similarity-block-size", type=int, default=64)
    args = parser.parse_args()
    if not -1.0 <= args.cosine_threshold <= 1.0:
        parser.error("--cosine-threshold must be in [-1, 1]")
    if args.read_batch_size < 1 or args.similarity_block_size < 1:
        parser.error("batch and block sizes must be positive")

    started = time.monotonic()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / "selection_v1_2_4"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")

    _manifest, feature_size, zero_eps = _load_manifest(run_dir)
    paths, shard_metadata = _discover_shards(run_dir, args.input_glob)
    loaded = _load_representations(
        paths,
        shard_metadata,
        feature_size=feature_size,
        zero_eps=zero_eps,
        read_batch_size=args.read_batch_size,
    )
    absolute_patterns = loaded.normalized_active.copy()
    np.abs(absolute_patterns.data, out=absolute_patterns.data)
    print(f"Converted {loaded.active_sample_ids.size:,} active sparse patterns to absolute values", flush=True)
    result, compute = _select_with_device(
        absolute_patterns,
        loaded.active_sample_ids,
        loaded.active_signal_strengths,
        cosine_threshold=args.cosine_threshold,
        block_size=args.similarity_block_size,
        requested_device=args.device,
    )
    manifest = _write_outputs(
        output_dir,
        loaded,
        result,
        run_dir=run_dir,
        input_glob=args.input_glob,
        cosine_threshold=args.cosine_threshold,
        block_size=args.similarity_block_size,
        compute=compute,
        method="selection_v1.2.4",
        input_columns=INPUT_COLUMNS,
        representation="absolute sample-level Token-TopK response mean",
        zero_signal_field="zero_signal_token_topk",
        elapsed_seconds=time.monotonic() - started,
        signal_strength_description="L2 norm of absolute Token-TopK response mean",
        similarity_description="exact cosine on absolute full Token-TopK response-mean patterns",
        extra_manifest={"transform": "element-wise absolute value before L2 normalization"},
    )
    summary = {
        "method": manifest["method"],
        "output_dir": str(output_dir),
        "counts": manifest["counts"],
        "compute": manifest["compute"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
