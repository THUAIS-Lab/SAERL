#!/usr/bin/env python3
"""Cosine selection on absolute raw response means in globally selected dimensions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from saerl.selection.raw.raw_gradient_common import (  # noqa: E402
    RAW_GRADIENT_COLUMNS,
    load_absolute_representations,
    load_dimension_indices,
    load_source,
)
from saerl.selection.select_topk_response_mean_v1_2 import (  # noqa: E402
    _hash_int64,
    _select_with_device,
    _write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--feature-indices", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--read-batch-size", type=int, default=64)
    parser.add_argument("--similarity-block-size", type=int, default=64)
    parser.add_argument(
        "--zero-eps",
        type=float,
        help="Override gradient_zero_eps (default: manifest value, or 1e-12 without a manifest)",
    )
    args = parser.parse_args()
    if not -1.0 <= args.cosine_threshold <= 1.0:
        parser.error("--cosine-threshold must be in [-1, 1]")
    if args.read_batch_size < 1 or args.similarity_block_size < 1:
        parser.error("batch and block sizes must be positive")

    started = time.monotonic()
    source = load_source(Path(args.run_dir), input_glob=args.input_glob, zero_eps=args.zero_eps)
    feature_indices_path = Path(args.feature_indices).expanduser().resolve()
    feature_indices = load_dimension_indices(feature_indices_path, source.response_mean_size)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else source.run_dir
        / (
            f"raw_gradient_response_mean_selection_abs_selected_{feature_indices_path.stem}"
            f"_threshold_{args.cosine_threshold:g}"
        )
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")

    loaded, projection_stats = load_absolute_representations(
        source,
        dimension_indices=feature_indices,
        read_batch_size=args.read_batch_size,
    )
    print(
        f"[abs] loaded {loaded.active_sample_ids.size:,} active vectors in "
        f"{feature_indices.size:,}/{source.response_mean_size:,} selected raw response-mean dimensions",
        flush=True,
    )
    result, compute = _select_with_device(
        loaded.normalized_active,
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
        run_dir=source.run_dir,
        input_glob=args.input_glob,
        cosine_threshold=args.cosine_threshold,
        block_size=args.similarity_block_size,
        compute=compute,
        method="raw_gradient_response_mean_baseline.abs_selected_features",
        input_columns=RAW_GRADIENT_COLUMNS,
        representation="absolute raw response mean projected to globally selected dimensions",
        zero_signal_field="zero_signal_raw_response_mean",
        elapsed_seconds=time.monotonic() - started,
        signal_strength_description="L2 norm of raw response-mean values in selected dimensions",
        similarity_description="exact cosine on absolute raw response-mean values in selected dimensions",
        silent_zero_reason="selected_raw_response_mean_dimensions_zero",
        extra_manifest={
            "transform": (
                "extract raw_grad_vals response_mean channel, select dimensions, "
                "then take element-wise absolute value and L2-normalize"
            ),
            "source_raw_gradient_representation_size": source.raw_gradient_size,
            "source_raw_response_mean_size": source.response_mean_size,
            "source_channel": "response_mean",
            "source_channel_index": 2,
            "source_slice": [2 * source.response_mean_size, 3 * source.response_mean_size],
            "selected_dimension_size": int(feature_indices.size),
            "feature_indices_file": str(feature_indices_path),
            "feature_indices_sha256": _hash_int64(feature_indices),
            "selected_dimension_indices": feature_indices.tolist(),
            "projection": projection_stats,
            "source_raw_gradient_representation_layout": (source.manifest or {}).get(
                "raw_gradient_representation_layout"
            ),
            "source_manifest": str(source.run_dir / "representation_manifest.json") if source.manifest else None,
        },
    )
    print(json.dumps({"counts": manifest["counts"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
