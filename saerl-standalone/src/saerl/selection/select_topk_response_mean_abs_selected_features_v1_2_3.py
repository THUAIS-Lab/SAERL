#!/usr/bin/env python3
"""V1.2.3 cosine selection on absolute selected-feature response mean."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from saerl.selection.select_topk_response_mean_v1_2 import (  # noqa: E402
    LoadedRepresentations,
    _select_with_device,
)
from saerl.selection.selected_features import (  # noqa: E402
    default_output_dir,
    load_projection,
    persist_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--feature-indices", required=True)
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
    projection = load_projection(
        Path(args.run_dir),
        Path(args.feature_indices),
        input_glob=args.input_glob,
        read_batch_size=args.read_batch_size,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(projection, "selection_v1_2_3")
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")

    active = projection.active_rows
    normalized = np.abs(projection.values[active])
    np.divide(normalized, projection.strengths[active, None], out=normalized)
    loaded = LoadedRepresentations(
        feature_size=projection.feature_size,
        zero_eps=projection.zero_eps,
        sample_ids=projection.sample_ids,
        signal_strengths=projection.strengths,
        zero_signal_flags=projection.zero_flags,
        active_all_rows=active,
        active_sample_ids=projection.sample_ids[active],
        active_signal_strengths=projection.strengths[active],
        normalized_active=normalized,
        shard_metadata=projection.shard_metadata,
    )
    print(f"[abs] converted {active.size:,} active vectors to element-wise absolute values", flush=True)
    result, compute = _select_with_device(
        loaded.normalized_active,
        loaded.active_sample_ids,
        loaded.active_signal_strengths,
        cosine_threshold=args.cosine_threshold,
        block_size=args.similarity_block_size,
        requested_device=args.device,
    )
    manifest = persist_selection(
        output_dir,
        projection,
        method="selection_v1.2.3",
        decision_rows=active,
        selected=result.selected,
        selection_ranks=result.selection_ranks,
        threshold_name="cosine_threshold",
        threshold=args.cosine_threshold,
        score_name="representative_cosine",
        scores=result.representative_cosines,
        representatives=result.representative_sample_ids,
        upper_bounds=None,
        compute=compute,
        details={
            "transform": "element-wise absolute value after selected-feature projection",
            "similarity": "exact cosine between absolute selected-feature response-mean patterns",
        },
        elapsed_seconds=time.monotonic() - started,
        representation="absolute Token-TopK response mean projected to selected features",
        signal_strength_description="L2 norm of absolute selected-feature response-mean values",
    )
    print(json.dumps({"counts": manifest["counts"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
