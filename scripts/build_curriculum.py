#!/usr/bin/env python3
"""Build a cluster-aware training order from calibrated difficulty scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from saerl.curriculum.order import (
    N_STAGES, STAGE_DECAY, UNLOCK_FRACTION, WINDOW_STAGES, build_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--ids-npy", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mix-tail", type=int, default=8)
    parser.add_argument("--window-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    predictions = pd.read_parquet(args.base_predictions)
    if args.ids_npy:
        selected_ids = np.load(args.ids_npy, allow_pickle=False)
        missing = np.setdiff1d(selected_ids, predictions.sample_id.to_numpy())
        if missing.size:
            raise ValueError(f"Selected sample ID {int(missing[0])} has no prediction")
        predictions = predictions[predictions.sample_id.isin(selected_ids)]
    order = build_order(
        predictions, batch_size=args.batch_size,
        mix_tail=args.mix_tail, window_batches=args.window_batches, seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order.to_parquet(args.output_dir / "training_order.parquet", index=False)
    torch.save(
        {"indices": order.sample_id.to_numpy(dtype=np.int64),
         "scores": order.ranking_score.to_numpy(dtype=np.float32),
         "meta": {"batch_size": args.batch_size, "mix_tail": args.mix_tail,
                  "window_batches": args.window_batches, "seed": args.seed,
                  "n_stages": N_STAGES, "window_stages": WINDOW_STAGES,
                  "stage_decay": STAGE_DECAY, "unlock_fraction": UNLOCK_FRACTION,
                  "tail_policy": "redistribute_then_repeat_within_cluster"}},
        args.output_dir / "curriculum.pt",
    )
    print(args.output_dir / "curriculum.pt")


if __name__ == "__main__":
    main()
