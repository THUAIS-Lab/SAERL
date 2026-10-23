#!/usr/bin/env python3
"""Map curriculum sample IDs to rows in the training Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-pt", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--output-pt", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.curriculum_pt, map_location="cpu", weights_only=False)
    train = pq.read_table(args.train_parquet, columns=["extra_info"])
    sample_ids = train.column("extra_info").combine_chunks().field("index").to_numpy()
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("Training extra_info.index must uniquely identify each sample")
    missing = np.setdiff1d(np.asarray(payload["indices"]), sample_ids)
    if missing.size:
        raise ValueError(f"Curriculum sample ID {int(missing[0])} is absent from the training Parquet")
    positions = {int(sample_id): row for row, sample_id in enumerate(sample_ids)}
    indices = torch.tensor([positions[int(sample_id)] for sample_id in payload["indices"]], dtype=torch.int64)
    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(indices, args.output_pt)
    print(args.output_pt)


if __name__ == "__main__":
    main()
