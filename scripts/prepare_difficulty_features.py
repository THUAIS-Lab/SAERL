#!/usr/bin/env python3
"""Build the prompt/solution SAE and metadata representation used by SAERL."""

from __future__ import annotations

import argparse
import json
import math
from importlib.resources import files
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


MAX_ID_OFFSET = 130915
SIDES = ("prompt", "sol")
BASE_COLUMNS = (
    "sample_id", "difficulty", "prompt_len_tok", "sol_len_tok", "full_len_tok",
    "latex_ratio", "digit_ratio", "sym_ratio", "was_truncated",
    "prompt_token_count", "sol_token_count",
)


def _metadata(row: dict) -> dict[str, float]:
    prompt_len = float(row["prompt_len_tok"])
    sol_len = float(row["sol_len_tok"])
    full_len = float(row["full_len_tok"])
    values = {
        "prompt_len_tok": prompt_len,
        "sol_len_tok": sol_len,
        "full_len_tok": full_len,
        "latex_ratio": float(row["latex_ratio"]),
        "digit_ratio": float(row["digit_ratio"]),
        "sym_ratio": float(row["sym_ratio"]),
        "was_truncated": float(bool(row["was_truncated"])),
        "log_prompt_len_tok": math.log1p(prompt_len),
        "log_sol_len_tok": math.log1p(sol_len),
        "log_full_len_tok": math.log1p(full_len),
    }
    for side in SIDES:
        means = row[f"{side}_feat_vals"] or []
        maxima = row[f"{side}_feat_max"] or []
        pooled = [*means, *maxima]
        sums = row[f"{side}_feat_sum"] or []
        counts = row[f"{side}_feat_count"] or []
        sum_total = float(sum(sums))
        count_total = float(sum(counts))
        token_count = max(float(row[f"{side}_token_count"]), 1.0)
        values.update({
            f"{side}_feat_mass": float(sum(pooled)),
            f"{side}_feat_max": float(max(pooled, default=0.0)),
            f"{side}_feat_n": float(len(pooled)),
            f"{side}_feat_sum_total": sum_total,
            f"{side}_feat_count_total": count_total,
            f"{side}_feat_mean_total": sum_total / max(count_total, 1.0),
            f"{side}_feat_density": len(pooled) / token_count,
            f"{side}_mass_per_token": sum_total / token_count,
        })
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-meanmax", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature-index", type=Path,
        default=Path(str(files("saerl.config").joinpath("feature_index.json"))),
    )
    args = parser.parse_args()

    index = json.loads(args.feature_index.read_text(encoding="utf-8"))
    sae_entries = sorted(index["sae_features"], key=lambda item: item["col"])
    meta_entries = sorted(index["meta_features"], key=lambda item: item["col"])
    sae_names = []
    feature_lookup = {}
    for entry in sae_entries:
        feature_id = int(entry["feature_id"])
        stat = "max" if feature_id >= MAX_ID_OFFSET else "mean"
        base_id = feature_id - MAX_ID_OFFSET if stat == "max" else feature_id
        name = f"f_{entry['side']}_{stat}_{base_id}"
        sae_names.append(name)
        feature_lookup[(entry["side"], stat, base_id)] = name
    meta_names = [entry["name"] for entry in meta_entries]
    feature_names = [*sae_names, *(f"f_meta_{name}" for name in meta_names)]

    source = pq.ParquetFile(args.samples_meanmax)
    columns = [*BASE_COLUMNS]
    for side in SIDES:
        columns.extend(f"{side}_feat_{part}" for part in ("ids", "vals", "max", "sum", "count"))
    schema = pa.schema(
        [("sample_id", pa.int64()), ("difficulty_true", pa.float64()),
         ("prompt_len_tok", pa.int64()), ("sol_len_tok", pa.int64()),
         ("full_len_tok", pa.int64())]
        + [(name, pa.float32()) for name in feature_names]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "difficulty_features.parquet"
    with pq.ParquetWriter(output, schema, compression="snappy") as writer:
        for batch in source.iter_batches(batch_size=256, columns=columns):
            rows = []
            for source_row in batch.to_pylist():
                row = {name: 0.0 for name in feature_names}
                row.update({
                    name: source_row[name]
                    for name in ("sample_id", "prompt_len_tok", "sol_len_tok", "full_len_tok")
                })
                row["difficulty_true"] = source_row["difficulty"]
                for side in SIDES:
                    ids = source_row[f"{side}_feat_ids"] or []
                    for stat, part in (("mean", "vals"), ("max", "max")):
                        for fid, value in zip(ids, source_row[f"{side}_feat_{part}"] or []):
                            name = feature_lookup.get((side, stat, int(fid)))
                            if name is not None:
                                row[name] = float(value)
                for name, value in _metadata(source_row).items():
                    row[f"f_meta_{name}"] = value
                rows.append(row)
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    print(output)


if __name__ == "__main__":
    main()
