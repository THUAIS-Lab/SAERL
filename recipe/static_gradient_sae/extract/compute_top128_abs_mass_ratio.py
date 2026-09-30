#!/usr/bin/env python3
"""Compute per-sample absolute-mass retention of the strongest 128 features.

The output contains one row for every sample x representation x span x
pooling combination. Token-TopK coordinates missing from storage are logical
zeros. ``max`` refers to the stored signed peak; taking absolute values removes
the sign. A zero vector has a null ratio because its ratio is 0/0.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .analyze_representations import (
        INPUT_COLUMNS,
        _dense_batch,
        _discover_inputs,
        _primitive_numpy,
        _ragged_column,
        _slice_ragged,
        _validate_manifest,
        _validate_sparse_span,
    )
except ImportError:
    from recipe.static_gradient_sae.extract.analyze_representations import (  # type: ignore[no-redef]
        INPUT_COLUMNS,
        _dense_batch,
        _discover_inputs,
        _primitive_numpy,
        _ragged_column,
        _slice_ragged,
        _validate_manifest,
        _validate_sparse_span,
    )


TOP_K = 256
CHANNELS = (
    ("prompt", "mean"),
    ("prompt", "max"),
    ("response", "mean"),
    ("response", "max"),
)


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("representation", pa.string()),
            ("span", pa.string()),
            ("pooling", pa.string()),
            ("top_k", pa.int32()),
            ("top128_abs_sum", pa.float64()),
            ("all_features_abs_sum", pa.float64()),
            ("top128_abs_mass_ratio", pa.float64()),
        ]
    )


def _mass_stats(values: np.ndarray) -> tuple[float, float, float | None]:
    absolute = np.abs(values).astype(np.float64, copy=False)
    total = float(np.sum(absolute, dtype=np.float64))
    if absolute.size <= TOP_K:
        top_sum = total
    else:
        top_sum = float(np.sum(np.partition(absolute, -TOP_K)[-TOP_K:], dtype=np.float64))
    return top_sum, total, top_sum / total if total > 0 else None


def _rows_for_representation(
    sample_id: int,
    representation: str,
    channel_values: tuple[np.ndarray, ...],
) -> list[dict[str, Any]]:
    rows = []
    for (span, pooling), values in zip(CHANNELS, channel_values, strict=True):
        top_sum, total, ratio = _mass_stats(values)
        rows.append(
            {
                "sample_id": sample_id,
                "representation": representation,
                "span": span,
                "pooling": pooling,
                "top_k": TOP_K,
                "top128_abs_sum": top_sum,
                "all_features_abs_sum": total,
                "top128_abs_mass_ratio": ratio,
            }
        )
    return rows


def compute_run(run_dir: str | Path, output: str | Path | None = None, *, overwrite: bool = False) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    manifest_path = run_dir / "representation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_size, _ = _validate_manifest(manifest, manifest_path)
    input_paths, _, _ = _discover_inputs(run_dir, "step-*.parquet")

    output = Path(output).expanduser().resolve() if output else run_dir / "top128_abs_mass_ratio.parquet"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.inprogress")

    writer: pq.ParquetWriter | None = None
    seen_sample_ids: set[int] = set()
    try:
        writer = pq.ParquetWriter(temporary, _schema(), compression="snappy", use_dictionary=True)
        for input_path in input_paths:
            parquet = pq.ParquetFile(input_path)
            for batch in parquet.iter_batches(batch_size=4, columns=list(INPUT_COLUMNS)):
                sample_ids = _primitive_numpy(
                    batch.column(batch.schema.get_field_index("sample_id")), dtype=np.int64
                )
                dense = _dense_batch(batch, feature_size)
                ragged = {
                    name: _ragged_column(
                        batch,
                        name,
                        np.int64 if name.endswith("_ids") else np.float32,
                    )
                    for name in INPUT_COLUMNS
                    if name.startswith("topk_")
                }
                output_rows = []
                for row_index in range(batch.num_rows):
                    sample_id = int(sample_ids[row_index])
                    if sample_id in seen_sample_ids:
                        raise ValueError(f"Duplicate sample_id across shards: {sample_id}")
                    seen_sample_ids.add(sample_id)

                    prompt_ids = _slice_ragged(*ragged["topk_prompt_feat_ids"], row_index)
                    prompt_mean = _slice_ragged(*ragged["topk_prompt_feat_mean"], row_index)
                    prompt_max = _slice_ragged(*ragged["topk_prompt_feat_signed_peak"], row_index)
                    response_ids = _slice_ragged(*ragged["topk_response_feat_ids"], row_index)
                    response_mean = _slice_ragged(*ragged["topk_response_feat_mean"], row_index)
                    response_max = _slice_ragged(*ragged["topk_response_feat_signed_peak"], row_index)
                    _validate_sparse_span(
                        sample_id, "prompt", prompt_ids, prompt_mean, prompt_max, feature_size
                    )
                    _validate_sparse_span(
                        sample_id, "response", response_ids, response_mean, response_max, feature_size
                    )

                    output_rows.extend(
                        _rows_for_representation(
                            sample_id,
                            "dense",
                            tuple(dense[row_index, channel] for channel in range(len(CHANNELS))),
                        )
                    )
                    output_rows.extend(
                        _rows_for_representation(
                            sample_id,
                            "token_topk",
                            (prompt_mean, prompt_max, response_mean, response_max),
                        )
                    )
                writer.write_table(pa.Table.from_pylist(output_rows, schema=_schema()))
        writer.close()
        writer = None
        os.replace(temporary, output)
        print(output)
        return output
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    compute_run(args.run_dir, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
