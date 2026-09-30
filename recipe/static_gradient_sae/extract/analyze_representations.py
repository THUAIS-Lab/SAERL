#!/usr/bin/env python3
"""Analyze V2 dense and token-TopK sample-level gradient-SAE representations.

The analyzer streams extraction shards and produces:

* ``sample_metrics.parquet``: per-sample/per-channel vector statistics, the
  strongest coordinates, and dense-vs-TopK agreement metrics;
* ``feature_metrics.parquet``: global statistics for every SAE feature in
  every representation/channel;
* ``feature_rankings.parquet``: reproducible global feature rankings;
* ``global_summary.json`` and ``summary.md``: dataset-level summaries;
* ``analysis_manifest.json``: source provenance and analyzer settings.
* ``visualization_report.html`` plus PNG/PDF diagnostics: joint-PCA geometry,
  dense-vs-TopK fidelity, global feature usage, and representative samples.

The sparse representation is never materialized as an N x 4F dense matrix.
Missing token-TopK coordinates are treated as logical zeros throughout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .representation_visualization import (
        VISUAL_OUTPUT_FILENAMES,
        GeometryReservoir,
        write_visualizations,
    )
except ImportError:
    from recipe.static_gradient_sae.extract.representation_visualization import (  # type: ignore[no-redef]
        VISUAL_OUTPUT_FILENAMES,
        GeometryReservoir,
        write_visualizations,
    )

ANALYSIS_SCHEMA_VERSION = 1
CHANNELS = (
    "prompt_mean",
    "prompt_signed_peak",
    "response_mean",
    "response_signed_peak",
)
SCOPES = ("all", *CHANNELS)
REPRESENTATIONS = ("dense", "token_topk")
VECTOR_INTEGER_METRICS = ("nnz", "positive_count", "negative_count")
VECTOR_FLOAT_METRICS = (
    "density",
    "signed_mean",
    "std",
    "mean_abs",
    "l1_norm",
    "l2_norm",
    "rms",
    "linf",
)
COMPARISON_FLOAT_METRICS = (
    "dot_product",
    "cosine_similarity",
    "relative_l2_error",
    "l1_retention",
    "l2_retention",
    "dense_l2_energy_on_topk_support_rate",
    "sign_agreement_rate",
)
REQUIRED_TYPES = {
    "sample_id": pa.int64(),
    "prompt_len_tok": pa.int64(),
    "dense_grad_feat_vals": pa.list_(pa.float32()),
    "topk_prompt_feat_ids": pa.list_(pa.int64()),
    "topk_prompt_feat_mean": pa.list_(pa.float32()),
    "topk_prompt_feat_signed_peak": pa.list_(pa.float32()),
    "topk_response_feat_ids": pa.list_(pa.int64()),
    "topk_response_feat_mean": pa.list_(pa.float32()),
    "topk_response_feat_signed_peak": pa.list_(pa.float32()),
    "zero_signal_dense": pa.bool_(),
    "zero_signal_token_topk": pa.bool_(),
}
INPUT_COLUMNS = tuple(REQUIRED_TYPES)
BASE_OUTPUT_FILENAMES = (
    "sample_metrics.parquet",
    "feature_metrics.parquet",
    "feature_rankings.parquet",
    "global_summary.json",
    "summary.md",
    "analysis_manifest.json",
)


@dataclass
class RawVectorStats:
    """Additive statistics used for one vector scope."""

    nnz: int
    positive_count: int
    negative_count: int
    signed_sum: float
    abs_sum: float
    squared_sum: float
    linf: float
    size: int

    @classmethod
    def from_values(cls, values: np.ndarray, logical_size: int | None = None) -> RawVectorStats:
        values = np.asarray(values)
        size = int(values.size if logical_size is None else logical_size)
        absolute = np.abs(values)
        return cls(
            nnz=int(np.count_nonzero(values)),
            positive_count=int(np.count_nonzero(values > 0)),
            negative_count=int(np.count_nonzero(values < 0)),
            signed_sum=float(np.sum(values, dtype=np.float64)),
            abs_sum=float(np.sum(absolute, dtype=np.float64)),
            squared_sum=float(np.dot(values.astype(np.float64, copy=False), values.astype(np.float64, copy=False))),
            linf=float(np.max(absolute)) if values.size else 0.0,
            size=size,
        )

    @classmethod
    def combine(cls, values: Iterable[RawVectorStats]) -> RawVectorStats:
        items = tuple(values)
        return cls(
            nnz=sum(item.nnz for item in items),
            positive_count=sum(item.positive_count for item in items),
            negative_count=sum(item.negative_count for item in items),
            signed_sum=sum(item.signed_sum for item in items),
            abs_sum=sum(item.abs_sum for item in items),
            squared_sum=sum(item.squared_sum for item in items),
            linf=max((item.linf for item in items), default=0.0),
            size=sum(item.size for item in items),
        )

    def materialize(self, zero_eps: float) -> dict[str, int | float | bool]:
        mean = self.signed_sum / self.size
        mean_square = self.squared_sum / self.size
        variance = max(mean_square - mean * mean, 0.0)
        l2_norm = math.sqrt(max(self.squared_sum, 0.0))
        return {
            "nnz": self.nnz,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "density": self.nnz / self.size,
            "signed_mean": mean,
            "std": math.sqrt(variance),
            "mean_abs": self.abs_sum / self.size,
            "l1_norm": self.abs_sum,
            "l2_norm": l2_norm,
            "rms": math.sqrt(max(mean_square, 0.0)),
            "linf": self.linf,
            "vector_is_zero": l2_norm < zero_eps,
        }


class FeatureAccumulator:
    """Streaming per-feature moments for four logical channels."""

    def __init__(self, feature_size: int, representation: str) -> None:
        shape = (len(CHANNELS), int(feature_size))
        self.representation = representation
        self.feature_size = int(feature_size)
        self.signed_sum = np.zeros(shape, dtype=np.float64)
        self.abs_sum = np.zeros(shape, dtype=np.float64)
        self.squared_sum = np.zeros(shape, dtype=np.float64)
        self.support_count = np.zeros(shape, dtype=np.int64)
        self.nonzero_count = np.zeros(shape, dtype=np.int64)
        self.positive_count = np.zeros(shape, dtype=np.int64)
        self.negative_count = np.zeros(shape, dtype=np.int64)
        self.minimum = np.full(shape, np.inf, dtype=np.float32)
        self.maximum = np.full(shape, -np.inf, dtype=np.float32)

    def update_dense_batch(self, dense: np.ndarray) -> None:
        """Update from [batch, 4, F] float32 values."""
        if dense.ndim != 3 or dense.shape[1:] != (len(CHANNELS), self.feature_size):
            raise ValueError(f"Unexpected dense batch shape: {dense.shape}")
        self.signed_sum += np.sum(dense, axis=0, dtype=np.float64)
        absolute = np.abs(dense)
        self.abs_sum += np.sum(absolute, axis=0, dtype=np.float64)
        self.squared_sum += np.einsum("bcf,bcf->cf", dense, dense, dtype=np.float64, casting="unsafe")
        self.support_count += dense.shape[0]
        self.nonzero_count += np.count_nonzero(dense, axis=0)
        self.positive_count += np.count_nonzero(dense > 0, axis=0)
        self.negative_count += np.count_nonzero(dense < 0, axis=0)
        self.minimum = np.minimum(self.minimum, np.min(dense, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(dense, axis=0))

    def update_sparse(self, channel: int, ids: np.ndarray, values: np.ndarray) -> None:
        """Update one sample/channel; ids must be unique within the row."""
        if ids.size == 0:
            return
        self.support_count[channel, ids] += 1
        self.signed_sum[channel, ids] += values
        absolute = np.abs(values)
        self.abs_sum[channel, ids] += absolute
        self.squared_sum[channel, ids] += values.astype(np.float64, copy=False) ** 2
        nonzero = values != 0
        self.nonzero_count[channel, ids[nonzero]] += 1
        self.positive_count[channel, ids[values > 0]] += 1
        self.negative_count[channel, ids[values < 0]] += 1
        self.minimum[channel, ids] = np.minimum(self.minimum[channel, ids], values)
        self.maximum[channel, ids] = np.maximum(self.maximum[channel, ids], values)


def _sample_schema() -> pa.Schema:
    fields: list[tuple[str, pa.DataType]] = [
        ("source_shard", pa.string()),
        ("source_row_index", pa.int64()),
        ("sample_id", pa.int64()),
        ("prompt_len_tok", pa.int64()),
        ("zero_signal_dense", pa.bool_()),
        ("zero_signal_token_topk", pa.bool_()),
        ("topk_prompt_feature_union_size", pa.int64()),
        ("topk_response_feature_union_size", pa.int64()),
        ("topk_prompt_response_feature_intersection_size", pa.int64()),
        ("topk_prompt_response_feature_jaccard", pa.float64()),
    ]
    for representation in REPRESENTATIONS:
        for scope in SCOPES:
            prefix = f"{representation}_{scope}"
            fields.extend((f"{prefix}_{metric}", pa.int64()) for metric in VECTOR_INTEGER_METRICS)
            fields.extend((f"{prefix}_{metric}", pa.float64()) for metric in VECTOR_FLOAT_METRICS)
            fields.append((f"{prefix}_vector_is_zero", pa.bool_()))
    for scope in SCOPES:
        prefix = f"dense_token_topk_{scope}"
        fields.extend((f"{prefix}_{metric}", pa.float64()) for metric in COMPARISON_FLOAT_METRICS)
    for representation in REPRESENTATIONS:
        fields.extend(
            [
                (f"{representation}_top_abs_coordinates", pa.list_(pa.int64())),
                (f"{representation}_top_abs_feature_ids", pa.list_(pa.int64())),
                (f"{representation}_top_abs_channel_ids", pa.list_(pa.int8())),
                (f"{representation}_top_abs_values", pa.list_(pa.float32())),
            ]
        )
    return pa.schema(fields)


def _feature_schema() -> pa.Schema:
    return pa.schema(
        [
            ("representation", pa.string()),
            ("channel", pa.string()),
            ("feature_id", pa.int64()),
            ("sample_count", pa.int64()),
            ("support_sample_count", pa.int64()),
            ("nonzero_sample_count", pa.int64()),
            ("positive_sample_count", pa.int64()),
            ("negative_sample_count", pa.int64()),
            ("support_rate", pa.float64()),
            ("nonzero_rate", pa.float64()),
            ("signed_sum", pa.float64()),
            ("l1_sum", pa.float64()),
            ("squared_l2_sum", pa.float64()),
            ("mean_including_zeros", pa.float64()),
            ("std_including_zeros", pa.float64()),
            ("mean_abs_including_zeros", pa.float64()),
            ("rms_including_zeros", pa.float64()),
            ("min_including_zeros", pa.float32()),
            ("max_including_zeros", pa.float32()),
        ]
    )


def _ranking_schema() -> pa.Schema:
    return pa.schema(
        [
            ("representation", pa.string()),
            ("scope", pa.string()),
            ("ranking_metric", pa.string()),
            ("rank", pa.int32()),
            ("feature_id", pa.int64()),
            ("score", pa.float64()),
        ]
    )


def _validate_manifest(manifest: dict[str, Any], path: Path) -> tuple[int, float]:
    feature_size = int(manifest.get("sae_feature_size", 0))
    if feature_size < 1:
        raise ValueError(f"Invalid sae_feature_size in {path}: {feature_size}")
    expected_size = 4 * feature_size
    for key in ("dense_representation_size", "topk_representation_size"):
        if int(manifest.get(key, -1)) != expected_size:
            raise ValueError(f"{key} in {path} must be 4F={expected_size}, got {manifest.get(key)!r}")
    if tuple(manifest.get("representation_layout", ())) != CHANNELS:
        raise ValueError(
            f"Unsupported representation_layout in {path}: {manifest.get('representation_layout')!r}; "
            f"expected {list(CHANNELS)!r}"
        )
    zero_eps = float(manifest.get("gradient_zero_eps", 1e-12))
    if not math.isfinite(zero_eps) or zero_eps < 0:
        raise ValueError(f"Invalid gradient_zero_eps in {path}: {zero_eps}")
    return feature_size, zero_eps


def _validate_input_schema(schema: pa.Schema, path: Path) -> None:
    for name, expected_type in REQUIRED_TYPES.items():
        field_index = schema.get_field_index(name)
        if field_index < 0:
            raise ValueError(f"Missing required column {name!r} in {path}")
        actual_type = schema.field(field_index).type
        if actual_type != expected_type:
            raise TypeError(f"Column {name!r} in {path} has type {actual_type}, expected {expected_type}")


def _discover_inputs(run_dir: Path, input_glob: str) -> tuple[list[Path], pa.Schema, list[dict[str, Any]]]:
    paths = sorted(
        path for path in run_dir.glob(input_glob) if path.is_file() and not path.name.endswith(".inprogress")
    )
    if not paths:
        raise FileNotFoundError(f"No input shards match {input_glob!r} under {run_dir}")
    first_schema = pq.ParquetFile(paths[0]).schema_arrow
    _validate_input_schema(first_schema, paths[0])
    metadata: list[dict[str, Any]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(first_schema):
            raise ValueError(f"Parquet schema mismatch: {path}")
        metadata.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "rows": parquet.metadata.num_rows,
                "row_groups": parquet.metadata.num_row_groups,
            }
        )
    return paths, first_schema, metadata


def _primitive_numpy(array: pa.Array, *, dtype: np.dtype | None = None) -> np.ndarray:
    if array.null_count:
        raise ValueError("Null values are not supported in representation columns")
    values = array.to_numpy(zero_copy_only=False)
    return values.astype(dtype, copy=False) if dtype is not None else values


def _ragged_numpy(array: pa.Array, *, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    if array.null_count:
        raise ValueError("Null lists are not supported in representation columns")
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise TypeError(f"Expected a list array, got {array.type}")
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if array.values.null_count:
        raise ValueError("Null elements are not supported in representation lists")
    values = array.values.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    return offsets, values


def _dense_batch(batch: pa.RecordBatch, feature_size: int) -> np.ndarray:
    array = batch.column(batch.schema.get_field_index("dense_grad_feat_vals"))
    offsets, values = _ragged_numpy(array, dtype=np.float32)
    expected = 4 * int(feature_size)
    lengths = np.diff(offsets)
    if np.any(lengths != expected):
        bad_row = int(np.flatnonzero(lengths != expected)[0])
        raise ValueError(f"Dense representation row {bad_row} has length {lengths[bad_row]}, expected {expected}")
    start, stop = int(offsets[0]), int(offsets[-1])
    return values[start:stop].reshape(batch.num_rows, len(CHANNELS), feature_size)


def _ragged_column(batch: pa.RecordBatch, name: str, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    return _ragged_numpy(batch.column(batch.schema.get_field_index(name)), dtype=dtype)


def _slice_ragged(offsets: np.ndarray, values: np.ndarray, row: int) -> np.ndarray:
    return values[int(offsets[row]) : int(offsets[row + 1])]


def _validate_sparse_span(
    sample_id: int,
    span: str,
    ids: np.ndarray,
    first_values: np.ndarray,
    second_values: np.ndarray,
    feature_size: int,
) -> None:
    if ids.size != first_values.size or ids.size != second_values.size:
        raise ValueError(
            f"sample_id={sample_id} {span} sparse arrays have inconsistent lengths: "
            f"ids={ids.size}, first={first_values.size}, second={second_values.size}"
        )
    if ids.size and (ids[0] < 0 or ids[-1] >= feature_size):
        raise ValueError(f"sample_id={sample_id} {span} feature ids fall outside [0, {feature_size})")
    if ids.size > 1 and np.any(ids[1:] <= ids[:-1]):
        raise ValueError(f"sample_id={sample_id} {span} feature ids must be strictly increasing")
    if not np.all(np.isfinite(first_values)) or not np.all(np.isfinite(second_values)):
        raise ValueError(f"sample_id={sample_id} {span} contains NaN or infinity")


def _put_vector_stats(
    output: dict[str, Any],
    representation: str,
    scope: str,
    stats: RawVectorStats,
    zero_eps: float,
) -> None:
    prefix = f"{representation}_{scope}"
    for metric, value in stats.materialize(zero_eps).items():
        output[f"{prefix}_{metric}"] = value


def _safe_ratio(numerator: float, denominator: float, *, both_zero_value: float = math.nan) -> float:
    if denominator != 0:
        return numerator / denominator
    return both_zero_value if numerator == 0 else math.nan


def _comparison_metrics(
    dense_stats: RawVectorStats,
    topk_stats: RawVectorStats,
    dot_product: float,
    l1_error: float,
    dense_support_squared_sum: float,
    sign_matches: int,
    sign_comparisons: int,
) -> dict[str, float]:
    dense_l2 = math.sqrt(max(dense_stats.squared_sum, 0.0))
    topk_l2 = math.sqrt(max(topk_stats.squared_sum, 0.0))
    residual_squared = max(dense_stats.squared_sum + topk_stats.squared_sum - 2.0 * dot_product, 0.0)
    residual_l2 = math.sqrt(residual_squared)
    cosine_denominator = dense_l2 * topk_l2
    return {
        "dot_product": dot_product,
        "cosine_similarity": dot_product / cosine_denominator if cosine_denominator > 0 else math.nan,
        "relative_l2_error": _safe_ratio(residual_l2, dense_l2, both_zero_value=0.0),
        "l1_retention": _safe_ratio(topk_stats.abs_sum, dense_stats.abs_sum),
        "l2_retention": _safe_ratio(topk_l2, dense_l2),
        "dense_l2_energy_on_topk_support_rate": _safe_ratio(math.sqrt(max(dense_support_squared_sum, 0.0)), dense_l2),
        "sign_agreement_rate": sign_matches / sign_comparisons if sign_comparisons else math.nan,
    }


def _top_coordinates(
    coordinates: np.ndarray,
    values: np.ndarray,
    feature_size: int,
    top_n: int,
) -> tuple[list[int], list[int], list[int], list[float]]:
    nonzero = np.flatnonzero(values != 0)
    if nonzero.size == 0 or top_n == 0:
        return [], [], [], []
    keep = min(int(top_n), int(nonzero.size))
    candidate_positions = nonzero
    if candidate_positions.size > keep:
        magnitudes = np.abs(values[candidate_positions])
        candidate_positions = candidate_positions[np.argpartition(magnitudes, -keep)[-keep:]]
    candidate_coordinates = coordinates[candidate_positions]
    candidate_values = values[candidate_positions]
    order = np.lexsort((candidate_coordinates, -np.abs(candidate_values)))
    selected_coordinates = candidate_coordinates[order]
    selected_values = candidate_values[order]
    return (
        selected_coordinates.astype(np.int64, copy=False).tolist(),
        (selected_coordinates % feature_size).astype(np.int64, copy=False).tolist(),
        (selected_coordinates // feature_size).astype(np.int8, copy=False).tolist(),
        selected_values.astype(np.float32, copy=False).tolist(),
    )


def _build_sample_row(
    *,
    source_shard: str,
    source_row_index: int,
    sample_id: int,
    prompt_len_tok: int,
    zero_signal_dense: bool,
    zero_signal_token_topk: bool,
    dense: np.ndarray,
    prompt_ids: np.ndarray,
    prompt_mean: np.ndarray,
    prompt_peak: np.ndarray,
    response_ids: np.ndarray,
    response_mean: np.ndarray,
    response_peak: np.ndarray,
    feature_size: int,
    zero_eps: float,
    top_n: int,
) -> tuple[dict[str, Any], tuple[np.ndarray, ...], tuple[RawVectorStats, ...]]:
    sparse_ids = (prompt_ids, prompt_ids, response_ids, response_ids)
    sparse_values = (prompt_mean, prompt_peak, response_mean, response_peak)
    dense_stats = tuple(RawVectorStats.from_values(dense[channel]) for channel in range(len(CHANNELS)))
    topk_stats = tuple(
        RawVectorStats.from_values(sparse_values[channel], logical_size=feature_size)
        for channel in range(len(CHANNELS))
    )
    dense_all = RawVectorStats.combine(dense_stats)
    topk_all = RawVectorStats.combine(topk_stats)

    intersection_size = int(np.intersect1d(prompt_ids, response_ids, assume_unique=True).size)
    union_size = int(prompt_ids.size + response_ids.size - intersection_size)
    row: dict[str, Any] = {
        "source_shard": source_shard,
        "source_row_index": int(source_row_index),
        "sample_id": int(sample_id),
        "prompt_len_tok": int(prompt_len_tok),
        "zero_signal_dense": bool(zero_signal_dense),
        "zero_signal_token_topk": bool(zero_signal_token_topk),
        "topk_prompt_feature_union_size": int(prompt_ids.size),
        "topk_response_feature_union_size": int(response_ids.size),
        "topk_prompt_response_feature_intersection_size": intersection_size,
        "topk_prompt_response_feature_jaccard": intersection_size / union_size if union_size else math.nan,
    }
    for channel, scope in enumerate(CHANNELS):
        _put_vector_stats(row, "dense", scope, dense_stats[channel], zero_eps)
        _put_vector_stats(row, "token_topk", scope, topk_stats[channel], zero_eps)
    _put_vector_stats(row, "dense", "all", dense_all, zero_eps)
    _put_vector_stats(row, "token_topk", "all", topk_all, zero_eps)

    comparison_parts: list[tuple[float, float, float, int, int]] = []
    for channel, scope in enumerate(CHANNELS):
        ids = sparse_ids[channel]
        values = sparse_values[channel]
        dense_on_support = dense[channel, ids]
        dot_product = float(np.dot(dense_on_support.astype(np.float64), values.astype(np.float64)))
        dense_support_abs = float(np.sum(np.abs(dense_on_support), dtype=np.float64))
        l1_error = (
            dense_stats[channel].abs_sum
            - dense_support_abs
            + float(np.sum(np.abs(dense_on_support - values), dtype=np.float64))
        )
        dense_support_squared_sum = float(
            np.dot(dense_on_support.astype(np.float64), dense_on_support.astype(np.float64))
        )
        sign_mask = (dense_on_support != 0) & (values != 0)
        sign_comparisons = int(np.count_nonzero(sign_mask))
        sign_matches = int(np.count_nonzero(np.signbit(dense_on_support[sign_mask]) == np.signbit(values[sign_mask])))
        comparison_parts.append((dot_product, l1_error, dense_support_squared_sum, sign_matches, sign_comparisons))
        metrics = _comparison_metrics(
            dense_stats[channel],
            topk_stats[channel],
            dot_product,
            l1_error,
            dense_support_squared_sum,
            sign_matches,
            sign_comparisons,
        )
        row.update({f"dense_token_topk_{scope}_{key}": value for key, value in metrics.items()})

    all_comparison = tuple(sum(item[index] for item in comparison_parts) for index in range(5))
    metrics = _comparison_metrics(dense_all, topk_all, *all_comparison)
    row.update({f"dense_token_topk_all_{key}": value for key, value in metrics.items()})

    dense_flat = dense.reshape(-1)
    dense_coordinates = np.arange(dense_flat.size, dtype=np.int64)
    topk_coordinates = np.concatenate(
        [
            prompt_ids,
            feature_size + prompt_ids,
            2 * feature_size + response_ids,
            3 * feature_size + response_ids,
        ]
    )
    topk_values = np.concatenate(sparse_values)
    for representation, coordinates, values in (
        ("dense", dense_coordinates, dense_flat),
        ("token_topk", topk_coordinates, topk_values),
    ):
        coords, feature_ids, channel_ids, selected_values = _top_coordinates(coordinates, values, feature_size, top_n)
        row[f"{representation}_top_abs_coordinates"] = coords
        row[f"{representation}_top_abs_feature_ids"] = feature_ids
        row[f"{representation}_top_abs_channel_ids"] = channel_ids
        row[f"{representation}_top_abs_values"] = selected_values
    return row, sparse_values, topk_stats


def _feature_columns(
    accumulator: FeatureAccumulator,
    channel: int,
    sample_count: int,
) -> dict[str, np.ndarray | list[str]]:
    signed_sum = accumulator.signed_sum[channel]
    abs_sum = accumulator.abs_sum[channel]
    squared_sum = accumulator.squared_sum[channel]
    support_count = accumulator.support_count[channel]
    mean = signed_sum / sample_count
    variance = np.maximum(squared_sum / sample_count - mean**2, 0.0)
    minimum = accumulator.minimum[channel].copy()
    maximum = accumulator.maximum[channel].copy()
    implicit_zeros = support_count < sample_count
    minimum[implicit_zeros] = np.minimum(minimum[implicit_zeros], 0.0)
    maximum[implicit_zeros] = np.maximum(maximum[implicit_zeros], 0.0)
    never_supported = support_count == 0
    minimum[never_supported] = 0.0
    maximum[never_supported] = 0.0
    return {
        "representation": [accumulator.representation] * accumulator.feature_size,
        "channel": [CHANNELS[channel]] * accumulator.feature_size,
        "feature_id": np.arange(accumulator.feature_size, dtype=np.int64),
        "sample_count": np.full(accumulator.feature_size, sample_count, dtype=np.int64),
        "support_sample_count": support_count,
        "nonzero_sample_count": accumulator.nonzero_count[channel],
        "positive_sample_count": accumulator.positive_count[channel],
        "negative_sample_count": accumulator.negative_count[channel],
        "support_rate": support_count / sample_count,
        "nonzero_rate": accumulator.nonzero_count[channel] / sample_count,
        "signed_sum": signed_sum,
        "l1_sum": abs_sum,
        "squared_l2_sum": squared_sum,
        "mean_including_zeros": mean,
        "std_including_zeros": np.sqrt(variance),
        "mean_abs_including_zeros": abs_sum / sample_count,
        "rms_including_zeros": np.sqrt(squared_sum / sample_count),
        "min_including_zeros": minimum,
        "max_including_zeros": maximum,
    }


def _rank_indices(scores: np.ndarray, count: int) -> np.ndarray:
    feature_ids = np.arange(scores.size, dtype=np.int64)
    finite_scores = np.nan_to_num(scores, nan=-np.inf)
    return np.lexsort((feature_ids, -finite_scores))[: min(count, scores.size)]


def _write_feature_outputs(
    feature_path: Path,
    ranking_path: Path,
    accumulators: dict[str, FeatureAccumulator],
    sample_count: int,
    ranking_top_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_writer = pq.ParquetWriter(feature_path, _feature_schema(), compression="snappy")
    ranking_records: list[dict[str, Any]] = []
    feature_summary: dict[str, Any] = {}
    try:
        for representation in REPRESENTATIONS:
            accumulator = accumulators[representation]
            representation_summary: dict[str, Any] = {}
            for channel_index, channel in enumerate(CHANNELS):
                columns = _feature_columns(accumulator, channel_index, sample_count)
                feature_writer.write_table(pa.Table.from_pydict(columns, schema=_feature_schema()))
                representation_summary[channel] = {
                    "ever_supported_features": int(np.count_nonzero(accumulator.support_count[channel_index])),
                    "ever_nonzero_features": int(np.count_nonzero(accumulator.nonzero_count[channel_index])),
                    "total_l1_mass": float(np.sum(accumulator.abs_sum[channel_index], dtype=np.float64)),
                    "total_squared_l2_energy": float(np.sum(accumulator.squared_sum[channel_index], dtype=np.float64)),
                }
                ranking_scores = {
                    "rms_including_zeros": np.asarray(columns["rms_including_zeros"]),
                    "mean_abs_including_zeros": np.asarray(columns["mean_abs_including_zeros"]),
                    "support_rate": np.asarray(columns["support_rate"]),
                }
                for metric, scores in ranking_scores.items():
                    for rank, feature_id in enumerate(_rank_indices(scores, ranking_top_n), start=1):
                        ranking_records.append(
                            {
                                "representation": representation,
                                "scope": channel,
                                "ranking_metric": metric,
                                "rank": rank,
                                "feature_id": int(feature_id),
                                "score": float(scores[feature_id]),
                            }
                        )

            aggregate_scores = {
                "rms_including_zeros": np.sqrt(
                    np.sum(accumulator.squared_sum, axis=0) / (sample_count * len(CHANNELS))
                ),
                "mean_abs_including_zeros": np.sum(accumulator.abs_sum, axis=0) / (sample_count * len(CHANNELS)),
                "support_rate": np.sum(accumulator.support_count, axis=0) / (sample_count * len(CHANNELS)),
            }
            for metric, scores in aggregate_scores.items():
                for rank, feature_id in enumerate(_rank_indices(scores, ranking_top_n), start=1):
                    ranking_records.append(
                        {
                            "representation": representation,
                            "scope": "all_channels",
                            "ranking_metric": metric,
                            "rank": rank,
                            "feature_id": int(feature_id),
                            "score": float(scores[feature_id]),
                        }
                    )
            feature_summary[representation] = representation_summary
    finally:
        feature_writer.close()

    ranking_table = pa.Table.from_pylist(ranking_records, schema=_ranking_schema())
    pq.write_table(ranking_table, ranking_path, compression="snappy")
    return feature_summary, ranking_records


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite_count": 0, "nonfinite_count": int(values.size)}
    quantile_levels = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    quantiles = np.quantile(finite, quantile_levels)
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(quantiles[7]),
    }


def _summarize_sample_metrics(sample_path: Path, schema: pa.Schema) -> tuple[dict[str, Any], dict[str, int]]:
    numeric_names = [
        field.name
        for field in schema
        if (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
        and field.name not in {"source_row_index", "sample_id"}
    ]
    boolean_names = [field.name for field in schema if pa.types.is_boolean(field.type)]
    table = pq.read_table(sample_path, columns=[*numeric_names, *boolean_names])
    summaries: dict[str, Any] = {}
    for name in numeric_names:
        values = table[name].combine_chunks().to_numpy(zero_copy_only=False)
        summaries[name] = _numeric_summary(values)
    boolean_true_counts = {
        name: int(np.count_nonzero(table[name].combine_chunks().to_numpy(zero_copy_only=False)))
        for name in boolean_names
    }
    return summaries, boolean_true_counts


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fmt(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.6g}"


def _make_report(
    summary: dict[str, Any],
    ranking_records: list[dict[str, Any]],
    *,
    visualizations: bool,
) -> str:
    sample_count = int(summary["sample_count"])
    zero = summary["zero_signal"]
    metrics = summary["sample_metric_distributions"]
    lines = [
        "# Gradient-SAE representation analysis",
        "",
        "## Dataset",
        "",
        f"- Samples analyzed: {sample_count:,}",
        f"- SAE features: {int(summary['sae_feature_size']):,}",
        f"- Logical dimensions per representation: {int(summary['representation_size']):,}",
        f"- Dense zero-signal flags: {zero['dense_count']:,} ({zero['dense_rate']:.2%})",
        f"- Token-TopK zero-signal flags: {zero['token_topk_count']:,} ({zero['token_topk_rate']:.2%})",
        f"- Zero-signal flag disagreements: {zero['disagreement_count']:,}",
        "",
        "## Sample-level overview (median)",
        "",
        "| Metric | Dense | Token-TopK |",
        "|---|---:|---:|",
        f"| L2 norm | {_fmt(metrics['dense_all_l2_norm'].get('p50'))} | "
        f"{_fmt(metrics['token_topk_all_l2_norm'].get('p50'))} |",
        f"| Nonzero density | {_fmt(metrics['dense_all_density'].get('p50'))} | "
        f"{_fmt(metrics['token_topk_all_density'].get('p50'))} |",
        f"| Mean absolute value | {_fmt(metrics['dense_all_mean_abs'].get('p50'))} | "
        f"{_fmt(metrics['token_topk_all_mean_abs'].get('p50'))} |",
        "",
        "## Dense vs Token-TopK (median over finite samples)",
        "",
        f"- Cosine similarity: {_fmt(metrics['dense_token_topk_all_cosine_similarity'].get('p50'))}",
        f"- Relative L2 error: {_fmt(metrics['dense_token_topk_all_relative_l2_error'].get('p50'))}",
        f"- TopK/Dense L2 retention: {_fmt(metrics['dense_token_topk_all_l2_retention'].get('p50'))}",
        "- Dense L2 energy on TopK support: "
        f"{_fmt(metrics['dense_token_topk_all_dense_l2_energy_on_topk_support_rate'].get('p50'))}",
        f"- Sign agreement on comparable TopK entries: "
        f"{_fmt(metrics['dense_token_topk_all_sign_agreement_rate'].get('p50'))}",
        "",
        "## Highest-energy SAE features across all four channels",
        "",
        "| Representation | Rank | Feature ID | RMS |",
        "|---|---:|---:|---:|",
    ]
    selected = [
        row
        for row in ranking_records
        if row["scope"] == "all_channels" and row["ranking_metric"] == "rms_including_zeros" and row["rank"] <= 10
    ]
    for row in selected:
        lines.append(f"| {row['representation']} | {row['rank']} | {row['feature_id']} | {_fmt(row['score'])} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `sample_metrics.parquet`: one row per source sample.",
            "- `feature_metrics.parquet`: one row per representation/channel/SAE feature.",
            "- `feature_rankings.parquet`: top global features under several ranking metrics.",
            "- `global_summary.json`: exact distribution summaries and feature-level totals.",
            "- `analysis_manifest.json`: source provenance and analyzer arguments.",
        ]
    )
    if visualizations:
        lines.extend(
            [
                "- `visualization_report.html`: self-contained visual report.",
                "- `global_pca.png`: shared CountSketch and joint-PCA sample geometry.",
                "- `dense_topk_fidelity.png`: sample-level fidelity diagnostics.",
                "- `feature_overview.png`: global feature energy and usage.",
                "- `sample_heatmaps.pdf`: representative original-feature heatmaps.",
                "- `geometry.parquet`: plotted sample IDs and PCA coordinates.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _temporary_path(final_path: Path) -> Path:
    return final_path.with_name(f"{final_path.name}.{os.getpid()}.inprogress")


def analyze_run(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    input_glob: str = "step-*.parquet",
    batch_size: int = 4,
    sample_top_n: int = 20,
    global_top_n: int = 100,
    max_samples: int | None = None,
    overwrite: bool = False,
    progress_every: int = 100,
    visualize: bool = True,
    visualization_max_samples: int = 2000,
    visualization_sketch_dim: int = 256,
    visualization_sample_count: int = 16,
    visualization_feature_count: int = 30,
    visualization_heatmap_features: int = 32,
    visualization_seed: int = 17,
) -> dict[str, Any]:
    """Analyze one extraction run and return the global summary payload."""
    started_at = time.time()
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    output_dir = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else run_dir / "representation_analysis"
    )
    if batch_size < 1 or sample_top_n < 0 or global_top_n < 1 or progress_every < 0:
        raise ValueError("batch_size/global_top_n must be positive; sample_top_n/progress_every must be non-negative")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")
    if visualize and (
        visualization_max_samples < 1
        or visualization_sketch_dim < 4
        or visualization_sketch_dim % 4
        or visualization_sample_count < 1
        or visualization_feature_count < 1
        or visualization_heatmap_features < 1
    ):
        raise ValueError(
            "Visualization sample/feature counts must be positive and sketch_dim must be divisible by four"
        )

    manifest_path = run_dir / "representation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    representation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_size, zero_eps = _validate_manifest(representation_manifest, manifest_path)
    input_paths, _, input_metadata = _discover_inputs(run_dir, input_glob)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_filenames = (*BASE_OUTPUT_FILENAMES, *(VISUAL_OUTPUT_FILENAMES if visualize else ()))
    final_paths = {name: output_dir / name for name in output_filenames}
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing analysis artifacts: {[str(path) for path in existing]}")
    temporary_paths = {name: _temporary_path(path) for name, path in final_paths.items()}

    sample_schema = _sample_schema()
    accumulators = {
        representation: FeatureAccumulator(feature_size, representation) for representation in REPRESENTATIONS
    }
    geometry_reservoir = (
        GeometryReservoir(
            feature_size=feature_size,
            sketch_dim=visualization_sketch_dim,
            max_samples=visualization_max_samples,
            heatmap_features=visualization_heatmap_features,
            seed=visualization_seed,
        )
        if visualize
        else None
    )
    seen_sample_ids: set[int] = set()
    sample_count = 0
    zero_signal_dense_count = 0
    zero_signal_topk_count = 0
    zero_signal_disagreement_count = 0
    sample_writer: pq.ParquetWriter | None = None
    try:
        sample_writer = pq.ParquetWriter(
            temporary_paths["sample_metrics.parquet"], sample_schema, compression="snappy", use_dictionary=True
        )
        stop = False
        for input_path in input_paths:
            parquet = pq.ParquetFile(input_path)
            source_row_index = 0
            for batch in parquet.iter_batches(batch_size=batch_size, columns=list(INPUT_COLUMNS)):
                if max_samples is not None:
                    remaining = max_samples - sample_count
                    if remaining <= 0:
                        stop = True
                        break
                    if batch.num_rows > remaining:
                        batch = batch.slice(0, remaining)
                        stop = True

                dense_batch = _dense_batch(batch, feature_size)
                if not np.all(np.isfinite(dense_batch)):
                    raise ValueError(f"Dense representations contain NaN or infinity in {input_path}")
                accumulators["dense"].update_dense_batch(dense_batch)

                sample_ids = _primitive_numpy(batch.column(batch.schema.get_field_index("sample_id")), dtype=np.int64)
                prompt_lengths = _primitive_numpy(
                    batch.column(batch.schema.get_field_index("prompt_len_tok")), dtype=np.int64
                )
                zero_dense = _primitive_numpy(
                    batch.column(batch.schema.get_field_index("zero_signal_dense")), dtype=np.bool_
                )
                zero_topk = _primitive_numpy(
                    batch.column(batch.schema.get_field_index("zero_signal_token_topk")), dtype=np.bool_
                )
                ragged = {
                    name: _ragged_column(batch, name, np.int64 if name.endswith("_ids") else np.float32)
                    for name in INPUT_COLUMNS
                    if name.startswith("topk_")
                }
                output_rows: list[dict[str, Any]] = []
                for row_index in range(batch.num_rows):
                    sample_id = int(sample_ids[row_index])
                    if sample_id in seen_sample_ids:
                        raise ValueError(f"Duplicate sample_id across extraction shards: {sample_id}")
                    seen_sample_ids.add(sample_id)
                    prompt_ids = _slice_ragged(*ragged["topk_prompt_feat_ids"], row_index)
                    prompt_mean = _slice_ragged(*ragged["topk_prompt_feat_mean"], row_index)
                    prompt_peak = _slice_ragged(*ragged["topk_prompt_feat_signed_peak"], row_index)
                    response_ids = _slice_ragged(*ragged["topk_response_feat_ids"], row_index)
                    response_mean = _slice_ragged(*ragged["topk_response_feat_mean"], row_index)
                    response_peak = _slice_ragged(*ragged["topk_response_feat_signed_peak"], row_index)
                    _validate_sparse_span(sample_id, "prompt", prompt_ids, prompt_mean, prompt_peak, feature_size)
                    _validate_sparse_span(
                        sample_id, "response", response_ids, response_mean, response_peak, feature_size
                    )
                    output_row, sparse_values, _ = _build_sample_row(
                        source_shard=input_path.name,
                        source_row_index=source_row_index + row_index,
                        sample_id=sample_id,
                        prompt_len_tok=int(prompt_lengths[row_index]),
                        zero_signal_dense=bool(zero_dense[row_index]),
                        zero_signal_token_topk=bool(zero_topk[row_index]),
                        dense=dense_batch[row_index],
                        prompt_ids=prompt_ids,
                        prompt_mean=prompt_mean,
                        prompt_peak=prompt_peak,
                        response_ids=response_ids,
                        response_mean=response_mean,
                        response_peak=response_peak,
                        feature_size=feature_size,
                        zero_eps=zero_eps,
                        top_n=sample_top_n,
                    )
                    output_rows.append(output_row)
                    sparse_ids = (prompt_ids, prompt_ids, response_ids, response_ids)
                    if geometry_reservoir is not None:
                        geometry_reservoir.consider(
                            sample_row=output_row,
                            dense=dense_batch[row_index],
                            sparse_ids=sparse_ids,
                            sparse_values=sparse_values,
                        )
                    for channel, (ids, values) in enumerate(zip(sparse_ids, sparse_values, strict=True)):
                        accumulators["token_topk"].update_sparse(channel, ids, values)
                    zero_signal_dense_count += int(zero_dense[row_index])
                    zero_signal_topk_count += int(zero_topk[row_index])
                    zero_signal_disagreement_count += int(bool(zero_dense[row_index]) != bool(zero_topk[row_index]))

                sample_writer.write_table(pa.Table.from_pylist(output_rows, schema=sample_schema))
                sample_count += batch.num_rows
                source_row_index += batch.num_rows
                if progress_every and (sample_count % progress_every < batch.num_rows):
                    print(f"Analyzed {sample_count:,} samples", flush=True)
                if stop:
                    break
            if stop:
                break
        sample_writer.close()
        sample_writer = None
        if sample_count == 0:
            raise ValueError("No samples were analyzed")

        feature_summary, ranking_records = _write_feature_outputs(
            temporary_paths["feature_metrics.parquet"],
            temporary_paths["feature_rankings.parquet"],
            accumulators,
            sample_count,
            global_top_n,
        )
        sample_distributions, boolean_true_counts = _summarize_sample_metrics(
            temporary_paths["sample_metrics.parquet"], sample_schema
        )
        global_summary = {
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "sample_count": sample_count,
            "sae_feature_size": feature_size,
            "representation_size": 4 * feature_size,
            "channels": list(CHANNELS),
            "zero_signal": {
                "dense_count": zero_signal_dense_count,
                "dense_rate": zero_signal_dense_count / sample_count,
                "token_topk_count": zero_signal_topk_count,
                "token_topk_rate": zero_signal_topk_count / sample_count,
                "disagreement_count": zero_signal_disagreement_count,
                "disagreement_rate": zero_signal_disagreement_count / sample_count,
            },
            "sample_boolean_true_counts": boolean_true_counts,
            "sample_metric_distributions": sample_distributions,
            "feature_level_summary": feature_summary,
        }
        if geometry_reservoir is not None:
            global_summary["visualization"] = write_visualizations(
                output_paths=temporary_paths,
                sample_metrics_path=temporary_paths["sample_metrics.parquet"],
                accumulators=accumulators,
                sample_count=sample_count,
                channels=CHANNELS,
                reservoir=geometry_reservoir,
                sample_plot_count=visualization_sample_count,
                feature_plot_count=visualization_feature_count,
            )
        _json_dump(temporary_paths["global_summary.json"], global_summary)
        temporary_paths["summary.md"].write_text(
            _make_report(global_summary, ranking_records, visualizations=visualize), encoding="utf-8"
        )
        analysis_manifest = {
            "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
            "created_unix_time": time.time(),
            "elapsed_seconds": time.time() - started_at,
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "input_glob": input_glob,
            "input_files": input_metadata,
            "source_representation_manifest": representation_manifest,
            "settings": {
                "batch_size": batch_size,
                "sample_top_n": sample_top_n,
                "global_top_n": global_top_n,
                "max_samples": max_samples,
                "zero_eps": zero_eps,
                "visualize": visualize,
                "visualization_max_samples": visualization_max_samples,
                "visualization_sketch_dim": visualization_sketch_dim,
                "visualization_sample_count": visualization_sample_count,
                "visualization_feature_count": visualization_feature_count,
                "visualization_heatmap_features": visualization_heatmap_features,
                "visualization_seed": visualization_seed,
            },
            "samples_analyzed": sample_count,
            "outputs": list(output_filenames),
            "python": sys.version,
            "numpy_version": np.__version__,
            "pyarrow_version": pa.__version__,
        }
        _json_dump(temporary_paths["analysis_manifest.json"], analysis_manifest)

        for name in output_filenames:
            os.replace(temporary_paths[name], final_paths[name])
        print(f"Analysis complete: {sample_count:,} samples -> {output_dir}", flush=True)
        return global_summary
    except BaseException:
        if sample_writer is not None:
            sample_writer.close()
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Extraction run containing representation_manifest.json.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: RUN_DIR/representation_analysis.",
    )
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--batch-size", type=int, default=4, help="Dense rows decoded per batch.")
    parser.add_argument("--sample-top-n", type=int, default=20)
    parser.add_argument("--global-top-n", type=int, default=100)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug-only prefix limit; default analyzes all rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-visualizations", action="store_true", help="Skip PNG/PDF/HTML visual outputs.")
    parser.add_argument(
        "--visualization-max-samples",
        type=int,
        default=2000,
        help="Uniform reservoir size for CountSketch/PCA geometry.",
    )
    parser.add_argument(
        "--visualization-sketch-dim",
        type=int,
        default=256,
        help="Shared CountSketch dimension; must be divisible by four.",
    )
    parser.add_argument(
        "--visualization-sample-count",
        type=int,
        default=16,
        help="Representative sample heatmaps included in PDF/HTML.",
    )
    parser.add_argument(
        "--visualization-feature-count",
        type=int,
        default=30,
        help="Global top features shown in the channel heatmap.",
    )
    parser.add_argument(
        "--visualization-heatmap-features",
        type=int,
        default=32,
        help="Original SAE feature IDs shown for each representative sample.",
    )
    parser.add_argument("--visualization-seed", type=int, default=17)
    args = parser.parse_args()
    analyze_run(
        args.run_dir,
        args.output_dir,
        input_glob=args.input_glob,
        batch_size=args.batch_size,
        sample_top_n=args.sample_top_n,
        global_top_n=args.global_top_n,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        visualize=not args.no_visualizations,
        visualization_max_samples=args.visualization_max_samples,
        visualization_sketch_dim=args.visualization_sketch_dim,
        visualization_sample_count=args.visualization_sample_count,
        visualization_feature_count=args.visualization_feature_count,
        visualization_heatmap_features=args.visualization_heatmap_features,
        visualization_seed=args.visualization_seed,
    )


if __name__ == "__main__":
    main()
