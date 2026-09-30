#!/usr/bin/env python3
"""Analyze how often every gradient-SAE feature is materially active.

This is the original, richer activation-analysis entry point.  It keeps exact
nonzero statistics, defines a per-sample/per-channel absolute Top-K
``dominant`` set, and optionally associates frequent features with source
prompt metadata.  For plain min/max/median summaries use
``summarize_feature_value_distributions.py`` instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from .analyze_representations import (
        CHANNELS,
        INPUT_COLUMNS,
        REPRESENTATIONS,
        FeatureAccumulator,
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
        CHANNELS,
        INPUT_COLUMNS,
        REPRESENTATIONS,
        FeatureAccumulator,
        _dense_batch,
        _discover_inputs,
        _primitive_numpy,
        _ragged_column,
        _slice_ragged,
        _validate_manifest,
        _validate_sparse_span,
    )


ACTIVATION_SCHEMA_VERSION = 1
ALL_CHANNELS = "all_channels"
SCOPES = (*CHANNELS, ALL_CHANNELS)
OUTPUT_FILENAMES = (
    "feature_activation_metrics.parquet",
    "dominant_activation_events.parquet",
    "high_frequency_features.parquet",
    "feature_examples.parquet",
    "activation_summary.json",
    "activation_summary.md",
    "activation_manifest.json",
)


@dataclass
class SampleMetadata:
    sample_id: int
    prompt: str = ""
    difficulty: float | None = None
    ground_truth: str = ""
    data_source: str = ""
    ability: str = ""


class DominanceAccumulator:
    """Streaming Top-K moments plus cross-channel sample unions."""

    def __init__(self, feature_size: int, representation: str) -> None:
        channel_shape = (len(CHANNELS), int(feature_size))
        feature_shape = (int(feature_size),)
        self.feature_size = int(feature_size)
        self.representation = representation
        self.dominant_coordinate_count = np.zeros(channel_shape, dtype=np.int64)
        self.dominant_signed_sum = np.zeros(channel_shape, dtype=np.float64)
        self.dominant_abs_sum = np.zeros(channel_shape, dtype=np.float64)
        self.dominant_squared_sum = np.zeros(channel_shape, dtype=np.float64)
        self.dominant_positive_count = np.zeros(channel_shape, dtype=np.int64)
        self.dominant_negative_count = np.zeros(channel_shape, dtype=np.int64)
        self.support_sample_union_count = np.zeros(feature_shape, dtype=np.int64)
        self.nonzero_sample_union_count = np.zeros(feature_shape, dtype=np.int64)
        self.dominant_sample_union_count = np.zeros(feature_shape, dtype=np.int64)
        self.signal_sample_count = 0

    def update_sample(
        self,
        *,
        support_ids: tuple[np.ndarray | None, ...],
        feature_ids: tuple[np.ndarray, ...],
        values: tuple[np.ndarray, ...],
        dominant: tuple[tuple[np.ndarray, np.ndarray], ...],
    ) -> None:
        if any(np.any(channel_values != 0) for channel_values in values):
            self.signal_sample_count += 1
        if all(ids is None for ids in support_ids):
            self.support_sample_union_count += 1
            dense_nonzero = np.zeros(self.feature_size, dtype=np.bool_)
            for channel_values in values:
                dense_nonzero |= channel_values != 0
            self.nonzero_sample_union_count += dense_nonzero
        else:
            supported = _unique_concat([ids for ids in support_ids if ids is not None])
            self.support_sample_union_count[supported] += 1
            nonzero_ids = _unique_concat(
                [ids[channel_values != 0] for ids, channel_values in zip(feature_ids, values, strict=True)]
            )
            self.nonzero_sample_union_count[nonzero_ids] += 1

        union: list[np.ndarray] = []
        for channel, (ids, selected_values) in enumerate(dominant):
            if not ids.size:
                continue
            union.append(ids)
            self.dominant_coordinate_count[channel, ids] += 1
            self.dominant_signed_sum[channel, ids] += selected_values
            absolute = np.abs(selected_values)
            self.dominant_abs_sum[channel, ids] += absolute
            self.dominant_squared_sum[channel, ids] += selected_values.astype(np.float64) ** 2
            self.dominant_positive_count[channel, ids[selected_values > 0]] += 1
            self.dominant_negative_count[channel, ids[selected_values < 0]] += 1
        self.dominant_sample_union_count[_unique_concat(union)] += 1


def _unique_concat(arrays: Iterable[np.ndarray]) -> np.ndarray:
    arrays = tuple(array for array in arrays if array.size)
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)


def _strongest(
    feature_ids: np.ndarray,
    values: np.ndarray,
    top_k: int,
    min_abs: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.flatnonzero(np.abs(values) > min_abs)
    if not positions.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    keep = min(top_k, int(positions.size))
    if positions.size > keep:
        magnitude = np.abs(values[positions])
        positions = positions[np.argpartition(magnitude, -keep)[-keep:]]
    ids = feature_ids[positions]
    selected = values[positions]
    order = np.lexsort((ids, -np.abs(selected)))
    return ids[order].astype(np.int64, copy=False), selected[order].astype(np.float32, copy=False)


def _event_schema() -> pa.Schema:
    return pa.schema(
        [
            ("representation", pa.string()),
            ("channel", pa.string()),
            ("sample_id", pa.int64()),
            ("feature_id", pa.int64()),
            ("rank_in_sample_channel", pa.int32()),
            ("value", pa.float32()),
            ("abs_value", pa.float32()),
            ("fraction_of_channel_max", pa.float32()),
            ("fraction_of_channel_l2_energy", pa.float32()),
        ]
    )


def _metric_schema() -> pa.Schema:
    return pa.schema(
        [
            ("representation", pa.string()),
            ("scope", pa.string()),
            ("feature_id", pa.int64()),
            ("sample_count", pa.int64()),
            ("signal_sample_count", pa.int64()),
            ("coordinate_count", pa.int64()),
            ("support_sample_count", pa.int64()),
            ("nonzero_sample_count", pa.int64()),
            ("nonzero_coordinate_count", pa.int64()),
            ("dominant_sample_count", pa.int64()),
            ("dominant_coordinate_count", pa.int64()),
            ("support_rate_all", pa.float64()),
            ("nonzero_rate_all", pa.float64()),
            ("nonzero_rate_signal", pa.float64()),
            ("dominant_rate_all", pa.float64()),
            ("dominant_rate_signal", pa.float64()),
            ("mean_including_zeros", pa.float64()),
            ("mean_abs_including_zeros", pa.float64()),
            ("rms_including_zeros", pa.float64()),
            ("mean_abs_when_nonzero", pa.float64()),
            ("rms_when_nonzero", pa.float64()),
            ("max_abs", pa.float64()),
            ("mean_when_dominant", pa.float64()),
            ("mean_abs_when_dominant", pa.float64()),
            ("rms_when_dominant", pa.float64()),
            ("positive_dominant_count", pa.int64()),
            ("negative_dominant_count", pa.int64()),
            ("dominant_sign_balance", pa.float64()),
            ("energy_fraction_within_scope", pa.float64()),
        ]
    )


def _high_frequency_schema() -> pa.Schema:
    fields = [
        ("representation", pa.string()),
        ("scope", pa.string()),
        ("rank", pa.int32()),
        ("feature_id", pa.int64()),
        ("dominant_sample_count", pa.int64()),
    ]
    fields.extend((name, pa.float64()) for name in (
        "dominant_rate_all", "dominant_rate_signal", "mean_abs_when_dominant",
        "rms_when_dominant", "max_abs", "dominant_sign_balance",
        "energy_fraction_within_scope",
    ))
    fields.extend(
        [
            ("active_prompt_count_with_metadata", pa.int64()),
            ("active_prompt_length_mean", pa.float64()),
            ("background_prompt_length_mean", pa.float64()),
            ("active_difficulty_mean", pa.float64()),
            ("background_difficulty_mean", pa.float64()),
            ("difficulty_delta", pa.float64()),
            ("top_enriched_terms", pa.list_(pa.string())),
        ]
    )
    return pa.schema(fields)


def _example_schema() -> pa.Schema:
    return pa.schema(
        [
            ("representation", pa.string()), ("scope", pa.string()),
            ("feature_rank", pa.int32()), ("feature_id", pa.int64()),
            ("example_rank", pa.int32()), ("sample_id", pa.int64()),
            ("strongest_channel", pa.string()), ("value", pa.float32()),
            ("abs_value", pa.float32()), ("rank_in_sample_channel", pa.int32()),
            ("fraction_of_channel_max", pa.float32()),
            ("fraction_of_channel_l2_energy", pa.float32()),
            ("prompt_len_tok", pa.int64()), ("difficulty", pa.float64()),
            ("data_source", pa.string()), ("ability", pa.string()),
            ("prompt", pa.string()), ("ground_truth", pa.string()),
        ]
    )


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray | int) -> np.ndarray:
    output = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=output, where=np.asarray(denominator) != 0)


def _metric_columns(
    base: FeatureAccumulator,
    dominant: DominanceAccumulator,
    channel_index: int | None,
    sample_count: int,
) -> dict[str, Any]:
    if channel_index is None:
        scope = ALL_CHANNELS
        coordinate_count = sample_count * len(CHANNELS)
        support_samples = dominant.support_sample_union_count
        nonzero_samples = dominant.nonzero_sample_union_count
        nonzero_coordinates = np.sum(base.nonzero_count, axis=0)
        dominant_samples = dominant.dominant_sample_union_count
        dominant_coordinates = np.sum(dominant.dominant_coordinate_count, axis=0)
        signed_sum = np.sum(base.signed_sum, axis=0)
        abs_sum = np.sum(base.abs_sum, axis=0)
        squared_sum = np.sum(base.squared_sum, axis=0)
        selected_signed = np.sum(dominant.dominant_signed_sum, axis=0)
        selected_abs = np.sum(dominant.dominant_abs_sum, axis=0)
        selected_squared = np.sum(dominant.dominant_squared_sum, axis=0)
        selected_positive = np.sum(dominant.dominant_positive_count, axis=0)
        selected_negative = np.sum(dominant.dominant_negative_count, axis=0)
        minimum = np.where(np.isfinite(base.minimum), base.minimum, 0.0)
        maximum = np.where(np.isfinite(base.maximum), base.maximum, 0.0)
        max_abs = np.max(np.maximum(np.abs(minimum), np.abs(maximum)), axis=0)
    else:
        scope = CHANNELS[channel_index]
        coordinate_count = sample_count
        support_samples = base.support_count[channel_index]
        nonzero_samples = base.nonzero_count[channel_index]
        nonzero_coordinates = nonzero_samples
        dominant_samples = dominant.dominant_coordinate_count[channel_index]
        dominant_coordinates = dominant_samples
        signed_sum = base.signed_sum[channel_index]
        abs_sum = base.abs_sum[channel_index]
        squared_sum = base.squared_sum[channel_index]
        selected_signed = dominant.dominant_signed_sum[channel_index]
        selected_abs = dominant.dominant_abs_sum[channel_index]
        selected_squared = dominant.dominant_squared_sum[channel_index]
        selected_positive = dominant.dominant_positive_count[channel_index]
        selected_negative = dominant.dominant_negative_count[channel_index]
        minimum = np.where(np.isfinite(base.minimum[channel_index]), base.minimum[channel_index], 0.0)
        maximum = np.where(np.isfinite(base.maximum[channel_index]), base.maximum[channel_index], 0.0)
        max_abs = np.maximum(np.abs(minimum), np.abs(maximum))
    signal_count = dominant.signal_sample_count
    energy_total = float(np.sum(squared_sum, dtype=np.float64))
    return {
        "representation": [base.representation] * base.feature_size,
        "scope": [scope] * base.feature_size,
        "feature_id": np.arange(base.feature_size, dtype=np.int64),
        "sample_count": np.full(base.feature_size, sample_count, dtype=np.int64),
        "signal_sample_count": np.full(base.feature_size, signal_count, dtype=np.int64),
        "coordinate_count": np.full(base.feature_size, coordinate_count, dtype=np.int64),
        "support_sample_count": support_samples,
        "nonzero_sample_count": nonzero_samples,
        "nonzero_coordinate_count": nonzero_coordinates,
        "dominant_sample_count": dominant_samples,
        "dominant_coordinate_count": dominant_coordinates,
        "support_rate_all": support_samples / sample_count,
        "nonzero_rate_all": nonzero_samples / sample_count,
        "nonzero_rate_signal": nonzero_samples / signal_count,
        "dominant_rate_all": dominant_samples / sample_count,
        "dominant_rate_signal": dominant_samples / signal_count,
        "mean_including_zeros": signed_sum / coordinate_count,
        "mean_abs_including_zeros": abs_sum / coordinate_count,
        "rms_including_zeros": np.sqrt(squared_sum / coordinate_count),
        "mean_abs_when_nonzero": _safe_divide(abs_sum, nonzero_coordinates),
        "rms_when_nonzero": np.sqrt(_safe_divide(squared_sum, nonzero_coordinates)),
        "max_abs": max_abs.astype(np.float64),
        "mean_when_dominant": _safe_divide(selected_signed, dominant_coordinates),
        "mean_abs_when_dominant": _safe_divide(selected_abs, dominant_coordinates),
        "rms_when_dominant": np.sqrt(_safe_divide(selected_squared, dominant_coordinates)),
        "positive_dominant_count": selected_positive,
        "negative_dominant_count": selected_negative,
        "dominant_sign_balance": _safe_divide(selected_signed, selected_abs),
        "energy_fraction_within_scope": squared_sum / energy_total,
    }


def _write_metrics(
    path: Path,
    bases: dict[str, FeatureAccumulator],
    dominances: dict[str, DominanceAccumulator],
    sample_count: int,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    writer = pq.ParquetWriter(path, _metric_schema(), compression="snappy", use_dictionary=True)
    materialized = {}
    try:
        for representation in REPRESENTATIONS:
            for channel_index in (*range(len(CHANNELS)), None):
                columns = _metric_columns(
                    bases[representation], dominances[representation], channel_index, sample_count
                )
                scope = ALL_CHANNELS if channel_index is None else CHANNELS[channel_index]
                materialized[(representation, scope)] = {
                    key: np.asarray(value)
                    for key, value in columns.items()
                    if key not in {"representation", "scope"}
                }
                writer.write_table(pa.Table.from_pydict(columns, schema=_metric_schema()))
    finally:
        writer.close()
    return materialized


def _rank_frequent(metrics: dict[tuple[str, str], dict[str, np.ndarray]], top_n: int) -> list[dict[str, Any]]:
    output = []
    copied = (
        "feature_id", "dominant_sample_count", "dominant_rate_all", "dominant_rate_signal",
        "mean_abs_when_dominant", "rms_when_dominant", "max_abs",
        "dominant_sign_balance", "energy_fraction_within_scope",
    )
    for representation in REPRESENTATIONS:
        for scope in SCOPES:
            columns = metrics[(representation, scope)]
            count = columns["dominant_sample_count"]
            rms = np.nan_to_num(columns["rms_when_dominant"], nan=-np.inf)
            ids = columns["feature_id"]
            order = np.lexsort((ids, -rms, -count))[:top_n]
            for rank, index in enumerate(order, 1):
                row = {"representation": representation, "scope": scope, "rank": rank}
                for name in copied:
                    value = columns[name][index]
                    row[name] = int(value) if name in {"feature_id", "dominant_sample_count"} else float(value)
                output.append(row)
    return output


def _infer_dataset_path(run_dir: Path) -> Path | None:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return None
    in_data = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "data:":
            in_data = True
        elif in_data and line and not line.startswith(" "):
            break
        elif in_data and line.startswith("  train_files:"):
            value = line.split(":", 1)[1].strip().strip("'\"")
            return Path(value).expanduser() if value and value not in {"null", "~"} else None
    return None


def _prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            str(item.get("content", "")) if isinstance(item, dict) else str(item)
            for item in value
        )
    return "" if value is None else str(value)


def _load_dataset_metadata(path: Path | None, wanted: set[int]) -> dict[int, SampleMetadata]:
    if path is None or not path.is_file():
        return {}
    available = set(pq.ParquetFile(path).schema_arrow.names)
    if not {"prompt", "extra_info"}.issubset(available):
        return {}
    columns = [name for name in ("prompt", "extra_info", "reward_model", "data_source", "ability") if name in available]
    output = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=columns):
        for row in batch.to_pylist():
            extra = row.get("extra_info") or {}
            sample_id = extra.get("index") if isinstance(extra, dict) else None
            if sample_id is None or int(sample_id) not in wanted:
                continue
            sample_id = int(sample_id)
            reward = row.get("reward_model") or {}
            difficulty = extra.get("difficulty")
            output[sample_id] = SampleMetadata(
                sample_id,
                _prompt_text(row.get("prompt")),
                float(difficulty) if difficulty is not None else None,
                str(reward.get("ground_truth", "")) if isinstance(reward, dict) else "",
                str(row.get("data_source") or ""),
                str(row.get("ability") or ""),
            )
    return output


_WORD_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z][A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "and", "answer", "are", "boxed", "can", "determine", "each", "final", "find", "for", "from",
    "given", "has", "have", "into", "let", "output", "problem", "step", "such", "that", "the", "their",
    "then", "there", "these", "think", "this", "using", "what", "when", "where", "which", "with", "within",
}


def _document_terms(text: str) -> set[str]:
    return {
        token.lower().lstrip("\\")
        for token in _WORD_RE.findall(text)
        if token.lower().lstrip("\\") not in _STOPWORDS
    }


def _mean_finite(values: Iterable[float | int | None]) -> float:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else math.nan


def _enriched_terms(
    active_ids: set[int],
    background_ids: set[int],
    terms: dict[int, set[str]],
    top_n: int,
) -> list[str]:
    active_ids &= background_ids & terms.keys()
    background_ids &= terms.keys()
    active_df: Counter[str] = Counter()
    background_df: Counter[str] = Counter()
    for sample_id in background_ids:
        background_df.update(terms[sample_id])
        if sample_id in active_ids:
            active_df.update(terms[sample_id])
    minimum = max(2, math.ceil(0.03 * len(active_ids)))
    scored = []
    for term, count in active_df.items():
        if count < minimum:
            continue
        active_rate = (count + 0.5) / (len(active_ids) + 1)
        background_rate = (background_df[term] + 0.5) / (len(background_ids) + 1)
        lift = active_rate / background_rate
        score = max(math.log(lift), 0.0) * math.sqrt(count)
        if score:
            scored.append((score, count, term, lift))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [f"{term} (lift={lift:.2f}, {count}/{len(active_ids)})" for _, count, term, lift in scored[:top_n]]


def _collect_selected_events(
    path: Path,
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, int], set[int]], dict[tuple[str, str, int], dict[int, dict[str, Any]]]]:
    selected = {(row["representation"], row["scope"], row["feature_id"]) for row in rows}
    samples = defaultdict(set)
    best = defaultdict(dict)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        for event in batch.to_pylist():
            for scope in (event["channel"], ALL_CHANNELS):
                key = (event["representation"], scope, event["feature_id"])
                if key not in selected:
                    continue
                sample_id = event["sample_id"]
                samples[key].add(sample_id)
                previous = best[key].get(sample_id)
                if previous is None or event["abs_value"] > previous["abs_value"]:
                    best[key][sample_id] = event
    return samples, best


def _decorate_rows(
    rows: list[dict[str, Any]],
    active_samples: dict[tuple[str, str, int], set[int]],
    metadata: dict[int, SampleMetadata],
    prompt_lengths: dict[int, int],
    signal_samples: dict[str, set[int]],
    term_count: int,
) -> None:
    terms = {sample_id: _document_terms(record.prompt) for sample_id, record in metadata.items()}
    for row in rows:
        key = (row["representation"], row["scope"], row["feature_id"])
        active = active_samples.get(key, set())
        background = signal_samples[row["representation"]]
        active_metadata = active & metadata.keys()
        background_metadata = background & metadata.keys()
        row["active_prompt_count_with_metadata"] = len(active_metadata)
        row["active_prompt_length_mean"] = _mean_finite(prompt_lengths.get(i) for i in active)
        row["background_prompt_length_mean"] = _mean_finite(prompt_lengths.get(i) for i in background)
        row["active_difficulty_mean"] = _mean_finite(metadata[i].difficulty for i in active_metadata)
        row["background_difficulty_mean"] = _mean_finite(metadata[i].difficulty for i in background_metadata)
        row["difficulty_delta"] = row["active_difficulty_mean"] - row["background_difficulty_mean"]
        row["top_enriched_terms"] = _enriched_terms(active, background, terms, term_count)


def _build_examples(
    rows: list[dict[str, Any]],
    best: dict[tuple[str, str, int], dict[int, dict[str, Any]]],
    metadata: dict[int, SampleMetadata],
    prompt_lengths: dict[int, int],
    count: int,
) -> list[dict[str, Any]]:
    output = []
    for feature in rows:
        key = (feature["representation"], feature["scope"], feature["feature_id"])
        events = sorted(best.get(key, {}).values(), key=lambda event: -event["abs_value"])[:count]
        for rank, event in enumerate(events, 1):
            sample_id = event["sample_id"]
            source = metadata.get(sample_id, SampleMetadata(sample_id))
            output.append(
                {
                    "representation": key[0], "scope": key[1], "feature_rank": feature["rank"],
                    "feature_id": key[2], "example_rank": rank, "sample_id": sample_id,
                    "strongest_channel": event["channel"], "value": event["value"],
                    "abs_value": event["abs_value"],
                    "rank_in_sample_channel": event["rank_in_sample_channel"],
                    "fraction_of_channel_max": event["fraction_of_channel_max"],
                    "fraction_of_channel_l2_energy": event["fraction_of_channel_l2_energy"],
                    "prompt_len_tok": prompt_lengths.get(sample_id, -1), "difficulty": source.difficulty,
                    "data_source": source.data_source, "ability": source.ability,
                    "prompt": source.prompt, "ground_truth": source.ground_truth,
                }
            )
    return output


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _report(sample_count: int, feature_size: int, top_k: int, rows: list[dict[str, Any]]) -> str:
    selected = [row for row in rows if row["scope"] == ALL_CHANNELS and row["rank"] <= 10]
    lines = [
        "# Gradient-SAE feature activation analysis", "", "## Definition", "",
        f"- Samples: {sample_count:,}; SAE features: {feature_size:,}.",
        f"- Dominant means absolute-value Top-{top_k} within a sample and channel.",
        "- These are signed gradient projections, not forward SAE latent activations.", "",
        "| Representation | Rank | Feature | Dominant samples | Signal rate |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['representation']} | {row['rank']} | {row['feature_id']} | "
        f"{row['dominant_sample_count']} | {row['dominant_rate_signal']:.2%} |"
        for row in selected
    )
    return "\n".join(lines) + "\n"


def analyze_feature_activations(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    input_glob: str = "step-*.parquet",
    batch_size: int = 4,
    active_top_k: int = 128,
    active_min_abs: float = 0.0,
    high_frequency_top_n: int = 20,
    examples_per_feature: int = 8,
    enriched_term_count: int = 8,
    dataset_path: str | Path | None = None,
    max_samples: int | None = None,
    overwrite: bool = False,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Run the original frequency/strength/metadata activation analysis."""
    started = time.time()
    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else run_dir / "feature_activation_analysis"
    if active_top_k < 1 or high_frequency_top_n < 1 or examples_per_feature < 1:
        raise ValueError("Top-K, ranking count, and example count must be positive")
    manifest_path = run_dir / "representation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_size, zero_eps = _validate_manifest(manifest, manifest_path)
    input_paths, _, input_metadata = _discover_inputs(run_dir, input_glob)
    resolved_dataset = (
        _infer_dataset_path(run_dir) if dataset_path is None
        else None if str(dataset_path).lower() in {"none", "null", "-"}
        else Path(dataset_path).expanduser()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    final = {name: output_dir / name for name in OUTPUT_FILENAMES}
    if not overwrite and any(path.exists() for path in final.values()):
        raise FileExistsError(f"Outputs already exist under {output_dir}")
    temporary = {name: path.with_name(f".{path.name}.{os.getpid()}.inprogress") for name, path in final.items()}
    bases = {name: FeatureAccumulator(feature_size, name) for name in REPRESENTATIONS}
    dominant = {name: DominanceAccumulator(feature_size, name) for name in REPRESENTATIONS}
    event_writer = pq.ParquetWriter(
        temporary["dominant_activation_events.parquet"], _event_schema(), compression="snappy"
    )
    sample_count = 0
    seen = set()
    prompt_lengths = {}
    signal_samples = {name: set() for name in REPRESENTATIONS}
    dense_ids = np.arange(feature_size, dtype=np.int64)
    try:
        stop = False
        for input_path in input_paths:
            parquet = pq.ParquetFile(input_path)
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
                bases["dense"].update_dense_batch(dense_batch)
                ids = _primitive_numpy(batch.column(batch.schema.get_field_index("sample_id")), dtype=np.int64)
                lengths = _primitive_numpy(batch.column(batch.schema.get_field_index("prompt_len_tok")), dtype=np.int64)
                ragged = {
                    name: _ragged_column(batch, name, np.int64 if name.endswith("_ids") else np.float32)
                    for name in INPUT_COLUMNS if name.startswith("topk_")
                }
                events = []
                for row_index in range(batch.num_rows):
                    sample_id = int(ids[row_index])
                    if sample_id in seen:
                        raise ValueError(f"Duplicate sample_id: {sample_id}")
                    seen.add(sample_id)
                    prompt_lengths[sample_id] = int(lengths[row_index])
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
                    sparse_ids = (prompt_ids, prompt_ids, response_ids, response_ids)
                    sparse_values = (prompt_mean, prompt_peak, response_mean, response_peak)
                    dense_values = tuple(dense_batch[row_index, channel] for channel in range(len(CHANNELS)))
                    for channel, (channel_ids, values) in enumerate(zip(sparse_ids, sparse_values, strict=True)):
                        bases["token_topk"].update_sparse(channel, channel_ids, values)
                    for representation, support, feature_ids, values in (
                        ("dense", (None,) * 4, (dense_ids,) * 4, dense_values),
                        ("token_topk", sparse_ids, sparse_ids, sparse_values),
                    ):
                        selected = tuple(
                            _strongest(channel_ids, channel_values, active_top_k, active_min_abs)
                            for channel_ids, channel_values in zip(feature_ids, values, strict=True)
                        )
                        dominant[representation].update_sample(
                            support_ids=support, feature_ids=feature_ids, values=values, dominant=selected
                        )
                        if any(np.any(channel_values != 0) for channel_values in values):
                            signal_samples[representation].add(sample_id)
                        for channel, (selected_ids, selected_values) in enumerate(selected):
                            full = values[channel]
                            energy = float(np.dot(full.astype(np.float64), full.astype(np.float64)))
                            maximum = float(np.max(np.abs(full))) if full.size else 0.0
                            for rank, (feature_id, value) in enumerate(
                                zip(selected_ids, selected_values, strict=True), 1
                            ):
                                absolute = abs(float(value))
                                events.append(
                                    {
                                        "representation": representation, "channel": CHANNELS[channel],
                                        "sample_id": sample_id, "feature_id": int(feature_id),
                                        "rank_in_sample_channel": rank, "value": float(value),
                                        "abs_value": absolute,
                                        "fraction_of_channel_max": absolute / maximum if maximum else math.nan,
                                        "fraction_of_channel_l2_energy": value * value / energy if energy else math.nan,
                                    }
                                )
                event_writer.write_table(pa.Table.from_pylist(events, schema=_event_schema()))
                sample_count += batch.num_rows
                if progress_every and sample_count % progress_every < batch.num_rows:
                    print(f"Analyzed {sample_count:,} samples", flush=True)
                if stop:
                    break
            if stop:
                break
        event_writer.close()
        metrics = _write_metrics(temporary["feature_activation_metrics.parquet"], bases, dominant, sample_count)
        frequent = _rank_frequent(metrics, high_frequency_top_n)
        active_samples, best = _collect_selected_events(temporary["dominant_activation_events.parquet"], frequent)
        metadata = _load_dataset_metadata(resolved_dataset, seen)
        _decorate_rows(frequent, active_samples, metadata, prompt_lengths, signal_samples, enriched_term_count)
        pq.write_table(
            pa.Table.from_pylist(frequent, schema=_high_frequency_schema()),
            temporary["high_frequency_features.parquet"],
        )
        examples = _build_examples(frequent, best, metadata, prompt_lengths, examples_per_feature)
        pq.write_table(pa.Table.from_pylist(examples, schema=_example_schema()), temporary["feature_examples.parquet"])
        summary = {
            "activation_schema_version": ACTIVATION_SCHEMA_VERSION,
            "sample_count": sample_count,
            "sae_feature_size": feature_size,
            "active_definition": {"name": "per_sample_per_channel_absolute_topk", "top_k": active_top_k},
            "signal_sample_count": {name: dominant[name].signal_sample_count for name in REPRESENTATIONS},
        }
        temporary["activation_summary.json"].write_text(
            json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        temporary["activation_summary.md"].write_text(
            _report(sample_count, feature_size, active_top_k, frequent), encoding="utf-8"
        )
        analysis_manifest = {
            "activation_schema_version": ACTIVATION_SCHEMA_VERSION,
            "created_unix_time": time.time(), "elapsed_seconds": time.time() - started,
            "run_dir": str(run_dir), "output_dir": str(output_dir), "input_files": input_metadata,
            "source_gradient_zero_eps": zero_eps, "settings": {
                "batch_size": batch_size, "active_top_k": active_top_k,
                "active_min_abs": active_min_abs, "high_frequency_top_n": high_frequency_top_n,
            }, "python": sys.version, "numpy_version": np.__version__, "pyarrow_version": pa.__version__,
        }
        temporary["activation_manifest.json"].write_text(
            json.dumps(_json_safe(analysis_manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        for name in OUTPUT_FILENAMES:
            os.replace(temporary[name], final[name])
        return summary
    except BaseException:
        try:
            event_writer.close()
        except Exception:
            pass
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--active-top-k", type=int, default=128)
    parser.add_argument("--active-min-abs", type=float, default=0.0)
    parser.add_argument("--high-frequency-top-n", type=int, default=20)
    parser.add_argument("--examples-per-feature", type=int, default=8)
    parser.add_argument("--enriched-term-count", type=int, default=8)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    analyze_feature_activations(
        args.run_dir, args.output_dir, input_glob=args.input_glob, batch_size=args.batch_size,
        active_top_k=args.active_top_k, active_min_abs=args.active_min_abs,
        high_frequency_top_n=args.high_frequency_top_n, examples_per_feature=args.examples_per_feature,
        enriched_term_count=args.enriched_term_count, dataset_path=args.dataset,
        max_samples=args.max_samples, overwrite=args.overwrite, progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
