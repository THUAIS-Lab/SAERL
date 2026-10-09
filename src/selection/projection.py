"""Project shared SAE features and persist core-set decisions for auditing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from saerl.selection.exact_cosine import SelectionResult
from saerl.selection.io import INPUT_COLUMNS, hash_int64, iter_response_means, load_feature_indices, load_runs


@dataclass(frozen=True)
class Projection:
    run_dirs: list[Path]
    feature_indices_path: Path
    source_feature_size: int
    feature_indices: np.ndarray
    zero_eps: float
    sample_ids: np.ndarray
    values: np.ndarray
    strengths: np.ndarray
    zero_flags: np.ndarray
    active_rows: np.ndarray
    shard_metadata: list[dict]


def load_projection(run_dirs: list[Path], feature_indices_path: Path, read_batch_size: int) -> Projection:
    run_dirs, source_size, zero_eps, paths, metadata = load_runs(run_dirs)
    feature_indices_path = feature_indices_path.expanduser().resolve()
    feature_indices = load_feature_indices(feature_indices_path, source_size)
    total = sum(item["rows"] for item in metadata)
    values = np.zeros((total, feature_indices.size), dtype=np.float32)
    sample_ids = np.empty(total, dtype=np.int64)
    strengths = np.empty(total, dtype=np.float64)
    zero_flags = np.empty(total, dtype=np.bool_)
    mapping = np.full(source_size, -1, dtype=np.int64)
    mapping[feature_indices] = np.arange(feature_indices.size, dtype=np.int64)
    loaded = 0
    for row, (sample_id, ids, means, zero_flag) in enumerate(iter_response_means(paths, source_size, read_batch_size)):
        coordinates = mapping[ids]
        keep = coordinates >= 0
        selected_values = means[keep]
        values[row, coordinates[keep]] = selected_values
        selected64 = selected_values.astype(np.float64)
        strengths[row] = np.linalg.norm(selected64)
        sample_ids[row], zero_flags[row] = sample_id, zero_flag
        loaded += 1
    if loaded != total:
        raise ValueError(f"Row-count mismatch: expected={total}, loaded={loaded}")
    active = np.flatnonzero((~zero_flags) & (strengths > zero_eps))
    print(f"[project] rows={total:,} selected-features={feature_indices.size:,} active={active.size:,}", flush=True)
    return Projection(
        run_dirs,
        feature_indices_path,
        source_size,
        feature_indices,
        zero_eps,
        sample_ids,
        values,
        strengths,
        zero_flags,
        active,
        metadata,
    )


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
    os.replace(temporary, path)


def persist_selection(
    output_dir: Path,
    projection: Projection,
    result: SelectionResult,
    threshold: float,
    compute: dict,
    elapsed_seconds: float,
) -> dict:
    total = projection.sample_ids.size
    active = projection.active_rows
    if result.selected.size != active.size:
        raise ValueError("Selection result size does not match projected active rows")
    statuses = np.full(total, "silent", dtype=object)
    statuses[active] = np.where(result.selected, "selected", "redundant")
    ranks = np.full(total, -1, dtype=np.int64)
    ranks[active] = result.selection_ranks
    cosines = np.full(total, np.nan, dtype=np.float64)
    cosines[active] = result.representative_cosines
    representatives = np.full(total, -1, dtype=np.int64)
    representatives[active] = result.representative_sample_ids
    reasons = []
    for flag, strength in zip(projection.zero_flags, projection.strengths, strict=True):
        parts = []
        if flag:
            parts.append("zero_signal_token_topk")
        if strength <= projection.zero_eps:
            parts.append("selected_feature_response_mean_zero")
        reasons.append("|".join(parts) if parts else None)

    order = np.argsort(projection.sample_ids, kind="stable")
    audit = pa.table(
        {
            "sample_id": pa.array(projection.sample_ids[order], type=pa.int64()),
            "selected_feature_response_mean_norm": pa.array(projection.strengths[order], type=pa.float64()),
            "status": pa.array(statuses[order].tolist(), type=pa.string()),
            "selection_rank": pa.array([None if ranks[row] < 0 else int(ranks[row]) for row in order], type=pa.int64()),
            "representative_sample_id": pa.array(
                [int(representatives[row]) if statuses[row] == "redundant" else None for row in order], type=pa.int64()
            ),
            "representative_cosine": pa.array(
                [float(cosines[row]) if np.isfinite(cosines[row]) else None for row in order], type=pa.float64()
            ),
            "zero_signal_token_topk": pa.array(projection.zero_flags[order], type=pa.bool_()),
            "silent_reason": pa.array([reasons[row] for row in order], type=pa.string()),
        }
    )
    selected_ids = np.sort(projection.sample_ids[active[result.selected]])
    silent_ids = np.sort(projection.sample_ids[statuses == "silent"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in (("selected_sample_ids", selected_ids), ("silent_sample_ids", silent_ids)):
        _atomic_parquet(pa.table({"sample_id": pa.array(ids, type=pa.int64())}), output_dir / f"{name}.parquet")
    _atomic_parquet(audit, output_dir / "selection_audit.parquet")
    manifest = {
        "method": "absolute_response_mean_exact_cosine_greedy",
        "run_dirs": [str(path) for path in projection.run_dirs],
        "input_columns": list(INPUT_COLUMNS),
        "representation": "absolute Token-TopK response mean projected to shared selected features",
        "source_sae_feature_size": projection.source_feature_size,
        "selected_feature_size": int(projection.feature_indices.size),
        "feature_indices_file": str(projection.feature_indices_path),
        "feature_indices_sha256": hash_int64(projection.feature_indices),
        "selected_feature_indices": projection.feature_indices.tolist(),
        "processing_order": "descending projected signal strength, then ascending sample_id",
        "signal_strength": "L2 norm of selected-feature response-mean values",
        "cosine_threshold": threshold,
        "compute": compute,
        "zero_eps": projection.zero_eps,
        "counts": {
            "total": int(total),
            "active": int(active.size),
            "selected": int(selected_ids.size),
            "redundant": int(active.size - selected_ids.size),
            "silent": int(silent_ids.size),
        },
        "selected_sample_ids_sha256": hash_int64(selected_ids),
        "elapsed_seconds": elapsed_seconds,
        "input_shards": projection.shard_metadata,
    }
    temporary = output_dir / "selection_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "selection_manifest.json")
    return manifest
