"""Shared selected-feature response-mean loading and output utilities."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from saerl.selection.select_topk_response_mean_selected_features_v1_2_2 import (
    _load_feature_indices,
)
from saerl.selection.select_topk_response_mean_v1_2 import (
    INPUT_COLUMNS,
    _discover_shards,
    _hash_int64,
    _load_manifest,
    _primitive_numpy,
    _ragged_numpy,
)


@dataclass(frozen=True)
class Projection:
    run_dir: Path
    run_dirs: tuple[Path, ...]
    input_glob: str
    feature_indices_path: Path
    source_feature_size: int
    feature_indices: np.ndarray
    zero_eps: float
    sample_ids: np.ndarray
    values: np.ndarray
    strengths: np.ndarray
    zero_flags: np.ndarray
    active_rows: np.ndarray
    shard_metadata: list[dict[str, Any]]
    stats: dict[str, int]

    @property
    def feature_size(self) -> int:
        return int(self.feature_indices.size)


def _log_at(current: int, total: int) -> bool:
    interval = max(1, total // 20)
    return current == 1 or current == total or current % interval == 0


def load_projection(
    run_dir: Path | list[Path],
    feature_indices_path: Path,
    *,
    input_glob: str,
    read_batch_size: int,
) -> Projection:
    """Load unchanged response-mean values into the NPY-defined coordinates."""
    run_dirs = run_dir if isinstance(run_dir, list) else [run_dir]
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    run_dirs = [Path(path).expanduser().resolve() for path in run_dirs]
    run_dir = run_dirs[0]
    feature_indices_path = Path(feature_indices_path).expanduser().resolve()
    manifest_info = [_load_manifest(path) for path in run_dirs]
    source_sizes = [info[1] for info in manifest_info]
    zero_eps_values = [info[2] for info in manifest_info]
    if len(set(source_sizes)) != 1:
        raise ValueError(f"Inconsistent sae_feature_size: {source_sizes}")
    if len(set(zero_eps_values)) != 1:
        raise ValueError(f"Inconsistent gradient_zero_eps: {zero_eps_values}")
    source_feature_size, zero_eps = source_sizes[0], zero_eps_values[0]
    feature_indices = _load_feature_indices(feature_indices_path, source_feature_size)
    paths, shard_metadata = [], []
    for path in run_dirs:
        run_paths, run_metadata = _discover_shards(path, input_glob)
        paths.extend(run_paths)
        shard_metadata.extend(run_metadata)
    total = sum(int(item["rows"]) for item in shard_metadata)
    feature_size = int(feature_indices.size)
    mapping = np.full(source_feature_size, -1, dtype=np.int64)
    mapping[feature_indices] = np.arange(feature_size, dtype=np.int64)

    sample_ids = np.empty(total, dtype=np.int64)
    values = np.zeros((total, feature_size), dtype=np.float32)
    strengths = np.empty(total, dtype=np.float64)
    zero_flags = np.empty(total, dtype=np.bool_)
    observed = np.zeros(feature_size, dtype=np.bool_)
    seen: set[int] = set()
    loaded = 0

    print(
        f"[config] runs={len(run_dirs)} features={feature_indices_path} "
        f"source_F={source_feature_size:,} selected_K={feature_size:,}",
        flush=True,
    )
    print(
        f"[features] min={int(feature_indices.min()):,} max={int(feature_indices.max()):,} "
        f"sha256={_hash_int64(feature_indices)}",
        flush=True,
    )
    print(f"[discover] shards={len(paths):,} rows={total:,}", flush=True)

    for shard_number, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=list(INPUT_COLUMNS),
            use_threads=True,
        ):
            ids_offsets, ids = _ragged_numpy(batch.column(1), np.int64)
            mean_offsets, means = _ragged_numpy(batch.column(2), np.float32)
            batch_sample_ids = _primitive_numpy(batch.column(0), np.int64)
            batch_zero_flags = _primitive_numpy(batch.column(3), np.bool_)

            for row in range(batch.num_rows):
                sample_id = int(batch_sample_ids[row])
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id across extraction shards: {sample_id}")
                seen.add(sample_id)
                row_ids = ids[int(ids_offsets[row]) : int(ids_offsets[row + 1])]
                row_means = means[int(mean_offsets[row]) : int(mean_offsets[row + 1])]
                if row_ids.size != row_means.size:
                    raise ValueError(f"sample_id={sample_id} has mismatched response ids and means")
                if row_ids.size and (row_ids[0] < 0 or row_ids[-1] >= source_feature_size):
                    raise ValueError(f"sample_id={sample_id} feature ids are outside [0, {source_feature_size})")
                if row_ids.size > 1 and np.any(row_ids[1:] <= row_ids[:-1]):
                    raise ValueError(f"sample_id={sample_id} feature ids must be strictly increasing")
                if not np.all(np.isfinite(row_means)):
                    raise ValueError(f"sample_id={sample_id} response mean contains NaN or infinity")

                coordinates = mapping[row_ids]
                keep = coordinates >= 0
                selected_coordinates = coordinates[keep]
                selected_values = row_means[keep]
                values[loaded, selected_coordinates] = selected_values
                selected64 = selected_values.astype(np.float64, copy=False)
                strengths[loaded] = math.sqrt(max(float(np.dot(selected64, selected64)), 0.0))
                sample_ids[loaded] = sample_id
                zero_flags[loaded] = bool(batch_zero_flags[row])
                if selected_values.size:
                    observed[selected_coordinates[selected_values != 0.0]] = True
                loaded += 1

        if _log_at(shard_number, len(paths)):
            print(f"[project] shards={shard_number:,}/{len(paths):,} rows={loaded:,}/{total:,}", flush=True)

    if loaded != total:
        raise ValueError(f"Row-count mismatch: expected={total}, loaded={loaded}")
    active_rows = np.flatnonzero((~zero_flags) & (strengths > zero_eps)).astype(np.int64, copy=False)
    source_silent = int(np.count_nonzero(zero_flags))
    projected_zero = int(np.count_nonzero((~zero_flags) & (strengths <= zero_eps)))
    stats = {
        "source_zero_signal_samples": source_silent,
        "projected_zero_samples": projected_zero,
        "active_samples": int(active_rows.size),
        "observed_nonzero_selected_features": int(np.count_nonzero(observed)),
    }
    print(
        f"[project] complete active={active_rows.size:,} source-silent={source_silent:,} "
        f"projected-zero={projected_zero:,} observed-features={stats['observed_nonzero_selected_features']:,}",
        flush=True,
    )
    return Projection(
        run_dir=run_dir,
        run_dirs=tuple(run_dirs),
        input_glob=input_glob,
        feature_indices_path=feature_indices_path,
        source_feature_size=source_feature_size,
        feature_indices=feature_indices,
        zero_eps=zero_eps,
        sample_ids=sample_ids,
        values=values,
        strengths=strengths,
        zero_flags=zero_flags,
        active_rows=active_rows,
        shard_metadata=shard_metadata,
        stats=stats,
    )


def default_output_dir(projection: Projection, method: str) -> Path:
    return projection.run_dir / f"{method}_{projection.feature_indices_path.stem}"


def _atomic_parquet(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def persist_selection(
    output_dir: Path,
    projection: Projection,
    *,
    method: str,
    decision_rows: np.ndarray,
    selected: np.ndarray,
    selection_ranks: np.ndarray,
    threshold_name: str,
    threshold: float,
    score_name: str,
    scores: np.ndarray,
    representatives: np.ndarray | None,
    upper_bounds: np.ndarray | None,
    compute: dict[str, Any],
    details: dict[str, Any],
    elapsed_seconds: float,
    representation: str = "Token-TopK response mean projected to selected features",
    signal_strength_description: str = "L2 norm of unchanged selected-feature response-mean values",
) -> dict[str, Any]:
    """Persist a common selected/silent/audit/manifest output set."""
    output_dir = Path(output_dir).expanduser().resolve()
    total = int(projection.sample_ids.size)
    active_count = int(projection.active_rows.size)
    if decision_rows.size != active_count or selected.size != active_count:
        raise ValueError("Selection result size does not match projected active rows")
    if not np.array_equal(np.sort(decision_rows), projection.active_rows):
        raise ValueError("Selection decision rows do not match projected active rows")

    statuses = np.full(total, "silent", dtype=object)
    ranks = np.full(total, -1, dtype=np.int64)
    score_values = np.full(total, np.nan, dtype=np.float64)
    statuses[decision_rows] = np.where(selected, "selected", "redundant")
    ranks[decision_rows] = selection_ranks
    score_values[decision_rows] = scores

    representative_values = np.full(total, -1, dtype=np.int64)
    if representatives is not None:
        representative_values[decision_rows] = representatives
    upper_values = np.full(total, np.nan, dtype=np.float64)
    if upper_bounds is not None:
        upper_values[decision_rows] = upper_bounds

    silent_reasons: list[str | None] = []
    for flag, strength in zip(projection.zero_flags, projection.strengths, strict=True):
        reasons: list[str] = []
        if bool(flag):
            reasons.append("zero_signal_token_topk")
        if float(strength) <= projection.zero_eps:
            reasons.append("selected_feature_response_mean_zero")
        silent_reasons.append("|".join(reasons) if reasons else None)

    order = np.argsort(projection.sample_ids, kind="stable")
    columns: dict[str, Any] = {
        "sample_id": pa.array(projection.sample_ids[order], type=pa.int64()),
        "selected_feature_response_mean_norm": pa.array(projection.strengths[order], type=pa.float64()),
        "status": pa.array(statuses[order].tolist(), type=pa.string()),
        "selection_rank": pa.array(
            [None if ranks[row] < 0 else int(ranks[row]) for row in order],
            type=pa.int64(),
        ),
        score_name: pa.array(
            [None if not np.isfinite(score_values[row]) else float(score_values[row]) for row in order],
            type=pa.float64(),
        ),
    }
    if representatives is not None:
        columns["representative_sample_id"] = pa.array(
            [None if representative_values[row] < 0 else int(representative_values[row]) for row in order],
            type=pa.int64(),
        )
    if upper_bounds is not None:
        columns["max_upper_bound"] = pa.array(
            [None if not np.isfinite(upper_values[row]) else float(upper_values[row]) for row in order],
            type=pa.float64(),
        )
    columns["zero_signal_token_topk"] = pa.array(projection.zero_flags[order], type=pa.bool_())
    columns["silent_reason"] = pa.array([silent_reasons[row] for row in order], type=pa.string())
    audit = pa.table(columns)

    selected_ids = np.sort(projection.sample_ids[decision_rows[selected]])
    silent_ids = np.sort(projection.sample_ids[statuses == "silent"])
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_sample_ids.parquet"
    silent_path = output_dir / "silent_sample_ids.parquet"
    audit_path = output_dir / "selection_audit.parquet"
    _atomic_parquet(pa.table({"sample_id": pa.array(selected_ids, type=pa.int64())}), selected_path)
    _atomic_parquet(pa.table({"sample_id": pa.array(silent_ids, type=pa.int64())}), silent_path)
    _atomic_parquet(audit, audit_path)

    manifest = {
        "method": method,
        "run_dir": str(projection.run_dir),
        "run_dirs": [str(path) for path in projection.run_dirs],
        "input_glob": projection.input_glob,
        "input_columns": list(INPUT_COLUMNS),
        "representation": representation,
        "source_sae_feature_size": projection.source_feature_size,
        "selected_feature_size": projection.feature_size,
        "feature_indices_file": str(projection.feature_indices_path),
        "feature_indices_sha256": _hash_int64(projection.feature_indices),
        "selected_feature_indices": projection.feature_indices.tolist(),
        "projection": projection.stats,
        "signal_strength": signal_strength_description,
        "processing_order": "descending projected signal strength, then ascending sample_id",
        threshold_name: threshold,
        "compute": compute,
        "details": details,
        "zero_eps": projection.zero_eps,
        "counts": {
            "total": total,
            "active": active_count,
            "selected": int(selected_ids.size),
            "redundant": int(active_count - selected_ids.size),
            "silent": int(silent_ids.size),
        },
        "retention_ratio_of_active": float(selected_ids.size / active_count) if active_count else 0.0,
        "selected_sample_ids_sha256": _hash_int64(selected_ids),
        "elapsed_seconds": elapsed_seconds,
        "input_shards": projection.shard_metadata,
        "outputs": {
            "selected_sample_ids": str(selected_path),
            "silent_sample_ids": str(silent_path),
            "audit": str(audit_path),
        },
    }
    _atomic_json(manifest, output_dir / "selection_manifest.json")
    print(f"[done] output={output_dir} counts={manifest['counts']}", flush=True)
    return manifest
