#!/usr/bin/env python3
"""Select a non-redundant core set from Token-TopK response-mean vectors.

This is the code implementation of ``selection_v1.2.md``. It reads only the
sample id, sparse response-mean representation, and Token-TopK zero-signal
flag. Samples are processed by descending response-mean L2 norm and removed
when their exact signed cosine with an already selected sample reaches the
configured threshold.

CUDA is used automatically when available and when the dense normalized
matrix fits in free GPU memory. Otherwise the same selection runs with SciPy
on CPU. Parquet loading and sparse validation always remain on CPU.

Example:
    python select_v1_2.py \
        --run-dir SAERL-0801/runs/selection/qwen2_5/20260815-030007-646291 \
        --cosine-threshold 0.98
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp

INPUT_COLUMNS = (
    "sample_id",
    "topk_response_feat_ids",
    "topk_response_feat_mean",
    "zero_signal_token_topk",
)
EXPECTED_TYPES = {
    "sample_id": pa.int64(),
    "topk_response_feat_ids": pa.list_(pa.int64()),
    "topk_response_feat_mean": pa.list_(pa.float32()),
    "zero_signal_token_topk": pa.bool_(),
}


@dataclass(frozen=True)
class LoadedRepresentations:
    """Validated response-mean vectors and row metadata."""

    feature_size: int
    zero_eps: float
    sample_ids: np.ndarray
    signal_strengths: np.ndarray
    zero_signal_flags: np.ndarray
    active_all_rows: np.ndarray
    active_sample_ids: np.ndarray
    active_signal_strengths: np.ndarray
    normalized_active: sp.csr_matrix | np.ndarray
    shard_metadata: list[dict[str, Any]]


@dataclass(frozen=True)
class SelectionResult:
    """Greedy decisions indexed by active-row position."""

    selected: np.ndarray
    representative_sample_ids: np.ndarray
    representative_cosines: np.ndarray
    selection_ranks: np.ndarray


def _load_manifest(run_dir: Path) -> tuple[dict[str, Any], int, float]:
    path = run_dir / "representation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    feature_size = int(manifest.get("sae_feature_size", 0))
    if feature_size < 1:
        raise ValueError(f"Invalid sae_feature_size in {path}: {feature_size}")
    layout = tuple(manifest.get("representation_layout", ()))
    if "response_mean" not in layout:
        raise ValueError(f"representation_layout in {path} has no response_mean: {layout!r}")
    expected_size = 4 * feature_size
    if int(manifest.get("topk_representation_size", -1)) != expected_size:
        raise ValueError(
            f"topk_representation_size in {path} must be 4F={expected_size}, "
            f"got {manifest.get('topk_representation_size')!r}"
        )
    zero_eps = float(manifest.get("gradient_zero_eps", 1e-12))
    if not math.isfinite(zero_eps) or zero_eps < 0.0:
        raise ValueError(f"Invalid gradient_zero_eps in {path}: {zero_eps}")
    return manifest, feature_size, zero_eps


def _validate_schema(schema: pa.Schema, path: Path) -> None:
    for name, expected in EXPECTED_TYPES.items():
        index = schema.get_field_index(name)
        if index < 0:
            raise ValueError(f"Missing required column {name!r} in {path}")
        actual = schema.field(index).type
        if actual != expected:
            raise TypeError(f"Column {name!r} in {path} has type {actual}, expected {expected}")


def _discover_shards(run_dir: Path, input_glob: str) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(
        path for path in run_dir.glob(input_glob) if path.is_file() and not path.name.endswith(".inprogress")
    )
    if not paths:
        raise FileNotFoundError(f"No input shards match {input_glob!r} under {run_dir}")
    first_schema = pq.ParquetFile(paths[0]).schema_arrow
    _validate_schema(first_schema, paths[0])
    metadata: list[dict[str, Any]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(first_schema):
            raise ValueError(f"Parquet schema mismatch: {path}")
        metadata.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "rows": int(parquet.metadata.num_rows),
                "row_groups": int(parquet.metadata.num_row_groups),
            }
        )
    return paths, metadata


def _primitive_numpy(array: pa.Array, dtype: np.dtype) -> np.ndarray:
    if array.null_count:
        raise ValueError("Null values are not supported in representation columns")
    return array.to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _ragged_numpy(array: pa.Array, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    if array.null_count:
        raise ValueError("Null lists are not supported in representation columns")
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise TypeError(f"Expected a list array, got {array.type}")
    if array.values.null_count:
        raise ValueError("Null elements are not supported in representation lists")
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = array.values.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    return offsets, values


def _load_representations(
    paths: list[Path],
    shard_metadata: list[dict[str, Any]],
    *,
    feature_size: int,
    zero_eps: float,
    read_batch_size: int,
) -> LoadedRepresentations:
    sample_ids: list[int] = []
    signal_strengths: list[float] = []
    zero_signal_flags: list[bool] = []
    active_all_rows: list[int] = []
    active_sample_ids: list[int] = []
    active_signal_strengths: list[float] = []
    active_indices: list[np.ndarray] = []
    active_values: list[np.ndarray] = []
    active_indptr = [0]
    seen_sample_ids: set[int] = set()

    for shard_number, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=list(INPUT_COLUMNS),
            use_threads=True,
        ):
            ids_array = batch.column(batch.schema.get_field_index("topk_response_feat_ids"))
            mean_array = batch.column(batch.schema.get_field_index("topk_response_feat_mean"))
            id_offsets, feature_ids = _ragged_numpy(ids_array, np.int64)
            mean_offsets, mean_values = _ragged_numpy(mean_array, np.float32)
            batch_sample_ids = _primitive_numpy(batch.column(batch.schema.get_field_index("sample_id")), np.int64)
            batch_zero_flags = _primitive_numpy(
                batch.column(batch.schema.get_field_index("zero_signal_token_topk")), np.bool_
            )

            for row in range(batch.num_rows):
                sample_id = int(batch_sample_ids[row])
                if sample_id in seen_sample_ids:
                    raise ValueError(f"Duplicate sample_id across extraction shards: {sample_id}")
                seen_sample_ids.add(sample_id)

                ids = feature_ids[int(id_offsets[row]) : int(id_offsets[row + 1])]
                values = mean_values[int(mean_offsets[row]) : int(mean_offsets[row + 1])]
                if ids.size != values.size:
                    raise ValueError(
                        f"sample_id={sample_id} has inconsistent response arrays: ids={ids.size}, mean={values.size}"
                    )
                if ids.size and (ids[0] < 0 or ids[-1] >= feature_size):
                    raise ValueError(f"sample_id={sample_id} response feature ids fall outside [0, {feature_size})")
                if ids.size > 1 and np.any(ids[1:] <= ids[:-1]):
                    raise ValueError(f"sample_id={sample_id} response feature ids must be strictly increasing")
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"sample_id={sample_id} response mean contains NaN or infinity")

                values64 = values.astype(np.float64, copy=False)
                norm = math.sqrt(max(float(np.dot(values64, values64)), 0.0))
                zero_flag = bool(batch_zero_flags[row])
                all_row = len(sample_ids)
                sample_ids.append(sample_id)
                signal_strengths.append(norm)
                zero_signal_flags.append(zero_flag)

                if zero_flag or norm <= zero_eps:
                    continue
                nonzero = values != 0.0
                row_ids = ids[nonzero].astype(np.int64, copy=True)
                row_values = (values[nonzero] / norm).astype(np.float32, copy=True)
                active_all_rows.append(all_row)
                active_sample_ids.append(sample_id)
                active_signal_strengths.append(norm)
                active_indices.append(row_ids)
                active_values.append(row_values)
                active_indptr.append(active_indptr[-1] + int(row_ids.size))

        print(
            f"Loaded shard {shard_number}/{len(paths)}: {path.name} ({len(sample_ids):,} total samples)",
            flush=True,
        )

    if not sample_ids:
        raise ValueError("No representation rows were found")

    if active_indices:
        indices = np.concatenate(active_indices)
        values = np.concatenate(active_values)
    else:
        indices = np.empty(0, dtype=np.int64)
        values = np.empty(0, dtype=np.float32)
    normalized = sp.csr_matrix(
        (values, indices, np.asarray(active_indptr, dtype=np.int64)),
        shape=(len(active_sample_ids), feature_size),
        dtype=np.float32,
    )
    normalized.sort_indices()

    return LoadedRepresentations(
        feature_size=feature_size,
        zero_eps=zero_eps,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        signal_strengths=np.asarray(signal_strengths, dtype=np.float64),
        zero_signal_flags=np.asarray(zero_signal_flags, dtype=np.bool_),
        active_all_rows=np.asarray(active_all_rows, dtype=np.int64),
        active_sample_ids=np.asarray(active_sample_ids, dtype=np.int64),
        active_signal_strengths=np.asarray(active_signal_strengths, dtype=np.float64),
        normalized_active=normalized,
        shard_metadata=shard_metadata,
    )


class _InsufficientCudaMemory(RuntimeError):
    """Raised before allocation when the normalized matrix will not fit."""


def _empty_selection_result(count: int) -> SelectionResult:
    return SelectionResult(
        selected=np.zeros(count, dtype=np.bool_),
        representative_sample_ids=np.full(count, -1, dtype=np.int64),
        representative_cosines=np.full(count, np.nan, dtype=np.float32),
        selection_ranks=np.full(count, -1, dtype=np.int64),
    )


def _apply_greedy_block(
    *,
    block_start: int,
    block_stop: int,
    order: np.ndarray,
    ordered_sample_ids: np.ndarray,
    previous_positions: np.ndarray,
    previous_cosines: np.ndarray,
    within_cosines: np.ndarray,
    cosine_threshold: float,
    selected_order_positions: list[int],
    result: SelectionResult,
) -> None:
    """Apply v1.2 decisions for one precomputed cosine block."""
    selected_locals: list[int] = []
    for local in range(block_stop - block_start):
        ordered_position = block_start + local
        active_row = int(order[ordered_position])
        best_cosine = -math.inf
        best_representative = -1

        if previous_positions.size:
            previous_row = previous_cosines[local]
            best_previous = int(np.argmax(previous_row))
            best_cosine = float(previous_row[best_previous])
            best_representative = int(ordered_sample_ids[previous_positions[best_previous]])

        if selected_locals:
            local_cosines = within_cosines[local, np.asarray(selected_locals, dtype=np.int64)]
            best_local_at = int(np.argmax(local_cosines))
            best_local_cosine = float(local_cosines[best_local_at])
            if best_local_cosine > best_cosine:
                best_cosine = best_local_cosine
                best_representative = int(ordered_sample_ids[block_start + selected_locals[best_local_at]])

        if best_representative >= 0 and best_cosine >= cosine_threshold:
            result.representative_sample_ids[active_row] = best_representative
            result.representative_cosines[active_row] = np.float32(min(1.0, max(-1.0, best_cosine)))
        else:
            result.selected[active_row] = True
            result.selection_ranks[active_row] = len(selected_order_positions)
            selected_order_positions.append(ordered_position)
            selected_locals.append(local)


def _greedy_select_cpu(
    normalized: sp.csr_matrix | np.ndarray,
    sample_ids: np.ndarray,
    signal_strengths: np.ndarray,
    *,
    cosine_threshold: float,
    block_size: int,
) -> SelectionResult:
    count = int(sample_ids.size)
    result = _empty_selection_result(count)
    if count == 0:
        return result

    order = np.lexsort((sample_ids, -signal_strengths)).astype(np.int64, copy=False)
    ordered_vectors = normalized[order]
    ordered_sample_ids = sample_ids[order]
    selected_order_positions: list[int] = []

    for block_start in range(0, count, block_size):
        block_stop = min(count, block_start + block_size)
        block = ordered_vectors[block_start:block_stop]
        previous_positions = np.asarray(selected_order_positions, dtype=np.int64)
        if previous_positions.size and isinstance(ordered_vectors, np.ndarray):
            all_previous_cosines = block @ ordered_vectors[:block_start].T
            previous_cosines = all_previous_cosines[:, previous_positions]
        elif previous_positions.size:
            previous_cosines = (block @ ordered_vectors[previous_positions].T).toarray()
        else:
            previous_cosines = np.empty((block_stop - block_start, 0), dtype=np.float32)
        within_product = block @ block.T
        within_cosines = within_product.toarray() if sp.issparse(within_product) else within_product
        _apply_greedy_block(
            block_start=block_start,
            block_stop=block_stop,
            order=order,
            ordered_sample_ids=ordered_sample_ids,
            previous_positions=previous_positions,
            previous_cosines=previous_cosines,
            within_cosines=within_cosines,
            cosine_threshold=cosine_threshold,
            selected_order_positions=selected_order_positions,
            result=result,
        )

        print(
            f"Selected through {block_stop:,}/{count:,} active samples; kept {len(selected_order_positions):,}",
            flush=True,
        )

    return result


def _greedy_select_torch_dense(
    normalized: sp.csr_matrix | np.ndarray,
    sample_ids: np.ndarray,
    signal_strengths: np.ndarray,
    *,
    cosine_threshold: float,
    block_size: int,
    torch: Any,
    device: Any,
) -> tuple[SelectionResult, dict[str, Any]]:
    """Run exact full-vector cosine blocks with dense PyTorch matmul."""
    count, feature_size = normalized.shape
    result = _empty_selection_result(int(count))
    if count == 0:
        return result, {"dense_matrix_bytes": 0, "gpu_free_bytes_before": None}

    dense_bytes = int(count) * int(feature_size) * np.dtype(np.float32).itemsize
    workspace_bytes = int(block_size) * (int(count) + int(block_size)) * np.dtype(np.float32).itemsize
    free_bytes: int | None = None
    total_bytes: int | None = None
    if device.type == "cuda":
        free_bytes, total_bytes = (int(value) for value in torch.cuda.mem_get_info(device))
        reserve_bytes = max(512 * 1024**2, int(0.05 * total_bytes))
        required_bytes = dense_bytes + workspace_bytes + reserve_bytes
        if required_bytes > free_bytes:
            raise _InsufficientCudaMemory(
                "CUDA does not have enough free memory for the dense normalized matrix: "
                f"required≈{required_bytes / 1024**3:.2f} GiB, free={free_bytes / 1024**3:.2f} GiB"
            )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

    order = np.lexsort((sample_ids, -signal_strengths)).astype(np.int64, copy=False)
    ordered_sample_ids = sample_ids[order]
    ordered = normalized[order]
    ordered_dense = ordered.toarray(order="C") if sp.issparse(ordered) else np.ascontiguousarray(ordered)
    ordered_dense = ordered_dense.astype(np.float32, copy=False)
    del ordered
    ordered_vectors = torch.from_numpy(ordered_dense).to(device=device)
    del ordered_dense
    selected_order_positions: list[int] = []

    with torch.inference_mode():
        for block_start in range(0, int(count), block_size):
            block_stop = min(int(count), block_start + block_size)
            block = ordered_vectors[block_start:block_stop]
            previous_positions = np.asarray(selected_order_positions, dtype=np.int64)
            if previous_positions.size:
                all_previous_cosines = (block @ ordered_vectors[:block_start].T).cpu().numpy()
                previous_cosines = all_previous_cosines[:, previous_positions]
            else:
                previous_cosines = np.empty((block_stop - block_start, 0), dtype=np.float32)
            within_cosines = (block @ block.T).cpu().numpy()
            _apply_greedy_block(
                block_start=block_start,
                block_stop=block_stop,
                order=order,
                ordered_sample_ids=ordered_sample_ids,
                previous_positions=previous_positions,
                previous_cosines=previous_cosines,
                within_cosines=within_cosines,
                cosine_threshold=cosine_threshold,
                selected_order_positions=selected_order_positions,
                result=result,
            )
            print(
                f"Selected through {block_stop:,}/{count:,} active samples; kept {len(selected_order_positions):,}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, {
        "dense_matrix_bytes": dense_bytes,
        "similarity_workspace_bytes_estimate": workspace_bytes,
        "gpu_free_bytes_before": free_bytes,
        "gpu_total_bytes": total_bytes,
    }


def _select_with_device(
    normalized: sp.csr_matrix | np.ndarray,
    sample_ids: np.ndarray,
    signal_strengths: np.ndarray,
    *,
    cosine_threshold: float,
    block_size: int,
    requested_device: str,
) -> tuple[SelectionResult, dict[str, Any]]:
    """Choose CUDA when available, otherwise retain the exact CPU path."""
    cpu_backend = "numpy_dense" if isinstance(normalized, np.ndarray) else "scipy_sparse"
    if requested_device == "cpu":
        result = _greedy_select_cpu(
            normalized,
            sample_ids,
            signal_strengths,
            cosine_threshold=cosine_threshold,
            block_size=block_size,
        )
        return result, {"requested": requested_device, "device": "cpu", "backend": cpu_backend}

    try:
        import torch
    except ImportError as exc:
        if requested_device == "cuda":
            raise RuntimeError("--device cuda requires PyTorch with CUDA support") from exc
        print("PyTorch is unavailable; falling back to CPU", flush=True)
        result = _greedy_select_cpu(
            normalized,
            sample_ids,
            signal_strengths,
            cosine_threshold=cosine_threshold,
            block_size=block_size,
        )
        return result, {
            "requested": requested_device,
            "device": "cpu",
            "backend": cpu_backend,
            "fallback_reason": "PyTorch is unavailable",
        }

    if not torch.cuda.is_available():
        if requested_device == "cuda":
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
        print("CUDA is unavailable; using CPU", flush=True)
        result = _greedy_select_cpu(
            normalized,
            sample_ids,
            signal_strengths,
            cosine_threshold=cosine_threshold,
            block_size=block_size,
        )
        return result, {
            "requested": requested_device,
            "device": "cpu",
            "backend": cpu_backend,
            "fallback_reason": "CUDA is unavailable",
            "torch_version": torch.__version__,
        }

    device = torch.device("cuda", torch.cuda.current_device())
    try:
        result, memory = _greedy_select_torch_dense(
            normalized,
            sample_ids,
            signal_strengths,
            cosine_threshold=cosine_threshold,
            block_size=block_size,
            torch=torch,
            device=device,
        )
    except (torch.cuda.OutOfMemoryError, _InsufficientCudaMemory) as exc:
        if requested_device == "cuda":
            raise
        torch.cuda.empty_cache()
        reason = str(exc)
        print(f"CUDA memory is insufficient; falling back to CPU: {reason}", flush=True)
        result = _greedy_select_cpu(
            normalized,
            sample_ids,
            signal_strengths,
            cosine_threshold=cosine_threshold,
            block_size=block_size,
        )
        return result, {
            "requested": requested_device,
            "device": "cpu",
            "backend": cpu_backend,
            "fallback_reason": reason,
            "torch_version": torch.__version__,
        }

    return result, {
        "requested": requested_device,
        "device": str(device),
        "backend": "torch_dense",
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(device),
        **memory,
    }


def _hash_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _atomic_write_parquet(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
    os.replace(temporary, path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_outputs(
    output_dir: Path,
    loaded: LoadedRepresentations,
    result: SelectionResult,
    *,
    run_dir: Path,
    input_glob: str,
    cosine_threshold: float,
    block_size: int,
    compute: dict[str, Any],
    method: str,
    input_columns: tuple[str, ...],
    representation: str,
    zero_signal_field: str,
    elapsed_seconds: float,
    signal_strength_description: str = "L2 norm of response mean",
    similarity_description: str = "exact signed cosine on the full response-mean vector",
    processing_order_description: str = "descending signal strength, then ascending sample_id",
    silent_zero_reason: str = "response_mean_zero",
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total = int(loaded.sample_ids.size)
    statuses = np.full(total, "silent", dtype=object)
    representative_ids = np.full(total, -1, dtype=np.int64)
    representative_cosines = np.full(total, np.nan, dtype=np.float32)
    selection_ranks = np.full(total, -1, dtype=np.int64)
    active_rows = loaded.active_all_rows
    statuses[active_rows] = np.where(result.selected, "selected", "redundant")
    representative_ids[active_rows] = result.representative_sample_ids
    representative_cosines[active_rows] = result.representative_cosines
    selection_ranks[active_rows] = result.selection_ranks

    silent_reasons: list[str | None] = []
    for flag, norm in zip(loaded.zero_signal_flags, loaded.signal_strengths, strict=True):
        reasons: list[str] = []
        if bool(flag):
            reasons.append(zero_signal_field)
        if float(norm) <= loaded.zero_eps:
            reasons.append(silent_zero_reason)
        silent_reasons.append("|".join(reasons) if reasons else None)

    sorted_rows = np.argsort(loaded.sample_ids, kind="stable")
    audit = pa.table(
        {
            "sample_id": pa.array(loaded.sample_ids[sorted_rows], type=pa.int64()),
            "signal_strength": pa.array(loaded.signal_strengths[sorted_rows], type=pa.float64()),
            "status": pa.array(statuses[sorted_rows].tolist(), type=pa.string()),
            "selection_rank": pa.array(
                [None if selection_ranks[row] < 0 else int(selection_ranks[row]) for row in sorted_rows],
                type=pa.int64(),
            ),
            "representative_sample_id": pa.array(
                [None if representative_ids[row] < 0 else int(representative_ids[row]) for row in sorted_rows],
                type=pa.int64(),
            ),
            "representative_cosine": pa.array(
                [
                    None if not np.isfinite(representative_cosines[row]) else float(representative_cosines[row])
                    for row in sorted_rows
                ],
                type=pa.float32(),
            ),
            zero_signal_field: pa.array(loaded.zero_signal_flags[sorted_rows], type=pa.bool_()),
            "silent_reason": pa.array([silent_reasons[row] for row in sorted_rows], type=pa.string()),
        }
    )

    selected_ids = np.sort(loaded.active_sample_ids[result.selected])
    silent_mask = statuses == "silent"
    silent_ids = np.sort(loaded.sample_ids[silent_mask])
    selected_table = pa.table({"sample_id": pa.array(selected_ids, type=pa.int64())})
    silent_table = pa.table({"sample_id": pa.array(silent_ids, type=pa.int64())})

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "selection_audit.parquet"
    selected_path = output_dir / "selected_sample_ids.parquet"
    silent_path = output_dir / "silent_sample_ids.parquet"
    _atomic_write_parquet(audit, audit_path)
    _atomic_write_parquet(selected_table, selected_path)
    _atomic_write_parquet(silent_table, silent_path)

    redundant_count = int(np.count_nonzero(statuses == "redundant"))
    manifest = {
        "method": method,
        "run_dir": str(run_dir),
        "input_glob": input_glob,
        "input_columns": list(input_columns),
        "feature_size": loaded.feature_size,
        "representation": representation,
        "signal_strength": signal_strength_description,
        "similarity": similarity_description,
        "processing_order": processing_order_description,
        "cosine_threshold": cosine_threshold,
        "similarity_block_size": block_size,
        "compute": compute,
        "zero_eps": loaded.zero_eps,
        "counts": {
            "total": total,
            "active": int(active_rows.size),
            "selected": int(selected_ids.size),
            "redundant": redundant_count,
            "silent": int(silent_ids.size),
        },
        "retention_ratio_of_active": (float(selected_ids.size / active_rows.size) if active_rows.size else 0.0),
        "selected_sample_ids_sha256": _hash_int64(selected_ids),
        "elapsed_seconds": elapsed_seconds,
        "input_shards": loaded.shard_metadata,
        "outputs": {
            "selected_sample_ids": str(selected_path),
            "silent_sample_ids": str(silent_path),
            "audit": str(audit_path),
        },
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    _atomic_write_json(manifest, output_dir / "selection_manifest.json")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Gradient-SAE extraction run directory")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: RUN_DIR/selection_v1_2)",
    )
    parser.add_argument("--input-glob", default="step-*.parquet")
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Compute device; auto uses CUDA when available and CPU otherwise",
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=4,
        help="Rows read from Parquet together; sparse union rows can be large",
    )
    parser.add_argument(
        "--similarity-block-size",
        type=int,
        default=64,
        help="Rows per exact sparse cosine block",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not -1.0 <= float(args.cosine_threshold) <= 1.0:
        raise ValueError("--cosine-threshold must be in [-1, 1]")
    if int(args.read_batch_size) < 1 or int(args.similarity_block_size) < 1:
        raise ValueError("--read-batch-size and --similarity-block-size must be positive")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else run_dir / "selection_v1_2"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    started = time.monotonic()
    _manifest, feature_size, zero_eps = _load_manifest(run_dir)
    paths, shard_metadata = _discover_shards(run_dir, str(args.input_glob))
    loaded = _load_representations(
        paths,
        shard_metadata,
        feature_size=feature_size,
        zero_eps=zero_eps,
        read_batch_size=int(args.read_batch_size),
    )
    result, compute = _select_with_device(
        loaded.normalized_active,
        loaded.active_sample_ids,
        loaded.active_signal_strengths,
        cosine_threshold=float(args.cosine_threshold),
        block_size=int(args.similarity_block_size),
        requested_device=str(args.device),
    )
    elapsed = time.monotonic() - started
    manifest = _write_outputs(
        output_dir,
        loaded,
        result,
        run_dir=run_dir,
        input_glob=str(args.input_glob),
        cosine_threshold=float(args.cosine_threshold),
        block_size=int(args.similarity_block_size),
        compute=compute,
        method="selection_v1.2",
        input_columns=INPUT_COLUMNS,
        representation="sample-level Token-TopK response mean",
        zero_signal_field="zero_signal_token_topk",
        elapsed_seconds=elapsed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
