"""Exact cosine greedy selection on the shared absolute response-mean patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SelectionResult:
    """Greedy decisions indexed by active-row position."""

    selected: np.ndarray
    representative_sample_ids: np.ndarray
    representative_cosines: np.ndarray
    selection_ranks: np.ndarray


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
    """Apply exact greedy decisions for one precomputed cosine block."""
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

        if math.isfinite(best_cosine) and best_cosine >= cosine_threshold:
            result.representative_sample_ids[active_row] = best_representative
            result.representative_cosines[active_row] = np.float32(min(1.0, max(-1.0, best_cosine)))
        else:
            result.selected[active_row] = True
            result.selection_ranks[active_row] = len(selected_order_positions)
            selected_order_positions.append(ordered_position)
            selected_locals.append(local)


def _greedy_select_cpu(
    normalized: np.ndarray,
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
        if previous_positions.size:
            all_previous_cosines = block @ ordered_vectors[:block_start].T
            previous_cosines = all_previous_cosines[:, previous_positions]
        else:
            previous_cosines = np.empty((block_stop - block_start, 0), dtype=np.float32)
        within_cosines = block @ block.T
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
    normalized: np.ndarray,
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
    ordered_dense = np.ascontiguousarray(ordered)
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


def select_with_device(
    normalized: np.ndarray,
    sample_ids: np.ndarray,
    signal_strengths: np.ndarray,
    *,
    cosine_threshold: float,
    block_size: int,
    requested_device: str,
) -> tuple[SelectionResult, dict[str, Any]]:
    """Choose CUDA when available, otherwise retain the exact CPU path."""
    cpu_backend = "numpy_dense"
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
