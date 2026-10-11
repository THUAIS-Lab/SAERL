from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from saerl.selection.response_pattern import ResponsePatterns, project_response_patterns


@dataclass(frozen=True)
class SelectionResult:

    selected: np.ndarray
    representative_sample_ids: np.ndarray
    representative_cosines: np.ndarray
    selection_ranks: np.ndarray


def exact_cosine_greedy(
    normalized: np.ndarray,
    sample_ids: np.ndarray,
    signal_strengths: np.ndarray,
    *,
    cosine_threshold: float = 0.98,
    block_size: int = 64,
    device: str = "cpu",
) -> tuple[SelectionResult, dict]:
    use_cuda = False
    if device != "cpu":
        try:
            import torch
        except ImportError:
            if device == "cuda":
                raise
        else:
            use_cuda = device == "cuda" or torch.cuda.is_available()

    count = sample_ids.size
    result = SelectionResult(
        selected=np.zeros(count, dtype=np.bool_),
        representative_sample_ids=np.full(count, -1, dtype=np.int64),
        representative_cosines=np.full(count, np.nan, dtype=np.float32),
        selection_ranks=np.full(count, -1, dtype=np.int64),
    )
    order = np.lexsort((sample_ids, -signal_strengths))
    ordered_ids = sample_ids[order]
    vectors = np.ascontiguousarray(normalized[order], dtype=np.float32)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        vectors = torch.from_numpy(vectors).to("cuda")

    selected_positions: list[int] = []
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        block = vectors[start:stop]
        previous_positions = np.asarray(selected_positions, dtype=np.int64)
        if selected_positions:
            previous_vectors = vectors[previous_positions]
            previous_cosines = block @ previous_vectors.T
            if use_cuda:
                previous_cosines = previous_cosines.cpu().numpy()
        else:
            previous_cosines = np.empty((stop - start, 0), dtype=np.float32)
        within_cosines = block @ block.T
        if use_cuda:
            within_cosines = within_cosines.cpu().numpy()

        selected_locals: list[int] = []
        for local in range(stop - start):
            row = int(order[start + local])
            best_cosine, representative = -np.inf, -1
            if previous_positions.size:
                best = int(np.argmax(previous_cosines[local]))
                best_cosine = float(previous_cosines[local, best])
                representative = int(ordered_ids[previous_positions[best]])
            if selected_locals:
                cosines = within_cosines[local, selected_locals]
                best = int(np.argmax(cosines))
                if float(cosines[best]) > best_cosine:
                    best_cosine = float(cosines[best])
                    representative = int(ordered_ids[start + selected_locals[best]])
            if best_cosine >= cosine_threshold:
                result.representative_sample_ids[row] = representative
                result.representative_cosines[row] = np.clip(best_cosine, -1.0, 1.0)
            else:
                result.selected[row] = True
                result.selection_ranks[row] = len(selected_positions)
                selected_positions.append(start + local)
                selected_locals.append(local)
        print(f"Selected through {stop:,}/{count:,} active samples; kept {len(selected_positions):,}", flush=True)
    return result, {
        "requested": device,
        "device": "cuda" if use_cuda else "cpu",
        "backend": "torch_dense" if use_cuda else "numpy_dense",
    }


def save_core_set(
    output_dir: Path, patterns: ResponsePatterns, result: SelectionResult,
    threshold: float, compute: dict, elapsed_seconds: float,
) -> dict:
    active = patterns.active_rows
    total = patterns.sample_ids.size
    statuses = np.full(total, "silent", dtype=object)
    statuses[active] = np.where(result.selected, "selected", "redundant")
    ranks = np.full(total, -1, dtype=np.int64)
    ranks[active] = result.selection_ranks
    cosines = np.full(total, np.nan, dtype=np.float32)
    cosines[active] = result.representative_cosines
    representatives = np.full(total, -1, dtype=np.int64)
    representatives[active] = result.representative_sample_ids
    reasons = []
    for flag, strength in zip(patterns.zero_flags, patterns.strengths, strict=True):
        reason = []
        if flag:
            reason.append("zero_signal_token_topk")
        if strength <= patterns.runs.zero_eps:
            reason.append("selected_feature_response_mean_zero")
        reasons.append("|".join(reason) or None)
    order = np.argsort(patterns.sample_ids, kind="stable")
    audit = pa.table(
        {
            "sample_id": pa.array(patterns.sample_ids[order], type=pa.int64()),
            "selected_feature_response_mean_norm": pa.array(patterns.strengths[order], type=pa.float64()),
            "status": pa.array(statuses[order].tolist(), type=pa.string()),
            "selection_rank": pa.array([int(ranks[row]) if ranks[row] >= 0 else None for row in order], type=pa.int64()),
            "representative_sample_id": pa.array(
                [int(representatives[row]) if statuses[row] == "redundant" else None for row in order], type=pa.int64(),
            ),
            "representative_cosine": pa.array(
                [float(cosines[row]) if np.isfinite(cosines[row]) else None for row in order], type=pa.float64(),
            ),
            "zero_signal_token_topk": pa.array(patterns.zero_flags[order], type=pa.bool_()),
            "silent_reason": pa.array([reasons[row] for row in order], type=pa.string()),
        }
    )
    selected_ids = np.sort(patterns.sample_ids[active[result.selected]])
    silent_ids = np.sort(patterns.sample_ids[statuses == "silent"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in (("selected_sample_ids", selected_ids), ("silent_sample_ids", silent_ids)):
        pq.write_table(pa.table({"sample_id": pa.array(ids, type=pa.int64())}), output_dir / f"{name}.parquet")
    pq.write_table(audit, output_dir / "selection_audit.parquet")
    manifest = {
        "method": "absolute_response_mean_exact_cosine_greedy",
        "run_dirs": [str(path) for path in patterns.runs.directories],
        "selected_feature_indices": patterns.feature_indices.tolist(),
        "processing_order": "descending projected signal strength, then ascending sample_id",
        "signal_strength": "L2 norm of selected-feature response-mean values",
        "cosine_threshold": threshold,
        "compute": compute,
        "zero_eps": patterns.runs.zero_eps,
        "counts": {
            "total": int(total), "active": int(active.size), "selected": int(selected_ids.size),
            "redundant": int(active.size - selected_ids.size), "silent": int(silent_ids.size),
        },
        "elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct the gradient core set with exact cosine greedy selection.")
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--feature-indices", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--read-batch-size", type=int, default=4)
    parser.add_argument("--similarity-block-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.cosine_threshold <= 1:
        parser.error("--cosine-threshold must be in [0, 1]")
    if args.read_batch_size < 1 or args.similarity_block_size < 1:
        parser.error("batch and block sizes must be positive")
    output_dir = (args.output_dir or args.run_dir[0] / "selection").expanduser().resolve()
    started = time.monotonic()
    patterns = project_response_patterns(args.run_dir, args.feature_indices, args.read_batch_size)
    active = patterns.active_rows
    normalized = patterns.values[active]
    np.divide(normalized, patterns.strengths[active, None], out=normalized)
    result, compute = exact_cosine_greedy(
        normalized, patterns.sample_ids[active], patterns.strengths[active],
        cosine_threshold=args.cosine_threshold, block_size=args.similarity_block_size, device=args.device,
    )
    manifest = save_core_set(output_dir, patterns, result, args.cosine_threshold, compute, time.monotonic() - started)
    print(json.dumps({"counts": manifest["counts"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
