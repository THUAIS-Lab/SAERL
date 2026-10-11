from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from saerl.selection.feature_selection import ExtractionRuns, iter_response_means, load_runs


@dataclass(frozen=True)
class ResponsePatterns:
    runs: ExtractionRuns
    sample_ids: np.ndarray
    values: np.ndarray
    strengths: np.ndarray
    zero_flags: np.ndarray

    @property
    def active_rows(self) -> np.ndarray:
        return np.flatnonzero((~self.zero_flags) & (self.strengths > self.runs.zero_eps))


def project_response_patterns(
    run_dirs: list[Path], feature_indices_path: Path, read_batch_size: int = 4,
) -> ResponsePatterns:
    runs = load_runs(run_dirs)
    feature_indices = np.load(feature_indices_path, allow_pickle=False).astype(np.int64, copy=False)
    values = np.zeros((runs.row_count, feature_indices.size), dtype=np.float32)
    sample_ids = np.empty(runs.row_count, dtype=np.int64)
    strengths = np.empty(runs.row_count, dtype=np.float64)
    zero_flags = np.empty(runs.row_count, dtype=np.bool_)
    feature_to_column = np.full(runs.feature_size, -1, dtype=np.int64)
    feature_to_column[feature_indices] = np.arange(feature_indices.size)
    for row, (sample_id, ids, means, zero_flag) in enumerate(iter_response_means(runs, read_batch_size)):
        columns = feature_to_column[ids]
        keep = columns >= 0
        values[row, columns[keep]] = np.abs(means[keep])
        strengths[row] = np.linalg.norm(means[keep].astype(np.float64))
        sample_ids[row], zero_flags[row] = sample_id, zero_flag
    return ResponsePatterns(runs, sample_ids, values, strengths, zero_flags)


def exact_cosine_greedy(
    normalized: np.ndarray,
    sample_ids: np.ndarray,
    signal_strengths: np.ndarray,
    *,
    cosine_threshold: float = 0.98,
    block_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
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
    selected = np.zeros(count, dtype=np.bool_)
    order = np.lexsort((sample_ids, -signal_strengths))
    vectors = np.ascontiguousarray(normalized[order], dtype=np.float32)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        vectors = torch.from_numpy(vectors).to("cuda:0")

    selected_positions: list[int] = []
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        block = vectors[start:stop]
        if selected_positions:
            previous_vectors = vectors[np.asarray(selected_positions, dtype=np.int64)]
            previous_cosines = block @ previous_vectors.T
            if use_cuda:
                previous_cosines = previous_cosines.cpu().numpy()
            previous_max = previous_cosines.max(axis=1)
        else:
            previous_max = np.full(stop - start, -np.inf)
        within_cosines = block @ block.T
        if use_cuda:
            within_cosines = within_cosines.cpu().numpy()

        selected_locals: list[int] = []
        for local in range(stop - start):
            best_cosine = float(previous_max[local])
            if selected_locals:
                best_cosine = max(best_cosine, float(within_cosines[local, selected_locals].max()))
            if best_cosine >= cosine_threshold:
                continue
            selected[int(order[start + local])] = True
            selected_positions.append(start + local)
            selected_locals.append(local)
        print(f"Selected through {stop:,}/{count:,} active samples; kept {len(selected_positions):,}", flush=True)
    return selected


def _zero_reward_ids(runs: ExtractionRuns) -> np.ndarray:
    parts = []
    for path in runs.shards:
        for batch in pq.ParquetFile(path).iter_batches(columns=["sample_id", "reward_mean"]):
            ids = batch.column(0).to_numpy(zero_copy_only=False)
            rewards = batch.column(1).to_numpy(zero_copy_only=False)
            parts.append(ids[rewards == 0.0])
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def export_subset(
    runs: ExtractionRuns, selected_ids: np.ndarray, dataset_path: Path, output_path: Path,
) -> dict:
    selected_ids = np.unique(selected_ids)
    zero_reward_ids = _zero_reward_ids(runs)
    combined_ids = np.union1d(selected_ids, zero_reward_ids)
    dataset = pq.read_table(dataset_path.expanduser().resolve())
    dataset_ids = dataset.column("extra_info").combine_chunks().field("index").to_numpy()
    if np.unique(dataset_ids).size != dataset_ids.size:
        raise ValueError("Source dataset extra_info.index must uniquely identify each prompt")
    missing = np.setdiff1d(combined_ids, dataset_ids)
    if missing.size:
        raise ValueError(f"Selected sample ID {int(missing[0])} is absent from the source dataset")
    training = dataset.filter(pc.is_in(pa.array(dataset_ids), value_set=pa.array(combined_ids)))
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(training, output_path, compression="snappy")
    return {
        "core_set": int(selected_ids.size),
        "zero_reward": int(zero_reward_ids.size),
        "overlap": int(np.intersect1d(selected_ids, zero_reward_ids).size),
        "training": int(training.num_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and export the final training subset.")
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--feature-indices", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cosine-threshold", type=float, default=0.98)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--read-batch-size", type=int, default=4)
    parser.add_argument("--similarity-block-size", type=int, default=64)
    args = parser.parse_args()
    if not 0 <= args.cosine_threshold <= 1:
        parser.error("--cosine-threshold must be in [0, 1]")
    if args.read_batch_size < 1 or args.similarity_block_size < 1:
        parser.error("batch and block sizes must be positive")
    output = (args.output or args.run_dir[0] / "selection/training.parquet").expanduser().resolve()
    patterns = project_response_patterns(args.run_dir, args.feature_indices, args.read_batch_size)
    active = patterns.active_rows
    normalized = patterns.values[active]
    np.divide(normalized, patterns.strengths[active, None], out=normalized)
    selected = exact_cosine_greedy(
        normalized, patterns.sample_ids[active], patterns.strengths[active],
        cosine_threshold=args.cosine_threshold, block_size=args.similarity_block_size, device=args.device,
    )
    counts = export_subset(patterns.runs, patterns.sample_ids[active[selected]], args.dataset, output)
    print(json.dumps({"counts": counts, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
