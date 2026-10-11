import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from saerl.selection.representations import ExtractionRuns, load_runs


def _zero_reward_ids(runs: ExtractionRuns) -> np.ndarray:
    parts = []
    for path in runs.shards:
        for batch in pq.ParquetFile(path).iter_batches(columns=["sample_id", "reward_mean"]):
            ids = batch.column(0).to_numpy(zero_copy_only=False)
            rewards = batch.column(1).to_numpy(zero_copy_only=False)
            parts.append(ids[rewards == 0.0])
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


def export_subset(run_dirs: list[Path], selection_dir: Path, dataset_path: Path) -> dict:
    runs = load_runs(run_dirs)
    directory = selection_dir.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    selected_ids = pq.read_table(directory / "selected_sample_ids.parquet")["sample_id"].to_numpy()
    zero_reward_ids = _zero_reward_ids(runs)
    combined_ids = np.union1d(selected_ids, zero_reward_ids)
    dataset = pq.read_table(dataset_path)
    dataset_ids = dataset.column("extra_info").combine_chunks().field("index").to_numpy()
    if np.unique(dataset_ids).size != dataset_ids.size:
        raise ValueError("Source dataset extra_info.index must uniquely identify each prompt")
    missing = np.setdiff1d(combined_ids, dataset_ids)
    if missing.size:
        raise ValueError(f"Selected sample ID {int(missing[0])} is absent from the source dataset")
    core_set = dataset.filter(pc.is_in(pa.array(dataset_ids), value_set=pa.array(selected_ids)))
    training = dataset.filter(pc.is_in(pa.array(dataset_ids), value_set=pa.array(combined_ids)))
    core_path, training_path = directory / "core_set.parquet", directory / "training.parquet"
    pq.write_table(core_set, core_path, compression="snappy")
    pq.write_table(training, training_path, compression="snappy")
    manifest = {
        "dataset": str(dataset_path),
        "run_dirs": [str(path) for path in runs.directories],
        "zero_reward_definition": "reward_mean == 0.0 across all extracted samples",
        "counts": {
            "core_set": int(core_set.num_rows),
            "zero_reward": int(zero_reward_ids.size),
            "overlap": int(np.intersect1d(selected_ids, zero_reward_ids).size),
            "training": int(training.num_rows),
        },
        "outputs": {"core_set": str(core_path), "training": str(training_path)},
    }
    (directory / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the core set union all zero-reward samples as the final training dataset.")
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_subset(args.run_dir, args.selection_dir, args.dataset), indent=2))


if __name__ == "__main__":
    main()
