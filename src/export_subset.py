"""Export the gradient core set and its union with all zero-reward samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from saerl.selection.io import hash_int64, load_runs


def _load_ids(path: Path) -> np.ndarray:
    column = pq.read_table(path, columns=["sample_id"]).column("sample_id").combine_chunks()
    if column.null_count or column.type != pa.int64():
        raise TypeError(f"{path} sample_id must be non-null int64")
    values = column.to_numpy(zero_copy_only=False)
    if np.unique(values).size != values.size:
        raise ValueError(f"{path} contains duplicate sample_id values")
    return values


def _dataset_indices(table: pa.Table, path: Path) -> np.ndarray:
    if "extra_info" not in table.column_names:
        raise ValueError(f"{path} has no extra_info column")
    extra_info = table.column("extra_info").combine_chunks()
    if extra_info.null_count or not pa.types.is_struct(extra_info.type) or extra_info.type.get_field_index("index") < 0:
        raise TypeError(f"{path} extra_info must be non-null and contain an index field")
    index = extra_info.field("index")
    if index.null_count or not pa.types.is_integer(index.type):
        raise TypeError(f"{path} extra_info.index must be a non-null integer")
    if len(index) and pa.types.is_unsigned_integer(index.type) and pc.max(index).as_py() > np.iinfo(np.int64).max:
        raise ValueError(f"{path} extra_info.index exceeds the int64 range")
    values = index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if np.unique(values).size != values.size:
        raise ValueError(f"{path} contains duplicate extra_info.index values")
    return values


def _load_rewards(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Read every extracted sample, including silent and redundant samples."""
    id_parts, reward_parts = [], []
    for path in paths:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        index = schema.get_field_index("reward_mean")
        if index < 0 or not pa.types.is_floating(schema.field(index).type):
            raise TypeError(f"{path} reward_mean must be floating point")
        for batch in parquet.iter_batches(columns=["sample_id", "reward_mean"]):
            if any(column.null_count for column in batch.columns):
                raise ValueError(f"{path} sample_id and reward_mean must be non-null")
            id_parts.append(batch.column(0).to_numpy(zero_copy_only=False))
            reward_parts.append(batch.column(1).to_numpy(zero_copy_only=False))
    ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.int64)
    rewards = np.concatenate(reward_parts) if reward_parts else np.empty(0, dtype=np.float32)
    if np.unique(ids).size != ids.size:
        raise ValueError("Duplicate sample IDs found across extraction shards")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("Sample reward_mean values must be finite")
    return ids, np.sort(ids[rewards == 0.0])


def _take_subset(table: pa.Table, dataset_ids: np.ndarray, requested_ids: np.ndarray, label: str) -> pa.Table:
    missing = np.setdiff1d(requested_ids, dataset_ids, assume_unique=True)
    if missing.size:
        raise ValueError(f"{label}: {missing.size} sample IDs are absent; first missing ID={int(missing[0])}")
    subset = table.filter(pc.is_in(pa.array(dataset_ids), value_set=pa.array(requested_ids)))
    if subset.num_rows != requested_ids.size:
        raise AssertionError(f"{label}: expected {requested_ids.size} rows, got {subset.num_rows}")
    return subset


def _write_exact(table: pa.Table, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        if not pq.read_table(temporary).equals(table, check_metadata=True):
            raise RuntimeError(f"Parquet round-trip changed content or schema metadata: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--selection-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    run_dirs, _, _, paths, _ = load_runs(args.run_dir)
    directory = args.selection_dir.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    selected_ids = _load_ids(directory / "selected_sample_ids.parquet")
    audit_ids = _load_ids(directory / "selection_audit.parquet")
    extracted_ids, zero_reward_ids = _load_rewards(paths)
    if not np.array_equal(np.sort(audit_ids), np.sort(extracted_ids)):
        raise ValueError("Extraction run IDs do not match the selection audit; use the same runs for all stages")
    if np.setdiff1d(selected_ids, audit_ids).size:
        raise ValueError("Selected sample IDs are absent from the selection audit")
    combined_ids = np.union1d(selected_ids, zero_reward_ids)
    dataset = pq.read_table(dataset_path)
    dataset_ids = _dataset_indices(dataset, dataset_path)
    missing = np.setdiff1d(extracted_ids, dataset_ids)
    if missing.size:
        raise ValueError(f"Extracted sample ID {int(missing[0])} is absent from the source dataset")
    core_set = _take_subset(dataset, dataset_ids, selected_ids, "core set")
    training = _take_subset(dataset, dataset_ids, combined_ids, "core set + zero-reward samples")
    core_path, training_path = directory / "core_set.parquet", directory / "training.parquet"
    manifest_path = directory / "export_manifest.json"
    for path in (core_path, training_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    _write_exact(core_set, core_path)
    _write_exact(training, training_path)
    manifest = {
        "dataset": str(dataset_path),
        "run_dirs": [str(path) for path in run_dirs],
        "zero_reward_definition": "reward_mean == 0.0 across all extracted samples",
        "counts": {
            "core_set": int(selected_ids.size),
            "zero_reward": int(zero_reward_ids.size),
            "overlap": int(np.intersect1d(selected_ids, zero_reward_ids).size),
            "training": int(combined_ids.size),
        },
        "training_sample_ids_sha256": hash_int64(combined_ids),
        "outputs": {"core_set": str(core_path), "training": str(training_path)},
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
