#!/usr/bin/env python3
"""Save DeepMath subsets using selection IDs and rewards from one or more runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _load_ids(path: Path) -> np.ndarray:
    table = pq.read_table(path, columns=["sample_id"])
    column = table.column("sample_id").combine_chunks()
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
    if not pa.types.is_struct(extra_info.type) or extra_info.type.get_field_index("index") < 0:
        raise TypeError(f"{path} extra_info must contain an index field")
    index = extra_info.field("index")
    if index.null_count or not pa.types.is_integer(index.type):
        raise TypeError(f"{path} extra_info.index must be a non-null integer")
    values = index.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    if np.unique(values).size != values.size:
        raise ValueError(f"{path} contains duplicate extra_info.index values")
    return values


def _load_zero_reward_silent_ids(run_dirs: list[Path], silent_ids: np.ndarray) -> np.ndarray:
    if silent_ids.size == 0:
        return silent_ids
    paths = []
    for run_dir in run_dirs:
        run_paths = sorted(
            path
            for path in run_dir.glob("step-*.parquet")
            if path.is_file() and not path.name.endswith(".inprogress")
        )
        if not run_paths:
            raise FileNotFoundError(f"No extraction shards matching 'step-*.parquet' under {run_dir}")
        paths.extend(run_paths)
    print(f"[discover] runs={len(run_dirs):,} shards={len(paths):,}", flush=True)

    silent_value_set = pa.array(silent_ids, type=pa.int64())
    matched_id_parts: list[np.ndarray] = []
    matched_reward_parts: list[np.ndarray] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        if schema.get_field_index("sample_id") < 0 or schema.field("sample_id").type != pa.int64():
            raise TypeError(f"{path} sample_id must be int64")
        if schema.get_field_index("reward_mean") < 0 or not pa.types.is_floating(schema.field("reward_mean").type):
            raise TypeError(f"{path} reward_mean must be floating point")
        for batch in parquet.iter_batches(columns=["sample_id", "reward_mean"], use_threads=True):
            sample_ids = batch.column(0)
            rewards = batch.column(1)
            if sample_ids.null_count or rewards.null_count:
                raise ValueError(f"{path} sample_id and reward_mean must be non-null")
            mask = pc.is_in(sample_ids, value_set=silent_value_set)
            matched_id_parts.append(sample_ids.filter(mask).to_numpy(zero_copy_only=False))
            matched_reward_parts.append(rewards.filter(mask).to_numpy(zero_copy_only=False))

    matched_ids = np.concatenate(matched_id_parts).astype(np.int64, copy=False)
    matched_rewards = np.concatenate(matched_reward_parts)
    if np.unique(matched_ids).size != matched_ids.size:
        raise ValueError("Duplicate silent sample IDs found across extraction shards")
    missing = np.setdiff1d(silent_ids, matched_ids, assume_unique=True)
    if missing.size:
        raise ValueError(
            f"{missing.size} silent sample IDs have no reward; first missing ID={int(missing[0])}"
        )
    if not np.all(np.isfinite(matched_rewards)):
        raise ValueError("Silent sample reward_mean values must be finite")
    return np.sort(matched_ids[matched_rewards == 0.0])


def _take_subset(table: pa.Table, dataset_ids: np.ndarray, requested_ids: np.ndarray, label: str) -> pa.Table:
    missing = np.setdiff1d(requested_ids, dataset_ids, assume_unique=True)
    if missing.size:
        raise ValueError(f"{label}: {missing.size} sample IDs are absent; first missing ID={int(missing[0])}")
    mask = pc.is_in(pa.array(dataset_ids), value_set=pa.array(requested_ids))
    subset = table.filter(mask)
    if subset.num_rows != requested_ids.size:
        raise AssertionError(f"{label}: expected {requested_ids.size} rows, got {subset.num_rows}")
    return subset


def _write_exact(table: pa.Table, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    temporary = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, temporary, compression="snappy", use_dictionary=True)
        roundtrip = pq.read_table(temporary)
        if not roundtrip.equals(table, check_metadata=True):
            raise RuntimeError(f"Parquet round-trip changed table content or schema metadata: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--selection-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dirs = [path.expanduser().resolve() for path in args.run_dir]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise NotADirectoryError(run_dir)
    directory = args.selection_dir.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    dataset_path = directory / "sorted_deepmath.parquet"
    selected_path = directory / "selected_sample_ids.parquet"
    silent_path = directory / "silent_sample_ids.parquet"
    for path in (dataset_path, selected_path, silent_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected_ids = _load_ids(selected_path)
    silent_ids = _load_ids(silent_path)
    overlap = np.intersect1d(selected_ids, silent_ids, assume_unique=True)
    if overlap.size:
        raise ValueError(f"selected and silent sets overlap; first shared ID={int(overlap[0])}")
    zero_reward_silent_ids = _load_zero_reward_silent_ids(run_dirs, silent_ids)
    combined_ids = np.union1d(selected_ids, zero_reward_silent_ids)
    print(
        f"[ids] selected={selected_ids.size:,} silent={silent_ids.size:,} "
        f"zero-reward-silent={zero_reward_silent_ids.size:,} "
        f"selected+zero-reward-silent={combined_ids.size:,}",
        flush=True,
    )

    dataset = pq.read_table(dataset_path)
    dataset_ids = _dataset_indices(dataset, dataset_path)
    print(
        f"[dataset] rows={dataset.num_rows:,} columns={dataset.num_columns} "
        f"row_position_equals_index={np.array_equal(dataset_ids, np.arange(dataset.num_rows))}",
        flush=True,
    )
    selected = _take_subset(dataset, dataset_ids, selected_ids, "selected")
    selected_plus_silent = _take_subset(
        dataset,
        dataset_ids,
        combined_ids,
        "selected+zero-reward-silent",
    )

    selected_output = directory / "selected_deepmath.parquet"
    combined_output = directory / "selected_plus_silent_deepmath.parquet"
    print(f"[write] {selected_output} rows={selected.num_rows:,}", flush=True)
    _write_exact(selected, selected_output)
    print(f"[write] {combined_output} rows={selected_plus_silent.num_rows:,}", flush=True)
    _write_exact(selected_plus_silent, combined_output)
    print("[done] subsets preserve source rows, column order, schema, and schema metadata", flush=True)


if __name__ == "__main__":
    main()
