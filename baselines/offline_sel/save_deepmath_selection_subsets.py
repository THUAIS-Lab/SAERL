#!/usr/bin/env python3
"""Save the DeepMath subset listed in deepmath_irds_ids.txt."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


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
    parser.add_argument("--dataset-path", required=True, type=Path)
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    ids_path = Path(__file__).with_name("deepmath_irds_ids.txt")
    selected_ids = np.loadtxt(ids_path, dtype=np.int64, ndmin=1)
    if np.unique(selected_ids).size != selected_ids.size:
        raise ValueError(f"{ids_path} contains duplicate IDs")

    dataset = pq.read_table(dataset_path)
    dataset_ids = _dataset_indices(dataset, dataset_path)
    print(
        f"[dataset] rows={dataset.num_rows:,} columns={dataset.num_columns} "
        f"row_position_equals_index={np.array_equal(dataset_ids, np.arange(dataset.num_rows))}",
        flush=True,
    )
    selected = _take_subset(dataset, dataset_ids, selected_ids, "irds")
    output = dataset_path.parent / "irds.parquet"
    print(f"[write] {output} rows={selected.num_rows:,}", flush=True)
    _write_exact(selected, output)


if __name__ == "__main__":
    main()
