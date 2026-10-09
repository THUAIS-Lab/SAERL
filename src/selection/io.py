"""Validated access to sample-level Token-TopK response-mean shards."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_COLUMNS = (
    "sample_id",
    "topk_response_feat_ids",
    "topk_response_feat_mean",
    "zero_signal_token_topk",
)
EXPECTED_TYPES = (pa.int64(), pa.list_(pa.int64()), pa.list_(pa.float32()), pa.bool_())


def load_runs(run_dirs: list[Path], input_glob: str = "step-*.parquet") -> tuple:
    """Require a shared SAE coordinate system across all input runs."""
    if not run_dirs:
        raise ValueError("At least one extraction run is required")
    run_dirs = [path.expanduser().resolve() for path in run_dirs]
    manifests, paths, metadata = [], [], []
    for directory in run_dirs:
        manifest_path = directory / "representation_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        size = int(manifest.get("sae_feature_size", 0))
        epsilon = float(manifest.get("gradient_zero_eps", 1e-12))
        layout = manifest.get("representation_layout", [])
        if size < 1 or not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError(f"Invalid feature size or zero epsilon in {manifest_path}")
        if layout not in (
            ["response_mean"],
            ["prompt_mean", "prompt_signed_peak", "response_mean", "response_signed_peak"],
        ):
            raise ValueError(f"Unsupported representation layout in {manifest_path}: {layout}")
        if int(manifest.get("topk_representation_size", -1)) != len(layout) * size:
            raise ValueError(f"Invalid Token-TopK representation size in {manifest_path}")
        manifests.append(manifest)
        shards = sorted(
            path for path in directory.glob(input_glob) if path.is_file() and not path.name.endswith(".inprogress")
        )
        if not shards:
            raise FileNotFoundError(f"No extraction shards match {input_glob!r} under {directory}")
        first_schema = None
        for path in shards:
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            for name, expected in zip(INPUT_COLUMNS, EXPECTED_TYPES, strict=True):
                index = schema.get_field_index(name)
                if index < 0 or schema.field(index).type != expected:
                    raise TypeError(f"{path}: {name} must have type {expected}")
            if first_schema is not None and not schema.equals(first_schema):
                raise ValueError(f"Parquet schema mismatch within extraction run: {path}")
            first_schema = schema
            paths.append(path)
            metadata.append({"path": str(path), "size_bytes": path.stat().st_size, "rows": parquet.metadata.num_rows})
    for key in (
        "sae_feature_size",
        "gradient_zero_eps",
        "sae_checkpoint",
        "hookpoint",
        "gradient_feature_topk",
        "decoder_normalization",
    ):
        if any(manifest.get(key) != manifests[0].get(key) for manifest in manifests[1:]):
            raise ValueError(f"Extraction runs have inconsistent {key}; shared SAE features require matching runs")
    return (
        run_dirs,
        int(manifests[0]["sae_feature_size"]),
        float(manifests[0].get("gradient_zero_eps", 1e-12)),
        paths,
        metadata,
    )


def iter_response_means(paths: list[Path], feature_size: int, batch_size: int = 1024) -> Iterator[tuple]:
    """Yield unique sample IDs, sorted feature IDs, signed means, and zero flags."""
    seen: set[int] = set()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(INPUT_COLUMNS)):
            if any(column.null_count for column in batch.columns):
                raise ValueError(f"Null representation values in {path}")
            ids_array, means_array = batch.column(1), batch.column(2)
            if ids_array.values.null_count or means_array.values.null_count:
                raise ValueError(f"Null representation list elements in {path}")
            ids_offsets = ids_array.offsets.to_numpy(zero_copy_only=False)
            mean_offsets = means_array.offsets.to_numpy(zero_copy_only=False)
            ids = ids_array.values.to_numpy(zero_copy_only=False)
            means = means_array.values.to_numpy(zero_copy_only=False)
            sample_ids = batch.column(0).to_numpy(zero_copy_only=False)
            zero_flags = batch.column(3).to_numpy(zero_copy_only=False)
            for row in range(batch.num_rows):
                sample_id = int(sample_ids[row])
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id across extraction shards: {sample_id}")
                seen.add(sample_id)
                row_ids = ids[ids_offsets[row] : ids_offsets[row + 1]]
                row_means = means[mean_offsets[row] : mean_offsets[row + 1]]
                if row_ids.size != row_means.size:
                    raise ValueError(f"sample_id={sample_id}: mismatched response feature IDs and means")
                if row_ids.size and (row_ids[0] < 0 or row_ids[-1] >= feature_size):
                    raise ValueError(f"sample_id={sample_id}: feature IDs outside [0, {feature_size})")
                if row_ids.size > 1 and np.any(row_ids[1:] <= row_ids[:-1]):
                    raise ValueError(f"sample_id={sample_id}: feature IDs must be strictly increasing")
                if not np.all(np.isfinite(row_means)):
                    raise ValueError(f"sample_id={sample_id}: response mean contains NaN or infinity")
                yield sample_id, row_ids, row_means, bool(zero_flags[row])


def load_feature_indices(path: Path, feature_size: int) -> np.ndarray:
    indices = np.load(path, allow_pickle=False)
    if indices.ndim != 1 or not indices.size or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Feature indices must be a nonempty, one-dimensional integer array")
    if np.any(indices < 0) or np.any(indices >= feature_size):
        raise ValueError(f"Feature indices must be in [0, {feature_size})")
    indices = indices.astype(np.int64, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError("Feature indices contain duplicates")
    return np.ascontiguousarray(indices)


def hash_int64(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()
