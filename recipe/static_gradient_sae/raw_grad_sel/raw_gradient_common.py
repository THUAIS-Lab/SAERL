"""Shared loading utilities for raw-gradient selection baselines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from recipe.static_gradient_sae.selection_v1.select_topk_response_mean_v1_2 import LoadedRepresentations

RAW_GRADIENT_COLUMNS = ("sample_id", "raw_grad_vals")
RAW_GRADIENT_TYPES = {
    "sample_id": pa.int64(),
    "raw_grad_vals": pa.list_(pa.float32()),
}
RAW_GRADIENT_LAYOUT = (
    "prompt_mean",
    "prompt_signed_peak",
    "response_mean",
    "response_signed_peak",
)


@dataclass(frozen=True)
class RawGradientSource:
    """Validated raw-gradient input metadata."""

    run_dir: Path
    input_glob: str
    paths: list[Path]
    shard_metadata: list[dict[str, Any]]
    manifest: dict[str, Any] | None
    raw_gradient_size: int
    response_mean_size: int
    zero_eps: float

    @property
    def response_mean_slice(self) -> slice:
        return slice(2 * self.response_mean_size, 3 * self.response_mean_size)


def _validate_required_schema(schema: pa.Schema, path: Path) -> None:
    for name, expected in RAW_GRADIENT_TYPES.items():
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

    metadata: list[dict[str, Any]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        _validate_required_schema(parquet.schema_arrow, path)
        metadata.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "rows": int(parquet.metadata.num_rows),
                "row_groups": int(parquet.metadata.num_row_groups),
            }
        )
    if not any(item["rows"] for item in metadata):
        raise ValueError("Raw-gradient shards contain no rows")
    return paths, metadata


def _ragged_float32(array: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    if array.null_count:
        raise ValueError("Null raw-gradient lists are not supported")
    if not (pa.types.is_list(array.type) or pa.types.is_large_list(array.type)):
        raise TypeError(f"Expected a list array for raw_grad_vals, got {array.type}")
    if array.values.null_count:
        raise ValueError("Null elements in raw-gradient lists are not supported")
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = array.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return offsets, values


def _sample_ids(array: pa.Array) -> np.ndarray:
    if array.null_count:
        raise ValueError("Null sample_id values are not supported")
    return array.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _infer_representation_size(paths: list[Path]) -> int:
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1, columns=["raw_grad_vals"]):
            offsets, _values = _ragged_float32(batch.column(0))
            size = int(offsets[1] - offsets[0])
            if size < 1:
                raise ValueError(f"First raw-gradient row in {path} is empty")
            return size
    raise ValueError("Raw-gradient shards contain no rows")


def load_source(
    run_dir: Path,
    *,
    input_glob: str,
    zero_eps: float | None = None,
) -> RawGradientSource:
    """Discover shards and resolve raw-gradient dimensionality and zero epsilon."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    paths, shard_metadata = _discover_shards(run_dir, input_glob)

    manifest_path = run_dir / "representation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    inferred_size = _infer_representation_size(paths)
    declared_size = int(manifest.get("raw_gradient_representation_size", 0)) if manifest else 0
    if declared_size and declared_size != inferred_size:
        raise ValueError(
            f"raw_gradient_representation_size in {manifest_path} is {declared_size}, "
            f"but Parquet rows have length {inferred_size}"
        )
    raw_gradient_size = declared_size or inferred_size
    if raw_gradient_size % len(RAW_GRADIENT_LAYOUT) != 0:
        raise ValueError(
            f"raw_grad_vals length must be divisible by four for {RAW_GRADIENT_LAYOUT}, got {raw_gradient_size}"
        )
    if manifest and "raw_gradient_representation_layout" in manifest:
        declared_layout = tuple(manifest["raw_gradient_representation_layout"])
        if declared_layout != RAW_GRADIENT_LAYOUT:
            raise ValueError(
                f"raw_gradient_representation_layout in {manifest_path} must be "
                f"{RAW_GRADIENT_LAYOUT}, got {declared_layout}"
            )
    response_mean_size = raw_gradient_size // len(RAW_GRADIENT_LAYOUT)

    resolved_zero_eps = (
        float(zero_eps) if zero_eps is not None else float((manifest or {}).get("gradient_zero_eps", 1e-12))
    )
    if not math.isfinite(resolved_zero_eps) or resolved_zero_eps < 0.0:
        raise ValueError(f"zero_eps must be finite and nonnegative, got {resolved_zero_eps}")
    return RawGradientSource(
        run_dir=run_dir,
        input_glob=input_glob,
        paths=paths,
        shard_metadata=shard_metadata,
        manifest=manifest,
        raw_gradient_size=raw_gradient_size,
        response_mean_size=response_mean_size,
        zero_eps=resolved_zero_eps,
    )


def load_dimension_indices(path: Path, representation_size: int) -> np.ndarray:
    """Load a unique one-dimensional integer coordinate array."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    indices = np.load(path, allow_pickle=False)
    if indices.ndim != 1:
        raise ValueError(f"Dimension-index array must be one-dimensional, got shape={indices.shape}")
    if indices.size == 0:
        raise ValueError("Dimension-index array must not be empty")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"Dimension-index array must have an integer dtype, got {indices.dtype}")
    indices = indices.astype(np.int64, copy=False)
    invalid = indices[(indices < 0) | (indices >= representation_size)]
    if invalid.size:
        raise ValueError(
            f"Dimension index {int(invalid[0])} falls outside raw response-mean range [0, {representation_size})"
        )
    if np.unique(indices).size != indices.size:
        raise ValueError("Dimension-index array contains duplicate indices")
    return np.ascontiguousarray(indices)


def _should_log_progress(current: int, total: int) -> bool:
    interval = max(1, total // 20)
    return current == 1 or current == total or current % interval == 0


def load_absolute_representations(
    source: RawGradientSource,
    *,
    dimension_indices: np.ndarray | None,
    read_batch_size: int,
) -> tuple[LoadedRepresentations, dict[str, int]]:
    """Load absolute, L2-normalized raw response means in full or selected coordinates."""
    if read_batch_size < 1:
        raise ValueError("read_batch_size must be positive")
    total_rows = sum(int(item["rows"]) for item in source.shard_metadata)
    if dimension_indices is None:
        selected_indices = None
        output_size = source.response_mean_size
    else:
        selected_indices = np.asarray(dimension_indices, dtype=np.int64)
        output_size = int(selected_indices.size)

    normalized_storage = np.empty((total_rows, output_size), dtype=np.float32)
    sample_ids = np.empty(total_rows, dtype=np.int64)
    signal_strengths = np.empty(total_rows, dtype=np.float64)
    source_zero_flags = np.empty(total_rows, dtype=np.bool_)
    active_all_rows = np.empty(total_rows, dtype=np.int64)
    active_sample_ids = np.empty(total_rows, dtype=np.int64)
    active_signal_strengths = np.empty(total_rows, dtype=np.float64)
    seen_sample_ids: set[int] = set()
    loaded_count = 0
    active_count = 0
    source_zero_count = 0
    projected_zero_count = 0

    for shard_number, path in enumerate(source.paths, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=read_batch_size,
            columns=list(RAW_GRADIENT_COLUMNS),
            use_threads=True,
        ):
            batch_sample_ids = _sample_ids(batch.column(0))
            offsets, flat_values = _ragged_float32(batch.column(1))
            for row in range(batch.num_rows):
                sample_id = int(batch_sample_ids[row])
                if sample_id in seen_sample_ids:
                    raise ValueError(f"Duplicate sample_id across extraction shards: {sample_id}")
                seen_sample_ids.add(sample_id)
                values = flat_values[int(offsets[row]) : int(offsets[row + 1])]
                if values.size != source.raw_gradient_size:
                    raise ValueError(
                        f"sample_id={sample_id} has raw_grad_vals length {values.size}, "
                        f"expected {source.raw_gradient_size}"
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"sample_id={sample_id} raw_grad_vals contains NaN or infinity")

                response_mean = values[source.response_mean_slice]
                response_mean64 = response_mean.astype(np.float64, copy=False)
                source_norm = math.sqrt(max(float(np.dot(response_mean64, response_mean64)), 0.0))
                if selected_indices is None:
                    projected = response_mean
                    projected_norm = source_norm
                else:
                    projected = response_mean[selected_indices]
                    projected64 = projected.astype(np.float64, copy=False)
                    projected_norm = math.sqrt(max(float(np.dot(projected64, projected64)), 0.0))
                source_zero = source_norm <= source.zero_eps

                sample_ids[loaded_count] = sample_id
                signal_strengths[loaded_count] = projected_norm
                source_zero_flags[loaded_count] = source_zero
                if source_zero:
                    source_zero_count += 1
                elif projected_norm <= source.zero_eps:
                    projected_zero_count += 1
                else:
                    normalized_storage[active_count] = np.abs(projected) / projected_norm
                    active_all_rows[active_count] = loaded_count
                    active_sample_ids[active_count] = sample_id
                    active_signal_strengths[active_count] = projected_norm
                    active_count += 1
                loaded_count += 1

        if _should_log_progress(shard_number, len(source.paths)):
            print(
                f"[load] shards={shard_number:,}/{len(source.paths):,} "
                f"rows={loaded_count:,}/{total_rows:,} active={active_count:,}",
                flush=True,
            )

    if loaded_count != total_rows:
        raise ValueError(f"Row-count mismatch: expected={total_rows}, loaded={loaded_count}")
    normalized_active = np.ascontiguousarray(normalized_storage[:active_count])
    loaded = LoadedRepresentations(
        feature_size=output_size,
        zero_eps=source.zero_eps,
        sample_ids=sample_ids,
        signal_strengths=signal_strengths,
        zero_signal_flags=source_zero_flags,
        active_all_rows=active_all_rows[:active_count],
        active_sample_ids=active_sample_ids[:active_count],
        active_signal_strengths=active_signal_strengths[:active_count],
        normalized_active=normalized_active,
        shard_metadata=source.shard_metadata,
    )
    stats = {
        "source_response_mean_zero_samples": source_zero_count,
        "projected_zero_samples": projected_zero_count,
        "active_samples": active_count,
    }
    return loaded, stats
