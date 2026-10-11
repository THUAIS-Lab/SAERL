from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

RESPONSE_COLUMNS = (
    "sample_id",
    "topk_response_feat_ids",
    "topk_response_feat_mean",
    "zero_signal_token_topk",
)


def gradient_sae_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("prompt_len_tok", pa.int64()),
            ("reward_mean", pa.float32()),
            ("topk_response_feat_ids", pa.list_(pa.int64())),
            ("topk_response_feat_mean", pa.list_(pa.float32())),
            ("zero_signal_token_topk", pa.bool_()),
        ]
    )


@dataclass(frozen=True)
class ExtractionRuns:

    directories: list[Path]
    shards: list[Path]
    feature_size: int
    zero_eps: float
    row_count: int


def load_runs(run_dirs: list[Path]) -> ExtractionRuns:
    directories = [path.expanduser().resolve() for path in run_dirs]
    manifests = [json.loads((path / "representation_manifest.json").read_text()) for path in directories]
    basis_keys = (
        "sae_feature_size", "sae_checkpoint", "hookpoint", "gradient_feature_topk",
        "decoder_normalization", "gradient_zero_eps",
    )
    if any(
        manifest.get(key) != manifests[0].get(key)
        for manifest in manifests[1:]
        for key in basis_keys
    ):
        raise ValueError("Extraction runs must share the same SAE and projection settings")
    shards = [path for directory in directories for path in sorted(directory.glob("step-*.parquet"))]
    return ExtractionRuns(
        directories=directories,
        shards=shards,
        feature_size=int(manifests[0]["sae_feature_size"]),
        zero_eps=float(manifests[0].get("gradient_zero_eps", 1e-12)),
        row_count=sum(pq.ParquetFile(path).metadata.num_rows for path in shards),
    )


def iter_response_means(runs: ExtractionRuns, batch_size: int = 1024) -> Iterator[tuple]:
    for path in runs.shards:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(RESPONSE_COLUMNS)):
            ids_array, means_array = batch.column(1), batch.column(2)
            ids_offsets = ids_array.offsets.to_numpy(zero_copy_only=False)
            means_offsets = means_array.offsets.to_numpy(zero_copy_only=False)
            ids = ids_array.values.to_numpy(zero_copy_only=False)
            means = means_array.values.to_numpy(zero_copy_only=False)
            sample_ids = batch.column(0).to_numpy(zero_copy_only=False)
            zero_flags = batch.column(3).to_numpy(zero_copy_only=False)
            for row in range(batch.num_rows):
                yield (
                    int(sample_ids[row]),
                    ids[ids_offsets[row]:ids_offsets[row + 1]],
                    means[means_offsets[row]:means_offsets[row + 1]],
                    bool(zero_flags[row]),
                )
