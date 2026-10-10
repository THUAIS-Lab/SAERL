"""Build absolute response-mean patterns on the globally selected SAE features."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from saerl.selection.representations import ExtractionRuns, iter_response_means, load_runs


@dataclass(frozen=True)
class ResponsePatterns:
    runs: ExtractionRuns
    feature_indices: np.ndarray
    sample_ids: np.ndarray
    values: np.ndarray
    strengths: np.ndarray
    zero_flags: np.ndarray

    @property
    def active_rows(self) -> np.ndarray:
        return np.flatnonzero((~self.zero_flags) & (self.strengths > self.runs.zero_eps))


def project_response_patterns(
    run_dirs: list[Path], feature_indices_path: Path, read_batch_size: int = 1024,
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
        # Signed token/rollout means have already been computed during extraction.
        values[row, columns[keep]] = np.abs(means[keep])
        strengths[row] = np.linalg.norm(means[keep].astype(np.float64))
        sample_ids[row], zero_flags[row] = sample_id, zero_flag
    return ResponsePatterns(runs, feature_indices, sample_ids, values, strengths, zero_flags)
