"""Normalized Energy Top-K on one shared Token-TopK response-mean SAE basis."""

from pathlib import Path

import numpy as np

from saerl.selection.io import iter_response_means, load_runs


def select(run_dirs: list[Path], top_k: int) -> tuple[np.ndarray, np.ndarray, int]:
    _, feature_size, zero_eps, paths, _ = load_runs(run_dirs)
    if not 1 <= top_k <= feature_size:
        raise ValueError(f"top_k must be in [1, {feature_size}]")
    energy = np.zeros(feature_size, dtype=np.float64)
    sample_count = 0
    for _, ids, means, zero_flag in iter_response_means(paths, feature_size):
        values = means.astype(np.float64)
        norm_squared = float(np.dot(values, values))
        if zero_flag or norm_squared <= zero_eps**2:
            continue
        energy[ids] += values**2 / norm_squared
        sample_count += 1
    if not sample_count:
        raise ValueError("No nonzero Token-TopK response-mean samples found")
    # Stable ties choose smaller feature IDs; all samples share this basis.
    indices = np.sort(np.argsort(-energy, kind="stable")[:top_k].astype(np.int64))
    return indices, energy, sample_count
