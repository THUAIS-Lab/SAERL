"""Cluster-first batching, stage interleaving, and local cross-cluster mixing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


N_STAGES = 20
WINDOW_STAGES = 3
STAGE_DECAY = 0.7
UNLOCK_FRACTION = 0.9


@dataclass
class _Batch:
    rows: pd.DataFrame
    cluster: int
    stage: int
    progress: float

    @property
    def score(self) -> float:
        return float(self.rows.ranking_score.mean())


def _cluster_batches(predictions: pd.DataFrame, batch_size: int) -> list[_Batch]:
    batches = []
    minimum = (batch_size + 1) // 2
    for cluster, group in predictions.groupby("cluster_id", sort=True):
        group = group.sort_values(["ranking_score", "sample_id"], kind="mergesort").copy()
        group["stage_progress"] = group.ranking_score.rank(pct=True)
        pieces = [group.iloc[start : start + batch_size].copy() for start in range(0, len(group), batch_size)]
        if len(pieces) > 1 and len(pieces[-1]) < minimum:
            # Move nearby samples to a tiny tail while keeping both groups at least half full.
            needed = minimum - len(pieces[-1])
            previous = pieces[-2]
            pieces[-2] = previous.iloc[:-needed].copy()
            pieces[-1] = pd.concat([previous.iloc[-needed:], pieces[-1]], ignore_index=True)
        for rows in pieces:
            progress = float(rows.stage_progress.median())
            stage = min(N_STAGES - 1, int(progress * N_STAGES))
            rows["source_batch_id"] = len(batches)
            rows["source_stage"] = stage
            rows["is_padding_copy"] = False
            if len(rows) < batch_size:
                # Complete this cluster's batch without joining it to another cluster.
                needed = batch_size - len(rows)
                donors = rows.iloc[np.arange(needed) % len(rows)].copy()
                donors["is_padding_copy"] = True
                rows = pd.concat([rows, donors], ignore_index=True)
            batches.append(_Batch(rows, int(cluster), stage, progress))
    return batches


def _interleave(batches: list[_Batch], seed: int) -> list[_Batch]:
    queues = {}
    for batch in batches:
        queues.setdefault(batch.cluster, deque()).append(batch)
    rotations = {}
    for stage in range(N_STAGES):
        scores = {}
        for batch in batches:
            if batch.stage == stage:
                scores.setdefault(batch.cluster, []).append(batch.score)
        rotations[stage] = deque(sorted(scores, key=lambda cluster: (np.median(scores[cluster]), cluster)))
    totals = np.bincount([batch.stage for batch in batches], minlength=N_STAGES)
    emitted = np.zeros(N_STAGES, dtype=np.int64)
    rng = np.random.default_rng(seed)
    ordered = []
    frontier = 0
    while len(ordered) < len(batches):
        while frontier < N_STAGES - 1 and (
            totals[frontier] == 0 or emitted[frontier] >= UNLOCK_FRACTION * totals[frontier]
        ):
            frontier += 1
        # Only each cluster's next batch is eligible, preserving its easy-to-hard order.
        heads = {cluster: queue[0] for cluster, queue in queues.items() if queue}
        stages = sorted({batch.stage for batch in heads.values()})
        candidates = [stage for stage in stages if frontier - WINDOW_STAGES < stage <= frontier]
        if not candidates:
            candidates = [stage for stage in stages if stage <= frontier]
        if not candidates:
            frontier = min(stages)
            candidates = [frontier]
        weights = np.array([STAGE_DECAY ** (frontier - stage) for stage in candidates])
        stage = int(rng.choice(candidates, p=weights / weights.sum()))
        rotation = rotations[stage]
        while True:
            cluster = rotation[0]
            rotation.rotate(-1)
            if cluster in heads and heads[cluster].stage == stage:
                break
        ordered.append(queues[cluster].popleft())
        emitted[stage] += 1
    return ordered


def _mix(batches: list[_Batch], mix_tail: int, window_batches: int) -> list[int]:
    scores = np.array([batch.score for batch in batches])
    lengths = np.array([np.log1p(batch.rows.full_len_tok).mean() for batch in batches])
    score_scale = max(float(scores.std()), 1e-8)
    length_scale = max(float(lengths.std()), 1e-8)
    available = {i for i, batch in enumerate(batches) if not batch.rows.is_padding_copy.any()}
    partners = [-1] * len(batches)
    if not mix_tail:
        return partners
    for left, batch in enumerate(batches):
        if left not in available:
            continue
        for stage_gap in (0, 1):
            candidates = [
                right for right in range(max(0, left - window_batches), min(len(batches), left + window_batches + 1))
                if right != left and right in available
                and batches[right].cluster != batch.cluster
                and abs(batch.stage - batches[right].stage) <= stage_gap
            ]
            if not candidates:
                continue
            right = min(candidates, key=lambda candidate: (
                1.6 * abs(scores[left] - scores[candidate]) / score_scale
                + 0.6 * abs(lengths[left] - lengths[candidate]) / length_scale
                + 0.4 * abs(batch.stage - batches[candidate].stage)
                + 0.15 * abs(left - candidate) / max(window_batches, 1),
                candidate,
            ))
            left_rows, right_rows = batch.rows, batches[right].rows
            batch.rows = pd.concat([left_rows.iloc[:-mix_tail], right_rows.iloc[-mix_tail:]], ignore_index=True)
            batches[right].rows = pd.concat([
                right_rows.iloc[:-mix_tail], left_rows.iloc[-mix_tail:],
            ], ignore_index=True)
            partners[left], partners[right] = right, left
            available.difference_update((left, right))
            break
    return partners


def build_order(
    predictions: pd.DataFrame,
    *,
    batch_size: int = 128,
    mix_tail: int = 8,
    window_batches: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Build fixed-size single-cluster batches, interleave stages, then swap tails."""
    if batch_size < 1 or not 0 <= mix_tail <= batch_size or window_batches < 0:
        raise ValueError("Require batch_size > 0, 0 <= mix_tail <= batch_size, and window_batches >= 0")
    if predictions.empty:
        raise ValueError("Cannot build a curriculum from an empty prediction table")
    if predictions.sample_id.isna().any() or predictions.sample_id.duplicated().any():
        raise ValueError("Prediction sample IDs must be non-null and unique")
    if predictions.cluster_id.isna().any():
        raise ValueError("Every sample must have a cluster ID")
    if not np.isfinite(predictions[["ranking_score", "full_len_tok"]].to_numpy()).all():
        raise ValueError("Ranking scores and sequence lengths must be finite")
    if (predictions.full_len_tok < 0).any():
        raise ValueError("Sequence lengths must be nonnegative")
    batches = _interleave(_cluster_batches(predictions, batch_size), seed)
    partners = _mix(batches, mix_tail, window_batches)
    for batch_id, batch in enumerate(batches):
        batch.rows["batch_id"] = batch_id
        batch.rows["base_cluster_id"] = batch.cluster
        batch.rows["stage"] = batch.stage
        batch.rows["stage_progress"] = batch.progress
        batch.rows["partner_batch_id"] = partners[batch_id]
    result = pd.concat([batch.rows for batch in batches], ignore_index=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=np.int64))
    return result
