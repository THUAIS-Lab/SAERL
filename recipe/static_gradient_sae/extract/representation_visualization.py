"""Visualization helpers for gradient-SAE representation analysis."""

from __future__ import annotations

import base64
import html
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VISUAL_OUTPUT_FILENAMES = (
    "geometry.parquet",
    "global_pca.png",
    "dense_topk_fidelity.png",
    "feature_overview.png",
    "sample_heatmaps.pdf",
    "visualization_report.html",
)


@dataclass
class GeometryRecord:
    """Compact visualization payload retained for one sample."""

    sample_id: int
    prompt_len_tok: int
    zero_signal_dense: bool
    zero_signal_token_topk: bool
    dense_l2_norm: float
    token_topk_l2_norm: float
    cosine_similarity: float
    relative_l2_error: float
    l2_retention: float
    energy_coverage: float
    dense_sketch: np.ndarray
    token_topk_sketch: np.ndarray
    heatmap_feature_ids: np.ndarray
    heatmap_values: np.ndarray


class GeometryReservoir:
    """Uniformly retain compact CountSketch and heatmap payloads."""

    def __init__(
        self,
        *,
        feature_size: int,
        sketch_dim: int,
        max_samples: int,
        heatmap_features: int,
        seed: int,
    ) -> None:
        if sketch_dim < 4 or sketch_dim % 4:
            raise ValueError("visualization sketch_dim must be positive and divisible by four")
        if max_samples < 1 or heatmap_features < 1:
            raise ValueError("visualization max_samples and heatmap_features must be positive")
        self.feature_size = int(feature_size)
        self.sketch_dim = int(sketch_dim)
        self.max_samples = int(max_samples)
        self.heatmap_features = int(heatmap_features)
        self.seed = int(seed)
        self.records: list[GeometryRecord] = []
        self.seen = 0

        mapping_rng = np.random.default_rng(self.seed)
        per_channel = self.sketch_dim // 4
        self.buckets = mapping_rng.integers(
            0,
            per_channel,
            size=(4, self.feature_size),
            dtype=np.int32,
        )
        self.signs = (mapping_rng.integers(0, 2, size=(4, self.feature_size), dtype=np.int8) * 2 - 1).astype(np.int8)
        self.reservoir_rng = np.random.default_rng(self.seed + 1)

    def _choose_slot(self) -> int | None:
        self.seen += 1
        if len(self.records) < self.max_samples:
            return len(self.records)
        candidate = int(self.reservoir_rng.integers(0, self.seen))
        return candidate if candidate < self.max_samples else None

    def _sketch_dense(self, dense: np.ndarray) -> np.ndarray:
        per_channel = self.sketch_dim // 4
        output = np.zeros(self.sketch_dim, dtype=np.float32)
        for channel in range(4):
            output[channel * per_channel : (channel + 1) * per_channel] = np.bincount(
                self.buckets[channel],
                weights=dense[channel] * self.signs[channel],
                minlength=per_channel,
            ).astype(np.float32)
        return output

    def _sketch_sparse(
        self,
        sparse_ids: tuple[np.ndarray, ...],
        sparse_values: tuple[np.ndarray, ...],
    ) -> np.ndarray:
        per_channel = self.sketch_dim // 4
        output = np.zeros(self.sketch_dim, dtype=np.float32)
        for channel, (ids, values) in enumerate(zip(sparse_ids, sparse_values, strict=True)):
            if ids.size:
                output[channel * per_channel : (channel + 1) * per_channel] = np.bincount(
                    self.buckets[channel, ids],
                    weights=values * self.signs[channel, ids],
                    minlength=per_channel,
                ).astype(np.float32)
        return output

    def _heatmap_payload(
        self,
        dense: np.ndarray,
        sparse_ids: tuple[np.ndarray, ...],
        sparse_values: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        scores = np.max(np.abs(dense), axis=0)
        for ids, values in zip(sparse_ids, sparse_values, strict=True):
            if ids.size:
                scores[ids] = np.maximum(scores[ids], np.abs(values))
        nonzero = np.flatnonzero(scores != 0)
        if nonzero.size == 0:
            return np.empty(0, dtype=np.int64), np.empty((8, 0), dtype=np.float32)
        keep = min(self.heatmap_features, int(nonzero.size))
        candidates = nonzero
        if candidates.size > keep:
            candidates = candidates[np.argpartition(scores[candidates], -keep)[-keep:]]
        order = np.lexsort((candidates, -scores[candidates]))
        feature_ids = candidates[order].astype(np.int64, copy=False)
        values = np.zeros((8, feature_ids.size), dtype=np.float32)
        values[:4] = dense[:, feature_ids]
        for channel, (ids, channel_values) in enumerate(zip(sparse_ids, sparse_values, strict=True)):
            if ids.size == 0:
                continue
            positions = np.searchsorted(ids, feature_ids)
            valid = positions < ids.size
            valid[valid] &= ids[positions[valid]] == feature_ids[valid]
            values[4 + channel, valid] = channel_values[positions[valid]]
        return feature_ids, values

    def consider(
        self,
        *,
        sample_row: dict[str, Any],
        dense: np.ndarray,
        sparse_ids: tuple[np.ndarray, ...],
        sparse_values: tuple[np.ndarray, ...],
    ) -> None:
        """Apply reservoir sampling and retain a compact payload if selected."""
        slot = self._choose_slot()
        if slot is None:
            return
        feature_ids, heatmap_values = self._heatmap_payload(dense, sparse_ids, sparse_values)
        record = GeometryRecord(
            sample_id=int(sample_row["sample_id"]),
            prompt_len_tok=int(sample_row["prompt_len_tok"]),
            zero_signal_dense=bool(sample_row["zero_signal_dense"]),
            zero_signal_token_topk=bool(sample_row["zero_signal_token_topk"]),
            dense_l2_norm=float(sample_row["dense_all_l2_norm"]),
            token_topk_l2_norm=float(sample_row["token_topk_all_l2_norm"]),
            cosine_similarity=float(sample_row["dense_token_topk_all_cosine_similarity"]),
            relative_l2_error=float(sample_row["dense_token_topk_all_relative_l2_error"]),
            l2_retention=float(sample_row["dense_token_topk_all_l2_retention"]),
            energy_coverage=float(sample_row["dense_token_topk_all_dense_l2_energy_on_topk_support_rate"]),
            dense_sketch=self._sketch_dense(dense),
            token_topk_sketch=self._sketch_sparse(sparse_ids, sparse_values),
            heatmap_feature_ids=feature_ids,
            heatmap_values=heatmap_values,
        )
        if slot == len(self.records):
            self.records.append(record)
        else:
            self.records[slot] = record


def _matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "saerl-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.colors import Normalize
    except ImportError as error:
        raise RuntimeError("Visualization requires matplotlib; install it with `pip install matplotlib`") from error
    return plt, PdfPages, Normalize


def _fit_joint_pca(records: list[GeometryRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense = np.stack([record.dense_sketch for record in records]).astype(np.float64)
    token_topk = np.stack([record.token_topk_sketch for record in records]).astype(np.float64)
    dense_norm = np.linalg.norm(dense, axis=1)
    topk_norm = np.linalg.norm(token_topk, axis=1)
    dense_normalized = np.divide(dense, dense_norm[:, None], out=np.zeros_like(dense), where=dense_norm[:, None] > 0)
    topk_normalized = np.divide(
        token_topk,
        topk_norm[:, None],
        out=np.zeros_like(token_topk),
        where=topk_norm[:, None] > 0,
    )
    combined = np.concatenate([dense_normalized, topk_normalized], axis=0)
    centered = combined - np.mean(combined, axis=0, keepdims=True)
    if centered.shape[0] < 2 or not np.any(centered):
        coordinates = np.zeros((centered.shape[0], 2), dtype=np.float64)
        explained = np.zeros(2, dtype=np.float64)
    else:
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        components = right_vectors[: min(2, right_vectors.shape[0])]
        coordinates = centered @ components.T
        if coordinates.shape[1] < 2:
            coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
        total_variance = float(np.sum(singular_values**2))
        explained = np.zeros(2, dtype=np.float64)
        if total_variance > 0:
            explained[: min(2, singular_values.size)] = singular_values[:2] ** 2 / total_variance
    count = len(records)
    return coordinates[:count], coordinates[count:], explained


def _geometry_schema() -> pa.Schema:
    return pa.schema(
        [
            ("sample_id", pa.int64()),
            ("prompt_len_tok", pa.int64()),
            ("zero_signal_dense", pa.bool_()),
            ("zero_signal_token_topk", pa.bool_()),
            ("dense_pca_x", pa.float64()),
            ("dense_pca_y", pa.float64()),
            ("token_topk_pca_x", pa.float64()),
            ("token_topk_pca_y", pa.float64()),
            ("dense_sketch_l2_norm", pa.float64()),
            ("token_topk_sketch_l2_norm", pa.float64()),
            ("dense_l2_norm", pa.float64()),
            ("token_topk_l2_norm", pa.float64()),
            ("cosine_similarity", pa.float64()),
            ("relative_l2_error", pa.float64()),
            ("l2_retention", pa.float64()),
            ("dense_energy_coverage", pa.float64()),
        ]
    )


def _write_geometry(
    path: Path,
    records: list[GeometryRecord],
    dense_coordinates: np.ndarray,
    topk_coordinates: np.ndarray,
) -> None:
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "sample_id": record.sample_id,
                "prompt_len_tok": record.prompt_len_tok,
                "zero_signal_dense": record.zero_signal_dense,
                "zero_signal_token_topk": record.zero_signal_token_topk,
                "dense_pca_x": float(dense_coordinates[index, 0]),
                "dense_pca_y": float(dense_coordinates[index, 1]),
                "token_topk_pca_x": float(topk_coordinates[index, 0]),
                "token_topk_pca_y": float(topk_coordinates[index, 1]),
                "dense_sketch_l2_norm": float(np.linalg.norm(record.dense_sketch)),
                "token_topk_sketch_l2_norm": float(np.linalg.norm(record.token_topk_sketch)),
                "dense_l2_norm": record.dense_l2_norm,
                "token_topk_l2_norm": record.token_topk_l2_norm,
                "cosine_similarity": record.cosine_similarity,
                "relative_l2_error": record.relative_l2_error,
                "l2_retention": record.l2_retention,
                "dense_energy_coverage": record.energy_coverage,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=_geometry_schema()), path, compression="snappy")


def _limits(*coordinates: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    joined = np.concatenate(coordinates, axis=0)
    minimum = np.min(joined, axis=0)
    maximum = np.max(joined, axis=0)
    width = np.maximum(maximum - minimum, 1e-6)
    padding = width * 0.06
    return (minimum[0] - padding[0], maximum[0] + padding[0]), (
        minimum[1] - padding[1],
        maximum[1] + padding[1],
    )


def _save_png(figure, path: Path) -> None:
    figure.savefig(path, format="png", dpi=170, bbox_inches="tight", facecolor="white")


def _plot_global_pca(
    path: Path,
    records: list[GeometryRecord],
    dense_coordinates: np.ndarray,
    topk_coordinates: np.ndarray,
    explained: np.ndarray,
) -> None:
    plt, _, Normalize = _matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = np.asarray([record.cosine_similarity for record in records], dtype=np.float64)
    finite = np.isfinite(colors)
    finite_colors = colors[finite]
    if finite_colors.size:
        color_min, color_max = np.quantile(finite_colors, [0.05, 0.95])
        if color_max - color_min < 1e-6:
            color_min -= 0.05
            color_max += 0.05
        color_min = max(float(color_min), -1.0)
        color_max = min(float(color_max), 1.0)
    else:
        color_min, color_max = -1.0, 1.0
    colors = np.where(finite, colors, (color_min + color_max) / 2.0)
    zero = np.asarray([record.zero_signal_dense and record.zero_signal_token_topk for record in records], dtype=bool)
    x_limits, y_limits = _limits(dense_coordinates, topk_coordinates)
    normalization = Normalize(vmin=color_min, vmax=color_max)
    scatter = None
    for axis, coordinates, title in (
        (axes[0], dense_coordinates, "Dense"),
        (axes[1], topk_coordinates, "Token-TopK"),
    ):
        active = ~zero
        scatter = axis.scatter(
            coordinates[active, 0],
            coordinates[active, 1],
            c=colors[active],
            cmap="viridis",
            norm=normalization,
            s=13,
            alpha=0.75,
            linewidths=0,
        )
        if np.any(zero):
            axis.scatter(
                coordinates[zero, 0],
                coordinates[zero, 1],
                marker="x",
                color="0.45",
                s=20,
                linewidths=0.7,
                label="zero-signal",
            )
            axis.legend(loc="best", frameon=False, fontsize=8)
        axis.set(xlim=x_limits, ylim=y_limits, title=title, xlabel="Joint PC1", ylabel="Joint PC2")
        axis.grid(alpha=0.18)

    pair_count = min(300, len(records))
    pair_indices = np.linspace(0, len(records) - 1, pair_count, dtype=int)
    for index in pair_indices:
        axes[2].plot(
            [dense_coordinates[index, 0], topk_coordinates[index, 0]],
            [dense_coordinates[index, 1], topk_coordinates[index, 1]],
            color="0.65",
            alpha=0.2,
            linewidth=0.5,
        )
    axes[2].scatter(dense_coordinates[pair_indices, 0], dense_coordinates[pair_indices, 1], s=8, label="Dense")
    axes[2].scatter(
        topk_coordinates[pair_indices, 0],
        topk_coordinates[pair_indices, 1],
        s=8,
        label="Token-TopK",
    )
    axes[2].set(
        xlim=x_limits,
        ylim=y_limits,
        title=f"Paired displacement ({pair_count} samples)",
        xlabel="Joint PC1",
        ylabel="Joint PC2",
    )
    axes[2].grid(alpha=0.18)
    axes[2].legend(frameon=False, fontsize=8)
    if scatter is not None:
        figure.colorbar(scatter, ax=axes[:2], label="Dense–TopK cosine")
    figure.suptitle(
        "CountSketch + joint PCA of sample representations\n"
        f"explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}",
        fontsize=13,
    )
    _save_png(figure, path)
    plt.close(figure)


def _finite_xy(x: np.ndarray, y: np.ndarray, *, positive: bool = False) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    if positive:
        mask &= (x > 0) & (y > 0)
    return x[mask], y[mask]


def _scatter_or_hexbin(axis, x: np.ndarray, y: np.ndarray, *, gridsize: int = 45) -> None:
    if x.size >= 30:
        axis.hexbin(x, y, gridsize=gridsize, mincnt=1, bins="log", cmap="viridis")
    else:
        axis.scatter(x, y, s=18, alpha=0.7)


def _plot_fidelity(path: Path, sample_metrics_path: Path) -> None:
    plt, _, _ = _matplotlib()
    names = [
        "dense_all_l2_norm",
        "token_topk_all_l2_norm",
        "token_topk_all_density",
        "dense_token_topk_all_cosine_similarity",
        "dense_token_topk_all_relative_l2_error",
        "dense_token_topk_all_dense_l2_energy_on_topk_support_rate",
    ]
    table = pq.read_table(sample_metrics_path, columns=names)
    values = {
        name: table[name].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        for name in names
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

    x, y = _finite_xy(
        values["dense_token_topk_all_cosine_similarity"],
        values["dense_token_topk_all_relative_l2_error"],
    )
    _scatter_or_hexbin(axes[0, 0], x, y)
    axes[0, 0].set(xlabel="Cosine similarity", ylabel="Relative L2 error", title="Direction vs reconstruction error")

    x, y = _finite_xy(values["dense_all_l2_norm"], values["token_topk_all_l2_norm"], positive=True)
    _scatter_or_hexbin(axes[0, 1], x, y)
    if x.size:
        lower, upper = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
        axes[0, 1].plot([lower, upper], [lower, upper], "--", color="0.45", linewidth=1)
        axes[0, 1].set(xscale="log", yscale="log")
    axes[0, 1].set(xlabel="Dense L2", ylabel="Token-TopK L2", title="Norm retention")

    x, y = _finite_xy(
        values["token_topk_all_density"],
        values["dense_token_topk_all_dense_l2_energy_on_topk_support_rate"],
    )
    _scatter_or_hexbin(axes[1, 0], x, y)
    axes[1, 0].set(
        xlabel="Token-TopK nonzero density",
        ylabel="Dense L2 energy on TopK support",
        title="Sparsity vs Dense support coverage",
    )

    cosine = values["dense_token_topk_all_cosine_similarity"]
    cosine = cosine[np.isfinite(cosine)]
    axes[1, 1].hist(cosine, bins=50, color="#4472c4", alpha=0.85)
    if cosine.size:
        axes[1, 1].axvline(np.median(cosine), linestyle="--", color="#b22222", label=f"median={np.median(cosine):.3f}")
        axes[1, 1].legend(frameon=False)
    axes[1, 1].set(xlabel="Cosine similarity", ylabel="Samples", title="Dense–TopK cosine distribution")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
    figure.suptitle("Dense vs Token-TopK fidelity", fontsize=14)
    _save_png(figure, path)
    plt.close(figure)


def _plot_feature_overview(
    path: Path,
    accumulators: dict[str, Any],
    sample_count: int,
    channels: tuple[str, ...],
    top_feature_count: int,
) -> None:
    plt, _, _ = _matplotlib()
    dense = accumulators["dense"]
    topk = accumulators["token_topk"]
    aggregate_rms = {
        "Dense": np.sqrt(np.sum(dense.squared_sum, axis=0) / (sample_count * len(channels))),
        "Token-TopK": np.sqrt(np.sum(topk.squared_sum, axis=0) / (sample_count * len(channels))),
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for label, scores in aggregate_rms.items():
        ranked = np.sort(scores[scores > 0])[::-1]
        axes[0, 0].loglog(np.arange(1, ranked.size + 1), ranked, label=label)
    axes[0, 0].set(xlabel="Feature rank", ylabel="RMS", title="Global feature energy long tail")
    axes[0, 0].legend(frameon=False)

    for label, accumulator in (("Dense", dense), ("Token-TopK", topk)):
        nonzero_rate = np.sum(accumulator.nonzero_count, axis=0) / (sample_count * len(channels))
        axes[0, 1].hist(nonzero_rate, bins=50, alpha=0.55, label=label)
    axes[0, 1].set(xlabel="Mean nonzero sample rate", ylabel="Features", title="Feature usage distribution")
    axes[0, 1].legend(frameon=False)

    combined_score = np.maximum(aggregate_rms["Dense"], aggregate_rms["Token-TopK"])
    keep = min(top_feature_count, combined_score.size)
    feature_ids = np.argpartition(combined_score, -keep)[-keep:]
    feature_ids = feature_ids[np.lexsort((feature_ids, -combined_score[feature_ids]))]
    channel_rms = np.concatenate(
        [
            np.sqrt(dense.squared_sum[:, feature_ids] / sample_count),
            np.sqrt(topk.squared_sum[:, feature_ids] / sample_count),
        ],
        axis=0,
    )
    positive = channel_rms[channel_rms > 0]
    floor = float(np.min(positive)) if positive.size else 1e-30
    image = axes[1, 0].imshow(np.log10(np.maximum(channel_rms, floor)), aspect="auto", cmap="viridis")
    axes[1, 0].set(
        title="Top feature RMS by representation/channel",
        yticks=np.arange(8),
        yticklabels=[*(f"Dense/{name}" for name in channels), *(f"TopK/{name}" for name in channels)],
        xticks=np.arange(feature_ids.size),
        xticklabels=feature_ids,
    )
    axes[1, 0].tick_params(axis="x", labelrotation=90, labelsize=7)
    figure.colorbar(image, ax=axes[1, 0], label="log10 RMS")

    x, y = _finite_xy(aggregate_rms["Dense"], aggregate_rms["Token-TopK"], positive=True)
    _scatter_or_hexbin(axes[1, 1], x, y, gridsize=55)
    if x.size:
        lower, upper = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
        axes[1, 1].plot([lower, upper], [lower, upper], "--", color="0.45", linewidth=1)
        axes[1, 1].set(xscale="log", yscale="log")
    axes[1, 1].set(xlabel="Dense RMS", ylabel="Token-TopK RMS", title="Per-feature energy preservation")
    for axis in axes.flat:
        axis.grid(alpha=0.16)
    figure.suptitle("Global SAE feature overview", fontsize=14)
    _save_png(figure, path)
    plt.close(figure)


def _representative_indices(
    records: list[GeometryRecord],
    dense_coordinates: np.ndarray,
    topk_coordinates: np.ndarray,
    count: int,
) -> list[int]:
    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is not None and index not in selected and len(selected) < count:
            selected.append(int(index))

    zero_indices = [
        index for index, record in enumerate(records) if record.zero_signal_dense or record.zero_signal_token_topk
    ]
    add(zero_indices[0] if zero_indices else None)
    metric_candidates = (
        (np.asarray([record.cosine_similarity for record in records]), "min"),
        (np.asarray([record.relative_l2_error for record in records]), "max"),
        (np.asarray([record.l2_retention for record in records]), "min"),
        (np.asarray([record.dense_l2_norm for record in records]), "max"),
        (np.asarray([record.token_topk_l2_norm for record in records]), "max"),
    )
    for values, direction in metric_candidates:
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size:
            local = np.argmin(values[finite]) if direction == "min" else np.argmax(values[finite])
            add(int(finite[local]))
    for coordinates in (dense_coordinates, topk_coordinates):
        for axis in range(2):
            add(int(np.argmin(coordinates[:, axis])))
            add(int(np.argmax(coordinates[:, axis])))

    midpoint = (dense_coordinates + topk_coordinates) / 2.0
    if not selected:
        add(int(np.argmax(np.linalg.norm(midpoint, axis=1))))
    while len(selected) < min(count, len(records)):
        chosen = midpoint[selected]
        squared_distances = np.sum((midpoint[:, None, :] - chosen[None, :, :]) ** 2, axis=2)
        nearest_distance = np.min(squared_distances, axis=1)
        nearest_distance[selected] = -1.0
        add(int(np.argmax(nearest_distance)))
    return selected


def _sample_heatmap_figure(record: GeometryRecord, channels: tuple[str, ...]):
    plt, _, _ = _matplotlib()
    figure, axis = plt.subplots(figsize=(max(11, record.heatmap_feature_ids.size * 0.34), 5.6), constrained_layout=True)
    if record.heatmap_feature_ids.size == 0:
        axis.text(0.5, 0.5, "Zero representation: no nonzero feature", ha="center", va="center", fontsize=14)
        axis.set_axis_off()
    else:
        order = np.asarray([0, 4, 1, 5, 2, 6, 3, 7])
        values = record.heatmap_values[order]
        maximum = float(np.max(np.abs(values)))
        maximum = maximum if maximum > 0 else 1.0
        image = axis.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        labels = []
        for channel in channels:
            labels.extend((f"Dense/{channel}", f"TopK/{channel}"))
        axis.set(
            xlabel="SAE feature ID",
            yticks=np.arange(8),
            yticklabels=labels,
            xticks=np.arange(record.heatmap_feature_ids.size),
            xticklabels=record.heatmap_feature_ids,
        )
        axis.tick_params(axis="x", labelrotation=90, labelsize=8)
        figure.colorbar(image, ax=axis, label="Signed representation value")
    axis.set_title(
        f"sample_id={record.sample_id} · prompt={record.prompt_len_tok} tokens · "
        f"cos={record.cosine_similarity:.3g} · rel-L2={record.relative_l2_error:.3g} · "
        f"L2 retention={record.l2_retention:.3g}"
    )
    return figure


def _write_sample_heatmaps(
    path: Path,
    records: list[GeometryRecord],
    selected_indices: list[int],
    channels: tuple[str, ...],
) -> list[tuple[GeometryRecord, str]]:
    plt, PdfPages, _ = _matplotlib()
    embedded: list[tuple[GeometryRecord, str]] = []
    with PdfPages(path) as pdf:
        for index in selected_indices:
            record = records[index]
            figure = _sample_heatmap_figure(record, channels)
            pdf.savefig(figure, dpi=160, bbox_inches="tight", facecolor="white")
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
            embedded.append((record, base64.b64encode(buffer.getvalue()).decode("ascii")))
            plt.close(figure)
    return embedded


def _encoded_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _write_html(
    path: Path,
    *,
    sample_count: int,
    geometry_count: int,
    sketch_dim: int,
    explained: np.ndarray,
    global_images: list[tuple[str, Path, str]],
    sample_images: list[tuple[GeometryRecord, str]],
) -> None:
    image_sections = []
    for section_id, image_path, description in global_images:
        image_sections.append(
            f'<section id="{section_id}"><h2>{html.escape(description)}</h2>'
            f'<a href="#{section_id}"><img src="data:image/png;base64,{_encoded_file(image_path)}" '
            f'alt="{html.escape(description)}"></a></section>'
        )
    sample_sections = []
    for record, encoded in sample_images:
        title = (
            f"sample {record.sample_id} · cosine={record.cosine_similarity:.4g} · "
            f"relative L2={record.relative_l2_error:.4g}"
        )
        sample_sections.append(
            f"<details><summary>{html.escape(title)}</summary>"
            f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"></details>'
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gradient-SAE representation visualization</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 1500px; padding: 24px; color: #202124; }}
nav {{ position: sticky; top: 0; background: #fffffff2; padding: 10px 0; border-bottom: 1px solid #ddd; z-index: 2; }}
nav a {{ margin-right: 18px; }}
section {{ margin: 34px 0; }}
img {{ display: block; width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }}
details {{ margin: 14px 0; }}
summary {{ cursor: pointer; font-weight: 600; padding: 8px 0; }}
.note {{ background: #f5f7fa; border-left: 4px solid #4472c4; padding: 12px 16px; }}
</style>
</head>
<body>
<h1>Gradient-SAE representation visualization</h1>
<p>Analyzed <strong>{sample_count:,}</strong> samples. Global geometry uses a deterministic reservoir of
<strong>{geometry_count:,}</strong> samples, a shared {sketch_dim}-dimensional CountSketch, and joint PCA.
PC1/PC2 explain {explained[0]:.2%}/{explained[1]:.2%} of sketched variance.</p>
<p class="note">PCA is for visual diagnosis, not an exact distance guarantee. Per-sample heatmaps use original SAE
feature IDs and signed values without dimensionality reduction.</p>
<nav><a href="#global-pca">Global PCA</a><a href="#fidelity">Dense vs TopK</a>
<a href="#features">Features</a><a href="#samples">Samples</a></nav>
{"".join(image_sections)}
<section id="samples"><h2>Representative sample heatmaps</h2>
<p>Click a row to expand it. Red and blue encode opposite signed gradient directions.</p>
{"".join(sample_sections)}</section>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def write_visualizations(
    *,
    output_paths: dict[str, Path],
    sample_metrics_path: Path,
    accumulators: dict[str, Any],
    sample_count: int,
    channels: tuple[str, ...],
    reservoir: GeometryReservoir,
    sample_plot_count: int,
    feature_plot_count: int,
) -> dict[str, Any]:
    """Write all visual artifacts and return visualization metadata."""
    if not reservoir.records:
        raise ValueError("No samples are available for visualization")
    records = sorted(reservoir.records, key=lambda record: record.sample_id)
    dense_coordinates, topk_coordinates, explained = _fit_joint_pca(records)
    _write_geometry(output_paths["geometry.parquet"], records, dense_coordinates, topk_coordinates)
    _plot_global_pca(
        output_paths["global_pca.png"],
        records,
        dense_coordinates,
        topk_coordinates,
        explained,
    )
    _plot_fidelity(output_paths["dense_topk_fidelity.png"], sample_metrics_path)
    _plot_feature_overview(
        output_paths["feature_overview.png"],
        accumulators,
        sample_count,
        channels,
        feature_plot_count,
    )
    selected_indices = _representative_indices(
        records,
        dense_coordinates,
        topk_coordinates,
        min(sample_plot_count, len(records)),
    )
    sample_images = _write_sample_heatmaps(output_paths["sample_heatmaps.pdf"], records, selected_indices, channels)
    _write_html(
        output_paths["visualization_report.html"],
        sample_count=sample_count,
        geometry_count=len(records),
        sketch_dim=reservoir.sketch_dim,
        explained=explained,
        global_images=[
            ("global-pca", output_paths["global_pca.png"], "Global CountSketch + joint PCA"),
            ("fidelity", output_paths["dense_topk_fidelity.png"], "Dense vs Token-TopK fidelity"),
            ("features", output_paths["feature_overview.png"], "Global SAE feature overview"),
        ],
        sample_images=sample_images,
    )
    return {
        "geometry_method": "shared_countsketch_then_row_l2_then_joint_pca",
        "geometry_population": "deterministic_uniform_reservoir",
        "geometry_sample_count": len(records),
        "geometry_seen_sample_count": reservoir.seen,
        "countsketch_dimension": reservoir.sketch_dim,
        "seed": reservoir.seed,
        "pca_explained_variance_ratio": explained.tolist(),
        "sample_heatmap_ids": [records[index].sample_id for index in selected_indices],
        "sample_heatmap_feature_count": reservoir.heatmap_features,
    }
