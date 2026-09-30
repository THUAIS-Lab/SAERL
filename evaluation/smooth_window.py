#!/usr/bin/env python3
"""Plot accuracy vs steps with smoothing for multiple experiments.

Example:
  python evaluation/smooth_window.py \
    --series "GRPO:evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_grpo.json" \
    --series "DAPO:evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_dapo.json" \
    --series "DIFFICULTY:evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_difficulty_baseline.json" \
    --series "GAINRL:evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_gainrl.json" \
    --series "AdaRFT:evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_adarft_grpo_100.json" \
    --series "SAERL(w/GRPO):evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_v4mix8.json" \
    --series "SAERL(w/DAPO):evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_dapo_v4mix8.json" \
    --no-markers \
    --smooth ema --ema-alpha 0.3 \
    --max-step 800 \
    --x-tick-step 100 --y-tick-step 1 \
    --figsize 10 5 \
    --out evaluation/a \
    --show-raw
"""

import argparse
import json
import os
import re
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "lines.linewidth": 1.3,
    "lines.markersize": 3,
})

COLORS = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#E76F51",
    "#0072B2",
    "#56B4E9",
    "#8C8C8C",
]
OUR_COLOR = "#E76F51"


def parse_step(value, fallback: int) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if value is None:
        return fallback
    text = str(value)
    match = re.search(r"(\d+)(?!.*\d)", text)
    if match:
        return int(match.group(1))
    return fallback


def load_series(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON format in {path}")

    points: List[Tuple[int, float]] = []
    for idx, row in enumerate(data):
        if isinstance(row, dict):
            acc = row.get("accuracy", row.get("acc", row.get("value")))
            step = row.get("step", row.get("global_step", row.get("model_path")))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            step, acc = row[0], row[1]
        else:
            continue
        if acc is None:
            continue
        step_val = parse_step(step, idx)
        points.append((step_val, float(acc)))

    points.sort(key=lambda x: x[0])
    steps = np.array([p[0] for p in points], dtype=float)
    values = np.array([p[1] for p in points], dtype=float)
    return steps, values


def smooth_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size == 0:
        return values
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def smooth_moving(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = values[start : i + 1].mean()
    return out


def parse_series(items: Iterable[str]) -> List[Tuple[str, str]]:
    series = []
    for item in items:
        if ":" not in item:
            raise ValueError("Series must be in 'Label:path' format")
        label, path = item.split(":", 1)
        series.append((label.strip(), path.strip()))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs steps with optional smoothing"
    )
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        help="Series in 'Label:path' format. Can be repeated.",
    )
    parser.add_argument(
        "--smooth",
        choices=["ema", "moving", "none"],
        default="ema",
        help="Smoothing method",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.3,
        help="EMA alpha for new points (lower = smoother)",
    )
    parser.add_argument(
        "--moving-window",
        type=int,
        default=5,
        help="Trailing window size for moving average",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Draw raw curves lightly behind smoothed curves",
    )
    parser.add_argument(
        "--no-markers",
        action="store_true",
        help="Disable markers on curves",
    )
    parser.add_argument(
        "--highlight",
        type=str,
        default="",
        help="Label to highlight with accent color",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Multiply accuracy by this scale (100 for percent, 1 for raw)",
    )
    parser.add_argument(
        "--x-label",
        type=str,
        default="Global Step",
        help="X-axis label",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=None,
        help="Only plot points with step <= max-step",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default="",
        help="Y-axis label (defaults based on scale)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional title",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[5.2, 3.2],
        help="Figure size in inches, e.g., --figsize 3.3 2.4",
    )
    parser.add_argument(
        "--x-tick-step",
        type=int,
        default=100,
        help="Major tick step on x-axis (0 to disable)",
    )
    parser.add_argument(
        "--y-tick-step",
        type=float,
        default=1.0,
        help="Major tick step on y-axis after scaling (0 to disable)",
    )
    parser.add_argument(
        "--x-limit",
        nargs=2,
        type=float,
        default=None,
        help="X-axis limits, e.g., --x-limit 0 1200",
    )
    parser.add_argument(
        "--y-limit",
        nargs=2,
        type=float,
        default=None,
        help="Y-axis limits, e.g., --y-limit 35 50",
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        default="lower right",
        help="Legend location",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures/fig_grpo_curves",
        help="Output path without extension",
    )

    args = parser.parse_args()

    series = parse_series(args.series)
    if not series:
        series = [
            ("GRPO", "plot/avg_grpo.json"),
            ("AdaRFT-GRPO-100", "plot/avg_adarft_grpo_100.json"),
        ]

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for idx, (label, path) in enumerate(series):
        steps, values = load_series(path)
        if args.max_step is not None:
            mask = steps <= args.max_step
            steps = steps[mask]
            values = values[mask]
        values = values * args.scale

        if args.smooth == "ema":
            smooth_values = smooth_ema(values, args.ema_alpha)
        elif args.smooth == "moving":
            smooth_values = smooth_moving(values, args.moving_window)
        else:
            smooth_values = values

        color = OUR_COLOR if label == args.highlight else COLORS[idx % len(COLORS)]
        marker = markers[idx % len(markers)]

        if args.show_raw:
            ax.plot(
                steps,
                values,
                color=color,
                alpha=0.25,
                linewidth=0.9,
            )

        plot_kwargs = {
            "label": label,
            "color": color,
            "linewidth": 1.3,
            "zorder": 3,
        }
        if not args.no_markers:
            plot_kwargs["marker"] = marker
            plot_kwargs["markevery"] = max(1, len(steps) // 10)

        ax.plot(steps, smooth_values, **plot_kwargs)

    ax.set_xlabel(args.x_label)
    if args.y_label:
        ax.set_ylabel(args.y_label)
    else:
        ax.set_ylabel("Accuracy (%)" if abs(args.scale - 100.0) < 1e-6 else "Accuracy")

    if args.title:
        ax.set_title(args.title)

    if args.x_tick_step and args.x_tick_step > 0:
        ax.xaxis.set_major_locator(MultipleLocator(args.x_tick_step))
    if args.y_tick_step and args.y_tick_step > 0:
        ax.yaxis.set_major_locator(MultipleLocator(args.y_tick_step))
    if args.x_limit:
        ax.set_xlim(args.x_limit)
    if args.y_limit:
        ax.set_ylim(args.y_limit)

    ax.legend(loc=args.legend_loc)
    fig.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(f"{args.out}.pdf")
    fig.savefig(f"{args.out}.png", dpi=300)


if __name__ == "__main__":
    main()
