"""Plot three-seed learning curves and training-step efficiency."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "final_results/Qwen2.5-Math-1.5B"
# DATASETS = ["aime", "amc", "gsm8k", "math", "minerva", "olympiad"]
# DISPLAY_NAMES = ["AIME24", "AMC23", "GSM8K", "MATH500", "Minerva", "Olympiad"]
DATASETS = ["amc", "gsm8k", "math", "minerva", "olympiad"]
DISPLAY_NAMES = ["AMC23", "GSM8K", "MATH500", "Minerva", "Olympiad"]
RUNS = {
    "GRPO": ["grpo_seed0", "grpo", "grpo_seed999"],
    "SAERL": [
        "sel_v1_plus_silent_v4mix8_seed0",
        "sel_v1_plus_silent_v4mix8_run2",
        "sel_v1_plus_silent_v4mix8_seed999",
    ],
}
COLORS = {"GRPO": "#0072B2", "SAERL": "#D55E00"}
MAX_STEP = 900

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def load_accuracy(dataset, run):
    metric_dir = "pass128" if dataset == "aime" else "mean128"
    path = ROOT / metric_dir / f"{dataset}_{run}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(re.search(r"global_step_(\d+)", row["model_path"]).group(1)):
        row["accuracy"]
        for row in data
    }


def main():
    results = {
        method: [
            [load_accuracy(dataset, run) for dataset in DATASETS]
            for run in runs
        ]
        for method, runs in RUNS.items()
    }
    all_series = [series for seeds in results.values() for datasets in seeds for series in datasets]
    steps = sorted(set.intersection(*(set(series) for series in all_series)))
    steps = [step for step in steps if step <= MAX_STEP]

    fig_curve, ax_curve = plt.subplots(figsize=(2.65, 2.0))
    peak_values = {}
    for method, seeds in results.items():
        per_seed = np.array([
            [np.mean([series[step] for series in datasets]) for step in steps]
            for datasets in seeds
        ]) * 100
        mean = per_seed.mean(axis=0)
        std = per_seed.std(axis=0, ddof=1)
        color = COLORS[method]
        ax_curve.plot(steps, mean, label=method, color=color, linewidth=1.7)
        ax_curve.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)
        peak_index = np.argmax(mean)
        peak_step, peak_value = steps[peak_index], mean[peak_index]
        peak_values[method] = peak_value
        ax_curve.scatter([peak_step], [peak_value], color=color, s=12,
                         zorder=5, clip_on=False)
        ax_curve.annotate(f"{peak_value:.1f}%", (peak_step, peak_value),
                          xytext=(-5, 5 if method == "SAERL" else -12),
                          textcoords="offset points", ha="right", color=color, fontsize=8)

    ax_curve.set(xlim=(0, MAX_STEP), xlabel="Training steps",
                 ylabel="Average accuracy (%)")
    ax_curve.set_xticks([0, 300, 600, 900])
    ax_curve.grid(alpha=0.15)
    handles, labels = ax_curve.get_legend_handles_labels()
    ax_curve.legend(handles[::-1], labels[::-1], loc="lower right")
    gain = peak_values["SAERL"] - peak_values["GRPO"]
    ax_curve.text(0.03, 0.96, f"Peak gap: +{gain:.1f} pp",
                  transform=ax_curve.transAxes, ha="left", va="top",
                  color=COLORS["SAERL"], fontsize=8)

    def first_step(series, target):
        return min(step for step, acc in series.items()
                   if step <= MAX_STEP and acc >= target)

    speedups = [[] for _ in range(len(DATASETS) + 1)]
    for grpo_seed, saerl_seed in zip(results["GRPO"], results["SAERL"]):
        grpo_avg = {step: np.mean([series[step] for series in grpo_seed]) for step in steps}
        saerl_avg = {step: np.mean([series[step] for series in saerl_seed]) for step in steps}
        avg_target = max(grpo_avg.values())
        speedups[0].append(first_step(grpo_avg, avg_target)
                           / first_step(saerl_avg, avg_target))
        for index, (grpo, saerl) in enumerate(zip(grpo_seed, saerl_seed), start=1):
            target = max(acc for step, acc in grpo.items() if step <= MAX_STEP)
            speedups[index].append(first_step(grpo, target) / first_step(saerl, target))

    fig_speedup, ax_speedup = plt.subplots(figsize=(2.65, 2.0))
    means = np.array([np.mean(values) for values in speedups])
    stds = np.array([np.std(values, ddof=1) for values in speedups])
    positions = np.arange(len(speedups))
    ax_speedup.barh(positions, means, xerr=stds, height=0.55,
                    color="#E9A87D",
                    error_kw={"ecolor": "#444444", "capsize": 2, "lw": 0.8})
    ax_speedup.axvline(1, color=COLORS["GRPO"], linestyle="--", linewidth=1.0)
    ax_speedup.axhline(0.5, color="0.8", linewidth=0.6)
    ax_speedup.set_yticks(positions, ["Overall", *DISPLAY_NAMES])
    ax_speedup.get_yticklabels()[0].set_fontweight("bold")
    ax_speedup.invert_yaxis()
    ax_speedup.set_xlim(0, 2.6)
    ax_speedup.set_xticks([0, 1, 2], ["0", "GRPO 1×", "2×"])
    ax_speedup.set_xlabel("SAERL speedup over GRPO (×)")
    ax_speedup.grid(axis="x", alpha=0.15)
    ax_speedup.text(means[0] + stds[0] + 0.05, 0, f"{means[0]:.2f}×",
                    va="center", fontsize=8, fontweight="bold")

    fig_curve.tight_layout(pad=0.4)
    fig_speedup.tight_layout(pad=0.4)
    for fig, name in [(fig_curve, "intro_main_performance.pdf"),
                      (fig_speedup, "intro_main_efficiency.pdf")]:
        fig.savefig(ROOT / name, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print(f"Saved {ROOT / name}")


if __name__ == "__main__":
    main()
