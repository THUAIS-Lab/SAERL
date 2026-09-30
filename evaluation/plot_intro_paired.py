"""Plot average accuracy and paired time-to-target steps for the intro figure."""

import matplotlib.pyplot as plt
import numpy as np

from plot_intro_main import COLORS, DATASETS, MAX_STEP, ROOT, RUNS, load_accuracy


def main():
    results = {
        method: [[load_accuracy(dataset, run) for dataset in DATASETS] for run in runs]
        for method, runs in RUNS.items()
    }
    all_series = [series for runs in results.values() for datasets in runs for series in datasets]
    steps = sorted(set.intersection(*(set(series) for series in all_series)))
    steps = [step for step in steps if step <= MAX_STEP]
    curves = {
        method: np.array([
            [np.mean([series[step] for series in datasets]) for step in steps]
            for datasets in runs
        ])
        for method, runs in results.items()
    }

    fig, (ax_curve, ax_steps) = plt.subplots(
        1, 2, figsize=(5.5, 2.4), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    peaks = {}
    for method, per_run in curves.items():
        mean = per_run.mean(axis=0) * 100
        std = per_run.std(axis=0, ddof=1) * 100
        color = COLORS[method]
        ax_curve.plot(steps, mean, label=method, color=color, linewidth=1.7)
        ax_curve.fill_between(steps, mean - std, mean + std, color=color, alpha=0.16)
        peak_index = np.argmax(mean)
        peak_step, peak_value = steps[peak_index], mean[peak_index]
        peaks[method] = peak_value
        ax_curve.scatter(peak_step, peak_value, color=color, s=22, zorder=5)
        ax_curve.annotate(f"{peak_value:.1f}%", (peak_step, peak_value),
                          xytext=(5, 5 if method == "SAERL" else -12),
                          textcoords="offset points", color=color, fontsize=8)

    ax_curve.set(xlim=(0, MAX_STEP), xlabel="Training steps",
                 ylabel="Six-benchmark average (%)")
    ax_curve.set_xticks([0, 300, 600, 900])
    ax_curve.grid(alpha=0.15)
    ax_curve.legend(loc="lower right")
    ax_curve.text(0.03, 0.96, f"Peak gap: +{peaks['SAERL'] - peaks['GRPO']:.1f} pp",
                  transform=ax_curve.transAxes, ha="left", va="top",
                  color=COLORS["SAERL"], fontsize=8)

    speedups = []
    for index, (grpo, saerl) in enumerate(zip(curves["GRPO"], curves["SAERL"])):
        target = grpo.max()
        grpo_step = steps[np.flatnonzero(grpo >= target)[0]]
        saerl_step = steps[np.flatnonzero(saerl >= target)[0]]
        speedups.append(grpo_step / saerl_step)
        y = len(RUNS["GRPO"]) - index - 1
        ax_steps.plot([saerl_step, grpo_step], [y, y], color="0.72", linewidth=1.8,
                      zorder=1)
        for method, step, offset in [("SAERL", saerl_step, -14),
                                     ("GRPO", grpo_step, 5)]:
            ax_steps.scatter(step, y, color=COLORS[method], s=28, zorder=3)
            ax_steps.annotate(str(step), (step, y), xytext=(0, offset),
                              textcoords="offset points", ha="center", fontsize=7,
                              color=COLORS[method])

    ax_steps.set(xlim=(0, 920), ylim=(-0.5, 3.05), xlabel="Steps to GRPO peak")
    ax_steps.set_xticks([0, 300, 600, 900])
    ax_steps.set_yticks([2, 1, 0], ["Run 1", "Run 2", "Run 3"])
    ax_steps.tick_params(axis="y", length=0)
    ax_steps.spines["left"].set_visible(False)
    ax_steps.grid(axis="x", alpha=0.15)
    ax_steps.text(0.03, 0.97, f"Mean speedup: {np.mean(speedups):.2f}×",
                  transform=ax_steps.transAxes, ha="left", va="top",
                  color=COLORS["SAERL"], fontsize=8)

    fig.tight_layout(pad=0.4, w_pad=0.7)
    output = ROOT / "intro_paired"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {output.with_suffix('.pdf')} and {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
