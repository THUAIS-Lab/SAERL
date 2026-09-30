"""Plot accuracy curves for multiple datasets in one figure."""

import json
import math
import os
import re

import matplotlib.pyplot as plt


COLORS = ["#009E73", "#56B4E9", "#E69F00", "#D55E00", "#0072B2"]
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "-", "-", "-"]

plt.rcParams.update(
    {
        # Times-compatible font used by ICLR/ICML/NeurIPS paper templates.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "lines.linewidth": 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def extract_step_from_path(model_path: str) -> int:
    match = re.search(r"global_step_(\d+)", model_path)
    if match is None:
        raise ValueError(f"Cannot extract global step from {model_path!r}")
    return int(match.group(1))


def extract_steps_and_accs(data, step_mod: int, max_step: int):
    rows = sorted(
        (
            extract_step_from_path(item["model_path"]),
            item["accuracy"],
        )
        for item in data
    )
    rows = [
        (step, acc)
        for step, acc in rows
        if step <= max_step and step % step_mod == 0
    ]
    return [step for step, _ in rows], [acc for _, acc in rows]


def smooth_values(values, window):
    """Centered moving average; window=1 keeps the original values."""
    left = (window - 1) // 2
    right = window // 2
    return [
        sum(values[max(0, i - left) : i + right + 1])
        / len(values[max(0, i - left) : i + right + 1])
        for i in range(len(values))
    ]


def plot_metrics(
    json_dir,
    datasets,
    dataset_display_names,
    runs,
    names,
    img_path,
    step_mod,
    max_step,
    smooth_window=1,
):
    if step_mod <= 0:
        raise ValueError("step_mod must be a positive integer")
    if max_step <= 0:
        raise ValueError("max_step must be a positive integer")
    if smooth_window <= 0:
        raise ValueError("smooth_window must be a positive integer")
    if len(names) != len(runs):
        raise ValueError("names and runs must have the same length")
    if len(dataset_display_names) != len(datasets):
        raise ValueError(
            "dataset_display_names and datasets must have the same length"
        )

    ncols = 2 if len(datasets) == 4 else min(3, len(datasets))
    nrows = math.ceil(len(datasets) / ncols)
    fig_height = 1.85 * nrows + 0.1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5, fig_height),
        sharex=True,
        squeeze=False,
    )

    for ax, dataset, dataset_display_name in zip(
        axes.flat, datasets, dataset_display_names
    ):
        for i, (run, name) in enumerate(zip(runs, names)):
            json_path = os.path.join(json_dir, f"{dataset}_{run}.json")
            with open(json_path, "r", encoding="utf-8") as file:
                steps, accs = extract_steps_and_accs(
                    json.load(file), step_mod, max_step
                )
            accs = smooth_values(accs, smooth_window)

            ax.plot(
                steps,
                accs,
                label=name,
                color=COLORS[i % len(COLORS)],
                marker=MARKERS[i % len(MARKERS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                markersize=2,
            )

        ax.set_title(dataset_display_name)
        ax.set_xlim(0, max_step)
        ax.set_xticks([max_step * i / 3 for i in range(4)])
        ax.tick_params(axis="x", labelbottom=True)

    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)

    fig.supxlabel("Training Steps", y=0.01)
    fig.supylabel("Accuracy", x=0.01)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs),
               bbox_to_anchor=(0.5, 0.99), fontsize=7.5,
               handlelength=1.4, handletextpad=0.35,
               columnspacing=0.8, borderaxespad=0)
    fig.subplots_adjust(left=0.11, right=0.99,
                        bottom=0.42 / fig_height, top=1 - 0.44 / fig_height,
                        wspace=0.35, hspace=0.30)
    # fig.subplots_adjust(
    #     left=0.11, right=0.99,
    #     bottom=0.30 / fig_height,
    #     top=1 - 0.44 / fig_height,
    #     wspace=0.35, hspace=0.30
    # )

    output_base, _ = os.path.splitext(img_path)
    fig.savefig(f"{output_base}.pdf")
    fig.savefig(f"{output_base}.png", dpi=300)
    plt.close(fig)
    print(f"Saved plots to {output_base}.pdf and {output_base}.png")


if __name__ == "__main__":
    json_dir = "/home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/final_results/Qwen2.5-Math-1.5B/mean128"
    datasets = ["math", "minerva", "gsm8k"]
    dataset_display_names = ["MATH-500", "Minerva Math", "GSM8K"]
    runs = ["grpo", "raw_grad_sel_v1_plus_silent", "sel_v1_plus_silent2"]
    names = ["GRPO", "Gradient Selection", "Ours"]
    img_name = "grad"
    step_mod = 60  # Set to 20, 40, or another positive integer.
    max_step = 900
    smooth_window = 1  # Set to 1 to disable smoothing.

    fig_dir = os.path.join(json_dir, "fig")
    os.makedirs(fig_dir, exist_ok=True)
    img_path = os.path.join(fig_dir, f"all_datasets_{img_name}.png")
    plot_metrics(
        json_dir,
        datasets,
        dataset_display_names,
        runs,
        names,
        img_path,
        step_mod,
        max_step,
        smooth_window,
    )
