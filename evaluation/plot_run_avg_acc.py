"""Plot each run's average accuracy across selected datasets."""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "GRPO": "#999999",
    "SAERL-S": "#45B8A7",
    "SAERL-C": "#F0A453",
    "SAERL": "#D94A43",
}
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = ["-", "-", "-", "-"]

plt.rcParams.update(
    {
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


def load_acc(path: Path, step_mod: int, max_step: int):
    records = sorted(
        json.loads(path.read_text(encoding="utf-8")),
        key=lambda row: row.get("timestamp", ""),
    )
    result = {}
    for row in records:
        step = extract_step_from_path(row["model_path"])
        if step <= max_step and step % step_mod == 0:
            result[step] = float(row["accuracy"])
    return result


def smooth_values(values, window):
    """Centered moving average; window=1 keeps the original values."""
    left = (window - 1) // 2
    right = window // 2
    return [
        sum(values[max(0, i - left) : i + right + 1])
        / len(values[max(0, i - left) : i + right + 1])
        for i in range(len(values))
    ]


def plot_run_average(
    root,
    datasets,
    dataset_display_names,
    runs,
    names,
    img_name,
    x_label,
    y_label,
    plot_title,
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
    if len(dataset_display_names) != len(datasets):
        raise ValueError(
            "dataset_display_names and datasets must have the same length"
        )
    if len(names) != len(runs):
        raise ValueError("names and runs must have the same length")

    root = Path(root)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    csv_rows = []

    for i, (run, name) in enumerate(zip(runs, names)):
        paths = [
            root / ("pass128" if dataset == "aime" else "mean128")
            / f"{dataset}_{run}.json"
            for dataset in datasets
        ]
        missing_paths = [path for path in paths if not path.exists()]
        if missing_paths:
            print(f"Skip {run}: missing {', '.join(str(path) for path in missing_paths)}")
            continue

        acc_by_dataset = [load_acc(path, step_mod, max_step) for path in paths]
        steps = sorted(
            set.intersection(*(set(acc_by_step) for acc_by_step in acc_by_dataset))
        )
        if not steps:
            print(f"Skip {run}: no shared steps")
            continue

        averages = [
            sum(acc_by_step[step] for acc_by_step in acc_by_dataset) / len(datasets)
            for step in steps
        ]
        averages = smooth_values(averages, smooth_window)
        csv_rows.extend(zip([run] * len(steps), steps, averages))
        ax.plot(
            steps,
            averages,
            label=name,
            color=COLORS.get(name),
            marker=MARKERS[i % len(MARKERS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            markersize=3,
        )

    if not csv_rows:
        raise ValueError("No complete results found for the selected datasets and runs")

    if plot_title:
        ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, max_step)
    ax.set_xticks([max_step * i / 3 for i in range(4)])
    ax.legend(loc="best")
    fig.tight_layout()

    output_base = root / img_name
    fig.savefig(output_base.with_suffix(".pdf"))
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    plt.close(fig)

    csv_path = output_base.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["run", "step", "avg_acc"])
        writer.writerows(csv_rows)
    print(
        f"Saved {output_base.with_suffix('.pdf')}, "
        f"{output_base.with_suffix('.png')}, and {csv_path}"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "final_results/Qwen2.5-Math-1.5B"
    datasets = ["math", "minerva", "gsm8k", 'aime', 'amc', 'olympiad']
    dataset_display_names = ["MATH500", "MinervaMath", "GSM8K", "AIME24", "AMC23", "OlympiadBench"]
    runs = ["grpo", "sel_v1_plus_silent2", "v4mix8", "sel_v1_plus_silent_v4mix8_run2", 'sel_hidden_diff_sae_batch_v4mix8', 'full_hidden_diff_sae_batch_v4mix8']
    names = ["GRPO", "SAERL-S", "SAERL-C", "SAERL", 'final-saerl', 'final-saerl-c']
    img_name = "intro40"
    x_label = "Training Steps"
    y_label = "Average Accuracy"
    plot_title = ""  # Leave empty to omit the title.
    step_mod = 20
    max_step = 900
    smooth_window = 1

    plot_run_average(
        root,
        datasets,
        dataset_display_names,
        runs,
        names,
        img_name,
        x_label,
        y_label,
        plot_title,
        step_mod,
        max_step,
        smooth_window,
    )
