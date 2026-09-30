import json
import re
import matplotlib.pyplot as plt
import glob
import os
import argparse

def extract_step_from_path(model_path: str):
    m = re.search(r"global_step_(\d+)", model_path)
    return int(m.group(1))

def extract_steps_and_accs(data):
    rows = []
    for i, item in enumerate(data):
        acc = item.get("accuracy")
        model_path = item.get("model_path")
        step = extract_step_from_path(model_path)
        assert step != None, "failure when extract step"
        rows.append((step, acc, model_path))
    rows.sort(key=lambda x: x[0])

    steps = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    return steps, accs

def plot_metrics(method: str, baseline: str, out_png: str = None, show: bool = False):
    with open(method, 'r', encoding='utf-8') as f1:
        method_data = json.load(f1)
    with open(baseline, 'r', encoding='utf-8') as f2:
        baseline_data = json.load(f2)
    
    steps_method, accs_method = extract_steps_and_accs(method_data)
    steps_baseline, accs_baseline = extract_steps_and_accs(baseline_data)
    print(len(steps_method), len(accs_method))
    print(len(steps_baseline), len(accs_baseline))

    plt.figure(figsize=(15,7))
    plt.plot(steps_method, accs_method, marker="o", linestyle="-", label="Ours")
    plt.plot(steps_baseline, accs_baseline, marker="s", linestyle="--", label='GRPO')

    # for x,y in zip(steps_method, accs_method):
    #     plt.text(x, y, f"{y:.3f}", fontsize=8, va="bottom", ha="center")
    # for x,y in zip(steps_baseline, accs_baseline):
    #     plt.text(x, y, f"{y:.3f}", fontsize=8, va="bottom", ha="center")
    plt.xlabel("Training Step")
    plt.ylabel("Accuracy")
    # plt.title(os.path.basename(out_png).replace(".png", "")+f"(total={method_data[0]["total"]//8}, rollout=8)")
    # plt.title(os.path.basename(out_png).replace(".png", ""))
    plt.grid(True)
    plt.legend(loc='best')
    if out_png:
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.savefig(out_png.replace('.png', '.pdf'), format='pdf')
        print(f"Saved plot to {out_png}")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    json_dir = "/home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/final_results/Qwen2.5-Math-1.5B/mean128"
    datasets = ['aime', 'amc', 'math', 'minerva', 'olympiad', 'gsm8k']
    method = 'v4mix8'
    baseline = 'grpo'
    img_name = 'v4mix8'

    os.makedirs(os.path.join(json_dir, "fig"), exist_ok=True)

    json_method = [os.path.join(json_dir, f"{dataset}_{method}.json") for dataset in datasets]
    json_baseline = [os.path.join(json_dir, f"{dataset}_{baseline}.json") for dataset in datasets]
    img_path = [os.path.join(json_dir, f"fig/{dataset}_{img_name}.png") for dataset in datasets]

    for method, baseline, out in zip(json_method, json_baseline, img_path):
        if os.path.basename(method).split('_')[0] == os.path.basename(baseline).split('_')[0]:
            print(method, baseline, out)
            plot_metrics(method, baseline, out)