"""multiple lines in one plot"""
import json
import re
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def extract_step_from_path(model_path: str):
    m = re.search(r"global_step_(\d+)", model_path)
    return int(m.group(1))

def extract_steps_and_accs(data):
    rows = []
    for item in data:
        acc = item.get("accuracy")
        model_path = item.get("model_path")
        step = extract_step_from_path(model_path)
        assert step != None, "failure when extract step"
        rows.append((step, acc))
    rows.sort(key=lambda x: x[0])

    steps = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    return steps, accs

def plot_metrics(json_paths, run):
    step2acc = dict()
    for json_path in json_paths:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            step, acc = extract_steps_and_accs(json_data)
            assert step == sorted(step), "not in order"
            for s, a in zip(step, acc):
                if s not in step2acc:
                    step2acc[s] = []
                step2acc[s].append(a)
    
    steps = []
    accs = []
    for s, accs_list in step2acc.items():
        if s % 40 == 0:
            steps.append(s)
            accs.append(np.mean(accs_list))

    plt.plot(steps, accs, label=run)
    for x,y in zip(steps, accs):
        plt.text(x, y, f"{y:.4f}", fontsize=8, va="bottom", ha="center")

if __name__ == "__main__":
    mean_dir = "/home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/final_results/Qwen2.5-Math-1.5B/mean128"
    datasets = ['aime', 'amc', 'math', 'minerva', 'olympiad', 'gsm8k']
    runs = ['v5mix8', 'v5r', 'v6r', 'v4mix8']

    img_path = "/home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/ablation.png"
    
    plt.figure(figsize=(30,20))
    for run in runs:
        json_paths = []
        for ds in datasets:
            if ds == 'aime':
                json_paths.append(os.path.join(mean_dir.replace("mean128", "pass128"), f"{ds}_{run}.json"))
            else:
                json_paths.append(os.path.join(mean_dir, f"{ds}_{run}.json"))
        plot_metrics(json_paths, run)
    
    plt.xlabel("global step")
    plt.ylabel("accuracy")
    plt.title(os.path.basename(img_path).replace(".png", ""))
    plt.grid(True)
    plt.legend(loc='best')
    if img_path:
        plt.savefig(img_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {img_path}")
    plt.close()