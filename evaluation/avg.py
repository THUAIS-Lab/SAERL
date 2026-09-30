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

if __name__ == "__main__":
    mean_dir = "/home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/final_results/Qwen2.5-Math-1.5B/mean128"
    pass_dir = mean_dir.replace("mean128", "pass128")
    avg_dir = mean_dir.replace("mean128", "avg128")
    os.makedirs(avg_dir, exist_ok=True)
    datasets = ['aime', 'amc', 'math', 'minerva', 'olympiad', 'gsm8k']
    runs = ['grpo', 'difficulty_baseline', 'v4mix8']

    for run in runs:
        json_paths = []
        step2acc = dict()
        avg_result = []
        for ds in datasets:
            if ds == 'aime':
                json_paths.append(os.path.join(pass_dir, f"{ds}_{run}.json"))
            else:
                json_paths.append(os.path.join(mean_dir, f"{ds}_{run}.json"))
        for p in json_paths:
            print(p)
            with open(p, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                steps, accs = extract_steps_and_accs(json_data)
                for step, acc in zip(steps, accs):
                    if step not in step2acc:
                        step2acc[step] = []
                    step2acc[step].append(acc)
        for step, acc_list in step2acc.items():
            if len(acc_list) == 6:
                avg_result.append(
                    {
                        'accuracy': np.mean(acc_list),
                        'model_path': f"global_step_{step}"
                    }
                )
        with open(os.path.join(avg_dir, f"avg_{run}.json"), 'w', encoding='utf-8') as f:
            json.dump(avg_result, f, indent=2)
                
        print('###################################')
  