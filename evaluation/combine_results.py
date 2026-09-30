
"""generate results from eval_*.json"""
import json
import glob
import os

def print_existing_files(p):
    jsons = glob.glob(os.path.join(p, "*.json"))
    print(len([p for p in jsons if 'math' in p]))
    print(len([p for p in jsons if 'amc' in p]))
    print(len([p for p in jsons if 'aime' in p]))
    print(len([p for p in jsons if 'minerva' in p]))
    print(len([p for p in jsons if 'olympiad' in p]))

result_dir = "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/deepmath128/dapo128/"
type = "dapo"
out_dir = "/home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/results/Qwen2.5-Math-1.5B/deepmath128"

print_existing_files(result_dir)
result_jsons = glob.glob(os.path.join(result_dir, '*.json'))

datasets = ["aime", "amc", "gsm8k", "math", "minerva", "olympiad"]
dataset_to_total = {
    'aime': 240, 
    'amc': 320, 
    'gsm8k': 10552, 
    'math': 4000, 
    'minerva': 2176, 
    'olympiad': 5400
}
if "Qwen2.5-Math-1.5B" in result_dir: 
    dataset_to_total['olympiad'] = 5384
results = dict()
for dataset in datasets:
    results[dataset] = []

for p in result_jsons:
    dataset = os.path.basename(p).split('_')[0]
    assert dataset in datasets, "false dataset name, please check!"
    with open(p, 'r') as f:
        data = json.load(f)
    metrics = data['metrics']
    assert metrics['total'] == dataset_to_total[dataset], f"{metrics['total']}, {dataset_to_total[dataset]}"
    assert type in metrics['model_path']
    results[dataset].append(metrics)


for dataset in datasets:
    if results[dataset] != []:
        fname = os.path.join(out_dir, f"{dataset}_{type}.json")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        results[dataset].sort(key=lambda x: x['model_path'])

        with open(fname, "w") as fp:
            json.dump(results[dataset], fp, indent=2)
        print(f"Results saved to {fname}")