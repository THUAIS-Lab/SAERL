"""check the jsons in /data/result, make sure they are in the correct path"""
import torch
import json
import os
import glob
import re
from tqdm import tqdm

def extract_step_from_path(model_path: str):
    m = re.search(r"global_step_(\d+)", model_path)
    return int(m.group(1))

p = "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-7B/sorted_deepmath128/v4mix8/"
print(p.split('/')[-2])
jsons = glob.glob(os.path.join(p, '*.json'))
jsons = sorted(jsons)

datasets = ["aime", "amc", "gsm8k", "math", "minerva", "olympiad"]
dataset_to_total = {
    'aime': 240, 
    'amc': 320, 
    'gsm8k': 10552, 
    'math': 4000, 
    'minerva': 2176, 
    'olympiad': 5384
}

for file in tqdm(jsons, total=len(jsons), desc="Checking JSON files"):
    dataset = os.path.basename(file).split('_')[0]
    step = os.path.basename(file).split('_')[1].replace('.json', '')
    with open(file, 'r') as f:
        data = json.load(f)
    metrics = data['metrics']
    assert metrics['total'] == dataset_to_total[dataset], f"Dataset: {dataset}, Total: {metrics['total']}, Expected: {dataset_to_total[dataset]}"
    model_path = metrics['model_path']
    assert extract_step_from_path(model_path) == int(step), f"Model Path: {model_path}, Step in Path: {extract_step_from_path(model_path)}, Step in Filename: {step}"
    method = model_path.split('/')[-3]
    assert method == file.split('/')[-2]
    dataset_bs = model_path.split('/')[-4]
    assert dataset_bs == file.split('/')[-3]
    model = model_path.split('/')[-5]
    assert model == file.split('/')[-4]

print("All JSON files are valid.")
