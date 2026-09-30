import torch
import glob
import os
import re
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("folder")
args = parser.parse_args()

datasets = ["gsm8k", "aime", "amc", "olympiad", "minerva", "math"]
folders = glob.glob(os.path.join(args.folder, "*"))
for p in folders:
    result_dir = p.replace('ckpt', 'result')
    ckpt_paths = glob.glob(os.path.join(p, "global_step_*/"))
    result_jsons = glob.glob(os.path.join(result_dir, '*.json'))

    if len(result_jsons) // len(datasets) < len(ckpt_paths):
        print("Some checkpoints are not evaluated yet.")
        print(f"Folder: {p}")
        print(f"Number of checkpoints: {len(ckpt_paths)}")
        print(f"Number of result jsons: {len(result_jsons)}")
        print(f"Waiting for evaluation: {len(ckpt_paths) * len(datasets) - len(result_jsons)}")
print('DONE')