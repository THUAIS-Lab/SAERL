import pandas as pd
import json
from datasets import load_dataset
import glob
import os
import torch
import datasets
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np

path_dir = "/home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/raw_grad_sel/dataset"
paths = glob.glob(os.path.join(path_dir, "*.parquet"))
for p1 in paths:
    parquet_file = pd.read_parquet(p1)
    print(f"数据条数: {len(parquet_file)}")
    print(f"shape: {parquet_file.shape}")
    print(f"列名: {parquet_file.columns.tolist()}\n")

    extra_info1=parquet_file['extra_info'].tolist()

    selected_ids = [i['index'] for i in extra_info1]
    print(len(selected_ids), min(selected_ids), max(selected_ids))

    ids_path = os.path.join(p1.replace('.parquet', '_ids.npy'))
    np.save(ids_path, np.array(selected_ids))

# paths = glob.glob(os.path.join(path_dir, "*.npy"))
# for path in paths:
#     data = np.load(path).tolist()
#     print(path, len(data), min(data), max(data))
