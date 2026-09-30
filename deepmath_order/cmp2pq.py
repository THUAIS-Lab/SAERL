import pandas as pd
import json
from datasets import load_dataset
import glob
import os
import torch
import datasets
from torch.utils.data import Dataset, Subset, DataLoader

p1="/home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813/test/a/selected_plus_silent_deepmath.parquet"
parquet_file = pd.read_parquet(p1)
print(f"数据条数: {len(parquet_file)}")
print(f"shape: {parquet_file.shape}")
print(f"列名: {parquet_file.columns.tolist()}\n")
prompt1=parquet_file['prompt'].tolist()
data_source1=parquet_file['data_source'].tolist()
ability1=parquet_file['ability'].tolist()
reward_model1=parquet_file['reward_model'].tolist()
extra_info1=parquet_file['extra_info'].tolist()


p = "/home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813/selection_v1_2_3_variance_feature_indices_thres_07/selected_plus_silent_deepmath.parquet"
pp = pd.read_parquet(p)
print(f"数据条数: {len(pp)}")
print(f"shape: {pp.shape}")
print(f"列名: {pp.columns.tolist()}\n")
prompt2=pp['prompt'].tolist()
data_source2=pp['data_source'].tolist()
ability2=pp['ability'].tolist()
reward_model2=pp['reward_model'].tolist()
extra_info2=pp['extra_info'].tolist()


if prompt1!=prompt2:
    print("prompt不同")
if data_source1!=data_source2:
    print("data_source不同")
if ability1!=ability2:
    print("ability不同")
if reward_model1!=reward_model2:
    print("reward_model不同")
if extra_info1!=extra_info2:
    print("extra_info不同")
    for e1, e2 in zip(extra_info1, extra_info2):
        if e1 != e2:
            print("extra_info不同")
            print(f"e1: {e1}")
            print(f"e2: {e2}")
            break
print("DONE")