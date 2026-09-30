import pandas as pd
import json
from datasets import load_dataset
import glob
import os
import torch
import datasets
from torch.utils.data import Dataset, Subset, DataLoader

# p1="/home/daiz/SAERL/deepmath_order/a.parquet"
# parquet_file = pd.read_parquet(p1)
# prompt1=parquet_file['prompt'].tolist()
# data_source1=parquet_file['data_source'].tolist()
# ability1=parquet_file['ability'].tolist()
# reward_model1=parquet_file['reward_model'].tolist()
# extra_info1=parquet_file['extra_info'].tolist()


# p = "/home/daiz/SAERL/deepmath_order/correct.parquet"
# pp = pd.read_parquet(p)
# print(f"数据条数: {len(pp)}")
# print(f"shape: {pp.shape}")
# print(f"列名: {pp.columns.tolist()}\n")
# prompt2=pp['prompt'].tolist()
# data_source2=pp['data_source'].tolist()
# ability2=pp['ability'].tolist()
# reward_model2=pp['reward_model'].tolist()
# extra_info2=pp['extra_info'].tolist()

# # p3 = "/home/daiz/SAERL/wrong_order.parquet"
# # pp3 = pd.read_parquet(p3)
# # prompt3=pp3['prompt'].tolist()
# # data_source3=pp3['data_source'].tolist()
# # ability3=pp3['ability'].tolist()
# # reward_model3=pp3['reward_model'].tolist()
# # extra_info3=pp3['extra_info'].tolist()

# if prompt1==prompt2:
#     print("prompt相同")
# if data_source1==data_source2:
#     print("data_source相同")
# if ability1==ability2:
#     print("ability相同")
# if reward_model1==reward_model2:
#     print("reward_model相同")
# if extra_info1==extra_info2:
#     print("extra_info相同")
# else:
#     for e1, e2 in zip(extra_info1, extra_info2):
#         if e1 != e2:
#             print("extra_info不同")
#             print(f"e1: {e1}")
#             print(f"e2: {e2}")
#             break

# ['/data/DATASET/DeepMath-103K/data/train-00003-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00001-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00008-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00004-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00000-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00002-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00007-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00006-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00005-of-00010.parquet', 
#  '/data/DATASET/DeepMath-103K/data/train-00009-of-00010.parquet']
# parquet_file_dir = "/data/DATASET/DeepMath-103K/data/"
# data_files = glob.glob(os.path.join(parquet_file_dir, '*.parquet'))
# data_files = sorted(data_files)
# ans_origin = []
# for i, data_file in enumerate(data_files):
#     print(data_file)
#     parquet_file = pd.read_parquet(data_file)
#     print(f"数据条数: {len(parquet_file)}")

"""change order"""
# p="/home/daiz/SAERL/rank/Qwen2.5-Math-1.5B/deepmath_difficulty_clustered_v4.pt"
# data=torch.load(p)
# # data_files={"train": "/home/daiz/SAERL/deepmath_order/wrong.parquet"}
# # dataset = datasets.load_dataset("parquet", data_files=data_files)["train"]
# a=[0, 10303, 20606, 30908, 41210, 51512, 61814, 72116, 82418, 92720]
# b=[3,1,8,4,0,2,7,6,5,9]
# pq_length=[10303,10303,10302,10302,10302,10302,10302,10302,10302,10302]
# correct2wrong = []
# for i in b:
#     correct2wrong.extend(list(range(a[i], a[i]+pq_length[i])))

# wrong2correct = [0] * len(correct2wrong)
# for k, idx in enumerate(correct2wrong):
#     wrong2correct[idx] = k

# correct_data_rank = torch.zeros_like(data)
# for i in range(len(data)):
#     wrong_no = data[i]
#     correct_data_rank[i] = wrong2correct[wrong_no]
# torch.save(correct_data_rank, "/home/daiz/SAERL/deepmath_order/correct_v4.pt")
# # # it doesn't change the index in extra_info, but the order of data is correctly changed
# # mapped_dataset = dataset.select(wrong2correct)
# # mapped_dataset.to_parquet("/home/daiz/SAERL/deepmath_order/a.parquet")

p1="/home/daiz/SAERL/deepmath_order/wrong_v4.pt"
data=torch.load(p1)
p2="/home/daiz/SAERL/rank/Qwen2.5-Math-1.5B/deepmath_difficulty_clustered_v4.pt"
data2=torch.load(p2)
if torch.all(data == data2):
    print("数据顺序相同")