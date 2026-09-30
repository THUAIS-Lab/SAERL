import pandas as pd
import json
from datasets import load_dataset
import glob
import os

parquet_file_dir = "/data/DATASET/DeepMath-103K/data/"
data_files = glob.glob(os.path.join(parquet_file_dir, '*.parquet'))
data_files = sorted(data_files)
ans_origin = []
for i, data_file in enumerate(data_files):
    print(data_file)
    parquet_file = pd.read_parquet(data_file)
    ans_origin.extend(parquet_file['final_answer'].tolist())

    # print(f"数据条数: {len(parquet_file)}")
    # print(f"shape: {parquet_file.shape}")
    # print(f"列名: {parquet_file.columns.tolist()}\n")
    # print('data source: ', parquet_file['question'][0])
    # print('prompt: ', parquet_file['final_answer'][10301])
    # print('ability: ', parquet_file['difficulty'][10301])
# print('reward_model: ', parquet_file['reward_model'][10301])
# print('extra_info: ', parquet_file['extra_info'][10301])

p = "/data/daiz/data/SAERL/sorted_deepmath/train.parquet"
pp = pd.read_parquet(p)
print(f"数据条数: {len(pp)}")
print(f"shape: {pp.shape}")
print(f"列名: {pp.columns.tolist()}\n")
# print('data_source', pp['data_source'][0])
# print('prompt: ', pp['prompt'][0])
# print('ability: ', pp['ability'][0])
# print('reward_model: ', pp['reward_model'][0])
# print('extra_info: ', pp['extra_info'][0])


ans2= pp['reward_model'].tolist()
ans2 = [x['ground_truth'] for x in ans2]
# print(pp['prompt'][0])
# print(pp['prompt'][10303])
# print(pp['prompt'][20606])
# print(pp['prompt'][30908])
# print(pp['prompt'][41210])
# print(pp['prompt'][51512])
# print(pp['prompt'][61814])
# print(pp['prompt'][72116])
# print(pp['prompt'][82418])
# print(pp['prompt'][92720])

if ans_origin == ans2:
    print("The order of the answers is the same.")
    print(ans_origin[:5])
    print(ans2[:5])
else:
    print("The order of the answers is different.")
# print(pp['prompt'][10302*9+10301])
# print(pp['reward_model'][10302*9+10301])