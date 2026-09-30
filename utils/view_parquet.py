import pandas as pd
import json
from datasets import load_dataset

parquet_file_path = "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/sorted_deepmath/irds.parquet"
parquet_file = pd.read_parquet(parquet_file_path)
print(f"数据条数: {len(parquet_file)}")
print(f"shape: {parquet_file.shape}")
print(f"列名: {parquet_file.columns.tolist()}\n")
print(parquet_file)
# print('data source: ', parquet_file['data_source'][0])
# print('prompt: ', parquet_file['prompt'][0])
# # print('ability: ', parquet_file['ability'][0])
# print('reward_model: ', parquet_file['reward_model'][0])
# print('extra_info: ', parquet_file['extra_info'][0])

# p = "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/deepmath/train_simplified.parquet"
# pp = pd.read_parquet(p)
# print(f"数据条数: {len(pp)}")
# print(f"shape: {pp.shape}")
# print(f"列名: {pp.columns.tolist()}\n")
# print(pp['question'][3])
# print(pp['answer'][3])
# print(pp['index'][3])


# chat_lst = parquet_file['reward_model'].tolist()
# print(type(parquet_file['reward_model']), parquet_file['reward_model'][0:2])
# print('-------------------------')
# print(type(chat_lst), chat_lst[0:2])
# print('-------------------------')
# indices = [
#     x['index']
#     for x in parquet_file['extra_info']
# ]
# print(len(indices), type(indices))
# # print(indices)
# print('---------------------')
# for i in range(len(indices)):
#     if i != indices[i]:
#         print('err')
# print('---------------------')
# item = parquet_file.head(1)
# print(item)


# print(item['problem'][0], '\n')
# print('-----------------------------------------------------------')

# print(item['starter_code'][0], '\n')
# # # dict = json.loads(item['tests'][0])
# # # print(dict.keys())
# # # print(len(dict['inputs']))
# # # print(len(dict['outputs']))
# print('-----------------------------------------------------------')
# print(item['tests'][0], '\n')
# print('-----------------------------------------------------------')

# print(item['metadata'][0], '\n')

# 加载子集 A
# dataset = load_dataset(
#     "parquet", 
#     data_dir="/data0/dz/datasets/verifiable-coding-problems/data/",
#     cache_dir="/data0/dz/tmp/"
# )
# print(dataset)

# lcb_codegen = load_dataset(
#     "livecodebench/code_generation_lite", 
#     version_tag="release_v5", 
#     cache_dir="/data0/dz/datasets/code_generation_lite/",
#     trust_remote_code=True
# )
