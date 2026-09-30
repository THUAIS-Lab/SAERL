import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
import os

#####################################config##########################################
MODEL_PATH = "/data/MODEL/Qwen2.5-Math-1.5B/"
DATA_PATH = "/data/daiz/data/SAERL/sorted_deepmath/train.parquet"
out_path = "/data/daiz/data/SAERL/sorted_deepmath/train_filtered.parquet"
max_prompt_length = 1024
PROMPT_COLUMN = "prompt"
IS_CHAT_FORMAT = True
#####################################config##########################################
origin = pd.read_parquet(DATA_PATH)
filtered = pd.read_parquet(out_path)

print(len(origin), len(filtered))

for i in range(95332):
    assert filtered['prompt'][i] == origin['prompt'][i], f"问题文本不匹配！"
    assert filtered['reward_model'][i] == origin['reward_model'][i], f"答案文本不匹配"
    assert filtered['extra_info'][i] == origin['extra_info'][i], f"extra_info不匹配！"
    assert filtered['ability'][i] == origin['ability'][i], f"ability不匹配！"
    assert filtered['data_source'][i] == origin['data_source'][i], f"data_source不匹配！"

print(filtered['prompt'][12], origin['prompt'][12])
print(filtered['reward_model'][12], origin['reward_model'][12])
print(filtered['extra_info'][12], origin['extra_info'][12])
print(filtered['ability'][12], origin['ability'][12])
print(filtered['data_source'][12], origin['data_source'][12])
print('yes')
# [95332, 95847, 96136, 96706, 96811, 97464, 100402, 102635]
# [95332, 95847, 96136, 96706, 96811, 97464, 100402, 102635]