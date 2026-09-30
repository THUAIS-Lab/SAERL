# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the zwhe99/DeepMath-103K dataset to parquet format
"""

import argparse
import torch
import os
import datasets
import glob
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
import os
from verl.utils.hdfs_io import copy, makedirs

#####################################config##########################################
MODEL_PATH = "/data/MODEL/Qwen2.5-Math-1.5B/"
DATA_PATH = "/data/daiz/data/SAERL/sorted_deepmath/train.parquet"
out_path = "/data/daiz/data/SAERL/sorted_deepmath/train_filtered.parquet"
max_prompt_length = 1024
PROMPT_COLUMN = "prompt"
IS_CHAT_FORMAT = True
#####################################config##########################################

def generate_rank(train_dataset):
    # from pprint import pprint
    # pprint(1)
    # breakpoint()
    results_list = []
    for i, item in enumerate(train_dataset):
        assert i == item['extra_info']['index']
        results_list.append(
            {
                "index": i,
                "difficulty": item['extra_info']['difficulty']
            }
        )
        
    # difficulty from low to high
    sorted_results = sorted(results_list, key=lambda x: x['difficulty'], reverse=False)

    final_indices = [item['index'] for item in sorted_results]
    final_difficulties = [item['difficulty'] for item in sorted_results]

    save_data = {
        "indices": final_indices,              # List[int]
        "difficulty": final_difficulties                 # List[float]
    }

    OUTPUT_PATH = "/data/daiz/data/deepmath.pt"
    torch.save(save_data, OUTPUT_PATH)
    print(f"Saved {len(final_indices)} records to {OUTPUT_PATH}")
    print("Top 5 indices (Easiest):", final_indices[:5], final_difficulties[:5])
    print("Bottom 5 indices (Hardest):", final_indices[-5:], final_difficulties[-5:])

def check_dataset_length():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Loading dataset from {DATA_PATH}...")
    # 根据文件类型自动加载
    if DATA_PATH.endswith('.parquet'):
        ds = load_dataset('parquet', data_files=DATA_PATH, split='train')
    elif DATA_PATH.endswith('.json') or DATA_PATH.endswith('.jsonl'):
        ds = load_dataset('json', data_files=DATA_PATH, split='train')
    else:
        # 尝试作为 HuggingFace 数据集加载
        try:
            ds = load_from_disk(DATA_PATH)
        except:
            ds = load_dataset(DATA_PATH, split='train')

    print(f"Total samples: {len(ds)}")
    
    lengths = []
    over_limit_indices = []

    print("Calculating token lengths...")
    for idx, item in enumerate(tqdm(ds)):
        raw_content = item[PROMPT_COLUMN]
        
        # 核心逻辑：获取实际输入给模型的文本
        if IS_CHAT_FORMAT and isinstance(raw_content, list):
            # 自动应用模版，不生成 generation prompt (只算 input)
            text_to_tokenize = tokenizer.apply_chat_template(
                raw_content, 
                tokenize=False, 
                add_generation_prompt=True 
            )
        else:
            # 如果已经是字符串
            text_to_tokenize = raw_content

        # 计算 token 数量
        token_ids = tokenizer.encode(text_to_tokenize, add_special_tokens=False)
        length = len(token_ids)
        
        lengths.append(length)
        if length > max_prompt_length:
            over_limit_indices.append(idx)

    print(f"\n{len(over_limit_indices)} items exceed max_prompt_length!")
    print(f"Their indices are:")
    print(over_limit_indices)
    print("="*40)

    return over_limit_indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "zwhe99/DeepMath-103K"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if local_dataset_path is not None:
        data_files = glob.glob(os.path.join(local_dataset_path, '*.parquet'))
        data_files = sorted(data_files)
        for i in data_files:
            print(i)
        dataset = datasets.load_dataset(
            "parquet", 
            data_files=data_files,
        )
    else:
        dataset = datasets.load_dataset(
            data_source,
        )

    train_dataset = dataset["train"]

    over_limit_indices = check_dataset_length()
    print(len(train_dataset), len(over_limit_indices))
    keep_indices = [i for i in range(len(train_dataset)) if i not in over_limit_indices]
    train_dataset = train_dataset.select(keep_indices)
    print(len(train_dataset))

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")

            question = question + " " + instruction_following

            answer = example.pop("final_answer")

            difficulty = example.pop("difficulty")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx, "difficulty": difficulty},
            }
            return data

        return process_fn

    old_columns = train_dataset.column_names
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=old_columns)

    train_dataset.to_parquet(out_path)
    
    # generate_rank(train_dataset=train_dataset)
