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

from verl.utils.hdfs_io import copy, makedirs

def generate_rank(train_dataset, rank_path):
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

    torch.save(save_data, rank_path)
    print(f"Saved {len(final_indices)} records to {rank_path}")
    print("Top 5 indices (Easiest):", final_indices[:5], final_difficulties[:5])
    print("Bottom 5 indices (Hardest):", final_indices[-5:], final_difficulties[-5:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/DeepMath-103K", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--rank-path", default=None)

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

    local_save_dir = args.local_save_dir

    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    
    if args.rank_path:
        generate_rank(train_dataset=train_dataset, rank_path=args.rank_path)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
