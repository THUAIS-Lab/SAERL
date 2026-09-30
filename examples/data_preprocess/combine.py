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

import argparse
import json
import os
import pandas as pd

import datasets

from verl.utils.hdfs_io import copy, makedirs



if __name__ == "__main__":
    # 1. 定义你的两个Parquet文件路径
    path1 = "/data/daiz/data/SAERL/my_data/deepmath/train.parquet"
    path2 = "/data/daiz/data/SAERL/my_data/numinamath_filtered/numinamath_107k.parquet"

    # 2. 读取文件到Pandas DataFrame
    print("正在读取文件...")
    df1 = pd.read_parquet(path1)
    df2 = pd.read_parquet(path2)

    # 3. 垂直合并两个DataFrame
    # ignore_index=True 会重置新DataFrame的索引，避免索引重复
    print("正在合并数据...")
    df_combined = pd.concat([df1, df2], ignore_index=True)

    # 4. 将合并后的DataFrame保存为新的Parquet文件
    output_path = "/data/daiz/data/SAERL/my_data/deepmath/deepmath_210k.parquet"
    print(f"正在保存到 {output_path} ...")
    df_combined.to_parquet(output_path, index=False)

    print("合并完成！")

    local_dataset_path = output_path

    dataset = datasets.load_dataset(
        "parquet", 
        data_files=local_dataset_path,
    )

    train_dataset = dataset["train"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            data_source = example.pop("data_source")
            extra_info = example.pop("extra_info")
            extra_info['index'] = idx
            data = {
                "data_source": data_source,
                "extra_info": extra_info,
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    local_save_dir = "/data/daiz/data/SAERL/my_data/deepmath/"

    train_dataset.to_parquet(os.path.join(local_save_dir, "deepmath_210k_new.parquet"))