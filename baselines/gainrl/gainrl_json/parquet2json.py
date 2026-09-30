import pandas as pd
import json
from datasets import load_dataset

def make_map_fn():
    def process_fn(example):
        problem = example.pop('extra_info')['question']
        answer = example.pop('reward_model')['ground_truth']
        data = {
            "problem": problem, 
            "answer": answer, 
        }
        return data
    return process_fn
              
if __name__ == "__main__":
    data_files = {
        "train": "/data/daiz/data/SAERL/my_data/deepmath/train.parquet",
    }
    output_path = "/home/daiz/SAERL/deepmath.json"
    dataset = load_dataset(path='parquet', data_files=data_files)
    train_dataset = dataset['train']

    train_dataset = train_dataset.map(function=make_map_fn(), remove_columns=train_dataset.column_names)
    
    train_dataset = list(train_dataset)
    with open(output_path, "w", encoding="utf-8") as fo:
        json.dump(train_dataset, fo, ensure_ascii=False, indent=4)
