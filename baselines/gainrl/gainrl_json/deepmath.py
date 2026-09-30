# import argparse
# import torch
# import json
# import os
# import datasets
# import glob
# from verl.utils.hdfs_io import copy, makedirs

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
#     parser.add_argument(
#         "--output_path", default=None, help="The save directory for the preprocessed dataset."
#     )

#     args = parser.parse_args()
#     local_dataset_path = args.local_dataset_path

#     data_files = glob.glob(os.path.join(local_dataset_path, '*.parquet'))
#     dataset = datasets.load_dataset(
#         "parquet", 
#         data_files=data_files,
#     )

#     train_dataset = dataset["train"]

#     # add a row to each data item that represents a unique id
#     def make_map_fn(split):
#         def process_fn(example, idx):
#             problem = example.pop("question")
#             answer = example.pop("final_answer")
#             data = {
#                 "problem": problem, 
#                 "answer": answer, 
#             }
#             return data

#         return process_fn

#     old_columns = train_dataset.column_names
#     train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, remove_columns=old_columns)
    
#     train_dataset.to_parquet('/home/daiz/SAERL/gainrl_json/deepmath.parquet')
#     train_dataset = list(train_dataset)
#     with open(args.output_path, "w", encoding="utf-8") as fo:
#         json.dump(train_dataset, fo, ensure_ascii=False, indent=4)
