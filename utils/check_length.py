from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import torch
import os

# ================= 配置区 =================
# 1. 你的模型路径 (用于加载 Tokenizer)
MODEL_PATH = "/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-1.5B"

# 2. 你的数据路径 (可以是 parquet, jsonl 或 huggingface hub 路径)
DATA_PATH = "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/sorted_deepmath/train.parquet"
IN_RANK = "/WORK/PUBLIC/lijuanzi1_work/dz/data/rank_sorted_deepmath/embedding/deepmath_full_hidden_batch_only_v5mix8.pt"
OUT_RANK = "/home/fit/lijuanzi1/WORK/dz/SAERL/rank_sorted/sorted_deepmath_emb_v5mix8.pt"

max_prompt_length = 1024 

# 3. 数据集中存放 Prompt 的列名
# 如果是对话格式(messages)，通常叫 'messages' 或 'prompt'
# 如果是纯文本，也可能叫 'input'
PROMPT_COLUMN = "prompt" 

# 4. 是否是 Chat 格式 (List[Dict])
# 如果你的数据是 [{"role": "user", "content": "..."}], 设为 True
# 如果你的数据已经是纯文本字符串, 设为 False
IS_CHAT_FORMAT = True
# =========================================

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

    # === 统计分析 ===
    lengths = np.array(lengths)
    print("\n" + "="*40)
    print("       LENGTH STATISTICS       ")
    print("="*40)
    print(f"Min Length:     {np.min(lengths)}")
    print(f"Mean Length:    {np.mean(lengths):.2f}")
    print(f"Median Length:  {np.median(lengths):.2f}")
    print(f"P95 Length:     {np.percentile(lengths, 95):.2f}")
    print(f"P99 Length:     {np.percentile(lengths, 99):.2f}")
    print(f"Max Length:     {np.max(lengths)}  <-- 重点关注这个")
    print("="*40)

    # 建议设置
    suggested_max = int(np.max(lengths)) + 64 # 留一点 buffer
    print(f"\n[建议] 请将 config 中的 data.max_prompt_length 设置为至少: {suggested_max}")

    print(f"\n{len(over_limit_indices)} items exceed max_prompt_length!")
    print(f"Their indices are:")
    print(over_limit_indices)
    print("="*40)

    return over_limit_indices

def filter_data(over_limit_indices, rank):
    print(f"The indices of over limit items: {over_limit_indices}")

    is_in_mask = torch.isin(rank, over_limit_indices)
    keep_mask = ~is_in_mask

    # 获取在over_limit_indices中出现的rank元素
    over_limit_indices_in_rank = rank[is_in_mask]
    rank = rank[keep_mask]

    return rank, over_limit_indices_in_rank

def double_check(part1, part2, total):
    merged_sorted = torch.cat([part1.flatten(), part2.flatten()]).sort().values
    target_sorted = total.flatten().sort().values
    print(f"The first sample and the last sample of rank: {target_sorted[0]}, {target_sorted[-1]}")
    # 2. 比较形状是否一致（防止元素个数都对不上）
    if merged_sorted.shape != target_sorted.shape:
        print("元素个数不同！")
    else:
        # 3. 比较内容
        is_same = torch.equal(merged_sorted, target_sorted)
        print(f"元素集合一致: {is_same}")

# def generate_filtered_dataset(indices_to_remove, ds):
#     keep_indices = [i for i in range(len(ds)) if i not in indices_to_remove]
#     ds_filtered = ds.select(keep_indices)
#     ds_filtered.to_parquet(DATA_PATH.replace('train', f'train_{max_prompt_length}').replace('test', f'test{max_prompt_length}'))


if __name__ == "__main__":
    over_limit_indices = check_dataset_length()
    over_limit_indices = torch.tensor(over_limit_indices, dtype=torch.int64)
    
    print(f"Loading pt from {IN_RANK}")
    print(f"Saving indices to {OUT_RANK}")

    pt_file = torch.load(IN_RANK)
    rank = pt_file['indices']
    print(len(rank))
    rank = torch.tensor(rank, dtype=torch.int64)
    filter_rank, over_limit_indices_in_rank = filter_data(over_limit_indices, rank)
    print(len(filter_rank), len(over_limit_indices_in_rank))
    print(f"The len of indices after filtering: {len(filter_rank)}")
    print(f"The len of filtered indices: {len(over_limit_indices_in_rank)}\nThey are: {over_limit_indices_in_rank}")
    torch.save(filter_rank, OUT_RANK)

    file = torch.load(OUT_RANK)
    double_check(part1=file, part2=over_limit_indices_in_rank, total=rank)