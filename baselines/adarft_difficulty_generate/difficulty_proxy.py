import pandas as pd
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import ray
import os
import json
from typing import List, Tuple, Any
from verl.utils.reward_score.deepscaler import compute_score

# ================= 配置区域 =================
MODEL_PATH = "/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-1.5B/"  # 你的本地模型路径
DATA_PATH = "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/sorted_deepmath/train.parquet" # 你的数据路径
OUTPUT_PATH = "/WORK/PUBLIC/lijuanzi1_work/dz/data/rank_sorted_deepmath/deepmath_difficulty_mean128.pt" # 输出文件路径
SAVE_RESPONSE_DIR = "/WORK/PUBLIC/lijuanzi1_work/dz/result/deepmath_difficulty_proxy_128/"
NUM_GPUS = 4  # 你想用的显卡数量

# vLLM 配置
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 4096
MAX_TOKENS = 3072

# 采样配置
N_SAMPLES = 128
TEMPERATURE = 0.6    # 必须 > 0 才有随机性
TOP_P = 0.95
# ===========================================

@ray.remote(num_gpus=1)
class LLMWorker:
    def __init__(self):
        print(f"Worker initializing on a GPU...")
        # 在这里初始化 vLLM，就像单卡一样写
        self.llm = LLM(
            model=MODEL_PATH,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True, 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            n=N_SAMPLES,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
    
    def process_batch(self, batch_data, save_response=False, save_response_dir=None):
        """
        Args:
            batch_data: list of (index, prompt, ground_truth)
            save_response: whether to save detailed responses
            save_response_dir: directory to save detailed responses
        """
        # 1. 拆包：将元组列表拆分为三个独立的列表
        indices = [item[0] for item in batch_data]
        prompts = [item[1] for item in batch_data]
        ground_truths = [item[2] for item in batch_data]
        assert len(indices) == len(prompts) == len(ground_truths), "mismatch length"
        
        # 2. 批量推理
        formatted = [
            self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        outputs = self.llm.generate(formatted, self.sampling_params)
        
        results = []
        print("Grading answers...")
        assert len(indices) == len(ground_truths) == len(outputs) == len(prompts), "mismatch length after generation"
        for idx, gt, output, prompt in tqdm(zip(indices, ground_truths, outputs, prompts), total=len(indices)):
            correct_count = 0
            full_responses = []
            for sample in output.outputs:
                pred_text = sample.text
                is_correct = compute_score(solution_str=pred_text, ground_truth=gt)
                correct_count += is_correct
                full_responses.append({
                    "index": idx, 
                    "question": prompt, 
                    "ground_truth": gt, 
                    "response": pred_text, 
                    "correct": is_correct
                })
            accuracy = correct_count / N_SAMPLES
            
            # 只保留轻量级结果
            results.append({
                "index": idx,
                "accuracy": accuracy
            })

            if save_response:
                os.makedirs(save_response_dir, exist_ok=True)
                fname = f"{idx:06d}.json"
                fname = os.path.join(save_response_dir, fname)
                with open(fname, "w") as fp:
                    json.dump(full_responses, fp, indent=2)
                print(f"Results saved to {fname}")
            
        return results


def main():
    # 初始化 Ray
    # ignore_reinit_error=True 防止在 notebook 里重复运行报错
    ray.init(ignore_reinit_error=True)
    
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    prompts = [p.tolist() for p in df['prompt']]
    ground_truths = [p['ground_truth'] for p in df['reward_model']]
    indices = [p['index'] for p in df['extra_info']]
    data_triplets = list(zip(indices, prompts, ground_truths))

    assert len(prompts) == len(ground_truths) == len(indices) == len(data_triplets), "data mismatch"
    for i in range(len(indices)):
        assert i == indices[i], "wrong order"

    os.makedirs(SAVE_RESPONSE_DIR, exist_ok=True)
    solved_list = os.listdir(SAVE_RESPONSE_DIR)
    data_triplets_unsolved = []
    for tri in data_triplets:
        idx = tri[0]
        if f"{idx:06d}.json" not in solved_list:
            data_triplets_unsolved.append(tri)
    print(f"Total samples: {len(data_triplets)}")
    print(f"Solved samples: {len(solved_list)}")
    print(f"Unsolved samples: {len(data_triplets_unsolved)}")
    assert len(data_triplets) == len(solved_list) + len(data_triplets_unsolved), "count mismatch"
    data_triplets = data_triplets_unsolved
    print(f"Proceeding with {len(data_triplets)} unsolved samples...")

    workers = [LLMWorker.remote() for _ in range(NUM_GPUS)]
    
    # 切分数据
    chunks = [[] for _ in range(NUM_GPUS)]
    for i, tri in enumerate(data_triplets):
        chunks[i % NUM_GPUS].append(tri)
    
    total = 0
    for chunk in chunks:
        total += len(chunk)
    assert total == len(data_triplets), "chunk error"
    
    print(f"Dispatching tasks to Ray workers...({total} prompts in total)")
    futures = []
    for i, worker in enumerate(workers):
        if i < len(chunks): # 防止 chunks 比 worker 少的情况
            # .remote() 是异步调用，立即返回一个 future 对象
            future = worker.process_batch.remote(
                batch_data=chunks[i], 
                save_response=True,
                save_response_dir=SAVE_RESPONSE_DIR
            )
            futures.append(future)
    print("Waiting for results...")
    results_nested = ray.get(futures)
    
    # 合并结果
    results_list = []
    for batch_res in results_nested:
        results_list.extend(batch_res)
    
    # # accuracy from high to low, aka difficulty from easy to hard
    # sorted_results = sorted(results_list, key=lambda x: x['accuracy'], reverse=True)

    # final_indices = [item['index'] for item in sorted_results]
    # final_accs = [item['accuracy'] for item in sorted_results]
    
    # save_data = {
    #     "indices": final_indices,              # List[int]
    #     "accuracy": final_accs                 # List[float]
    # }
    # torch.save(save_data, OUTPUT_PATH)
    # print(f"Saved {len(final_indices)} records to {OUTPUT_PATH}")
    # print("Top 5 indices (Easiest):", final_indices[:5])
    # print("Bottom 5 indices (Hardest):", final_indices[-5:])

    # 清理资源
    ray.shutdown()

if __name__ == "__main__":
    import time
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    time.sleep(1)
    print(f"difficulty proxy 运行时间: {end_time - start_time:.4f} 秒")
