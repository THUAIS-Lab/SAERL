"""Distributed evaluation with explicit vLLM EngineCore cleanup."""
import glob
import os
from tqdm import tqdm
import json
import argparse
import ray
import torch
import gc
import re
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from verl.utils import hf_tokenizer
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score # gsm8k
from verl.utils.reward_score.math_reward import compute_score as math_reward_compute_score # math, amc, aime
from verl.utils.reward_score.prime_math import compute_score as prime_math_compute_score # minerva, olympiad
from verl.utils.reward_score.deepscaler import compute_score as deepscaler_compute_score
############################Config################################
"""Qwen3-1.7B"""
# NUM_GPUS = 4
# MAX_MODEL_LEN = 10240
# GPU_MEMORY_UTILIZATION = 0.9
# MAX_TOKENS = 8192
# REWARD_FUNC_MAP = {
#     "gsm8k": gsm8k_compute_score,
#     "aime": math_reward_compute_score,
#     "amc": math_reward_compute_score,
#     "math": math_reward_compute_score,
#     "minerva": prime_math_compute_score,
#     "olympiad": prime_math_compute_score,
# }

"""Qwen2.5-Math"""
NUM_GPUS = 1
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.8
MAX_TOKENS = 3072
REWARD_FUNC_MAP = {
    "gsm8k": math_reward_compute_score, # in my format, gsm8k's answer is within box
    "aime": math_reward_compute_score,
    "amc": math_reward_compute_score,
    "math": math_reward_compute_score,
    "minerva": deepscaler_compute_score,
    "olympiad": deepscaler_compute_score,
}

"""ds"""
# NUM_GPUS = 2
# MAX_MODEL_LEN = 9216
# GPU_MEMORY_UTILIZATION = 0.9
# MAX_TOKENS = 8192
# REWARD_FUNC_MAP = {
#     "gsm8k": gsm8k_compute_score,
#     "aime": math_reward_compute_score,
#     "amc": math_reward_compute_score,
#     "math": math_reward_compute_score,
#     "minerva": deepscaler_compute_score,
#     "olympiad": deepscaler_compute_score,
# }
#################################################################

@ray.remote(num_gpus=1)
class LLMWorker:
    def __init__(self, dataset_path, dataset_name):
        print(f"Worker initializing on GPU...")
        # 1. 在 Worker 初始化时加载数据，避免重复 IO
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self._load_dataset()
        
    def _load_dataset(self):
        print(f"Loading dataset from {self.dataset_path} ...")
        df = pd.read_parquet(self.dataset_path)
        
        # 预处理 Prompt
        self.prompts = [p.tolist() for p in df['prompt'].tolist()]
        
        # 预处理 Ground Truth
        ground_truths = df['reward_model'].tolist()
        self.ground_truths = [p['ground_truth'] for p in ground_truths]
        
        print(f"Loaded {len(self.prompts)} samples")

    def process_batch(self, ckpt_paths, args):
        metrics = []
        # 获取对应的 Reward Function
        reward_func = REWARD_FUNC_MAP.get(args.dataset)
        if not reward_func:
            raise ValueError(f"Reward function not found for {args.dataset}")

        for ckpt_path in tqdm(ckpt_paths, desc="Processing Ckpts"):
            try:
                # 执行单个模型的评估
                res = self.evaluate_single_model(ckpt_path, args, reward_func)
                metrics.append(res)
            except Exception as e:
                print(f"Error evaluating {ckpt_path}: {e}")
            finally:
                # 关键：强制清理显存，防止下一个模型加载时 OOM
                self._free_gpu_memory()
        return metrics

    def _free_gpu_memory(self):
        """强制清理 vLLM 占用的显存"""
        # 注意：vLLM 对象必须在 evaluate_single_model 中被销毁
        # 这里进行垃圾回收和显存清空
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        print(
            f"GPU memory after cleanup: {free_bytes / gib:.2f}/"
            f"{total_bytes / gib:.2f} GiB free "
            f"(worker PID={os.getpid()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})."
        )

    def evaluate_single_model(self, model_path, args, reward_func):
        print(f"Evaluating: {model_path}")
        
        tokenizer = hf_tokenizer(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm = None
        try:
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=GPU_MEMORY_UTILIZATION, 
                max_model_len=MAX_MODEL_LEN,
                enable_chunked_prefill=True,
            )

            sampling_params = SamplingParams(
                n=8,
                temperature=0.6,
                max_tokens=MAX_TOKENS,
                stop_token_ids=[tokenizer.eos_token_id],
            )
            # sampling_params = SamplingParams(
            #     temperature=0.0,
            #     max_tokens=MAX_TOKENS,
            #     stop_token_ids=[tokenizer.eos_token_id],
            # )

            formatted_prompts = [
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in self.prompts
            ]

            outputs = llm.generate(formatted_prompts, sampling_params)

            correct, total = 0, 0
            results = []

            # 遍历输出结果
            for i, output_item in enumerate(outputs):
                truth = self.ground_truths[i]
            
                for sample_output in output_item.outputs:
                    response = sample_output.text
                    is_ok = reward_func(response, truth)
                
                    correct += int(is_ok)
                    total += 1
                
                    results.append({
                        "question": self.prompts[i],
                        "true_answer": truth,
                        "full_response": response,
                        "correct": is_ok,
                    })

            metrics = {
                "accuracy": correct / total if total else 0.0,
                "correct": correct, 
                "total": total,
                "model_path": model_path,
                "timestamp": datetime.now().isoformat(),
            }

            if args.result_dir:
                os.makedirs(args.result_dir, exist_ok=True)
                fname = f"{args.dataset}_{extract_step_from_path(model_path)}.json"
                fname = os.path.join(args.result_dir, fname)
                with open(fname, "w") as fp:
                    json.dump({"metrics": metrics, "results": results}, fp, indent=2)
                print(f"Results saved to {fname}")
            return metrics
        finally:
            if llm is not None:
                try:
                    # vLLM 0.11 V1: release the EngineCore subprocess resources.
                    llm.llm_engine.engine_core.shutdown()
                except Exception as cleanup_error:
                    # Keep the original evaluation exception, if any, visible.
                    print(f"Warning: vLLM engine shutdown failed: {cleanup_error}")
                finally:
                    del llm


def get_evaluated_ckpts(args):
    result_jsons = glob.glob(os.path.join(args.result_dir, '*.json'))
    # 过滤属于当前 dataset 的结果
    result_jsons = [p for p in result_jsons if os.path.basename(p).startswith(args.dataset)]

    ckpts = []
    for p in result_jsons:
        with open(p, 'r') as f:
            data = json.load(f)
        ckpts.append(data['metrics']['model_path'])
    return ckpts

def extract_step_from_path(model_path: str):
    m = re.search(r"global_step_(\d+)", model_path)
    return int(m.group(1))


if __name__ == '__main__':
    # 自动处理 Ray 初始化，如果已存在则连接
    ray.init(ignore_reinit_error=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset_path", type=str, required=True, help="eval parquet file path")
    parser.add_argument("--dataset", type=str, required=True, help="the name of dataset")
    # parser.add_argument("--version", type=str, default="v1", help="version of metric")
    parser.add_argument("--p", type=str, required=True, help="the dir of all ckpts")
    parser.add_argument("--result_dir", type=str, default="./results", help="the dir of evaluated results")
    args = parser.parse_args()

    ckpt_paths = glob.glob(os.path.join(args.p, "*/merged_hf_model"))
    evaluated_ckpt_paths = get_evaluated_ckpts(args)
    unevaluated_ckpt_paths = []
    for p in ckpt_paths:
        if p not in evaluated_ckpt_paths:
            unevaluated_ckpt_paths.append(p)

    print(f"Total ckpts: {len(ckpt_paths)}")
    print(f"Already evaluated: {len(evaluated_ckpt_paths)}")
    print(f"Remaining to evaluate: {len(unevaluated_ckpt_paths)}")

    if len(unevaluated_ckpt_paths) == 0:
        print("All checkpoints evaluated.")
        ray.shutdown()
        exit(0)

    workers = [LLMWorker.remote(args.eval_dataset_path, args.dataset) for _ in range(NUM_GPUS)]

    chunks = [[] for _ in range(NUM_GPUS)]
    for i, ckpt in enumerate(unevaluated_ckpt_paths):
        chunks[i % NUM_GPUS].append(ckpt)

    # 4. 异步提交任务
    futures = []
    for i, worker in enumerate(workers):
        if chunks[i]: # 只有分配了任务的 worker 才启动
            print(f"Worker {i} assigned {len(chunks[i])} checkpoints.")
            future = worker.process_batch.remote(chunks[i], args)
            futures.append(future)

    # 5. 收集结果
    # ray.get 会阻塞直到所有任务完成
    results_nested = ray.get(futures)
    
    # metric = []
    # for batch_res in results_nested:
    #     metric.extend(batch_res)

    # # 6. 保存汇总结果
    # fname = f"evaluation/{args.version}/{args.dataset}_{args.version}.json"
    # os.makedirs(os.path.dirname(fname), exist_ok=True)
    # with open(fname, "w") as fp:
    #     json.dump(metric, fp, indent=2)
    # print(f"Summary saved to {fname}")

    ray.shutdown()


"""single gpu"""
# import glob
# import os
# from tqdm import tqdm
# import json
# import argparse
# from evaluation.eval_aime import evaluate_model as evaluate_model_aime
# from evaluation.eval_amc import evaluate_model as evaluate_model_amc
# from evaluation.eval_gsm8k import evaluate_model as evaluate_model_gsm8k
# from evaluation.eval_math import evaluate_model as evaluate_model_math
# from evaluation.eval_minerva import evaluate_model as evaluate_model_minerva
# from evaluation.eval_olympiad import evaluate_model as evaluate_model_olympiad

# def evaluated_ckpts(args):
#     result_dir = "/data/daiz/result/SAERL/v2/"
#     result_jsons = glob.glob(os.path.join(result_dir, '*.json'))
#     result_jsons = [p for p in result_jsons if args.dataset == os.path.basename(p).split('_')[0]]
    
#     ckpts = []
#     for p in result_jsons:
#         with open(p, 'r') as f:
#             data = json.load(f)
#         ckpt_path = data['metrics']['model_path']
#         ckpts.append(ckpt_path)
#     print(f"{len(result_jsons)} ckpts have been evaluated")

#     return ckpts


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--eval_dataset_path", type=str, default=None, help="eval json file path")
#     parser.add_argument("--dataset", type=str, default=None, help="the name of dataset")
#     parser.add_argument("--type", type=str, default=None, help="sae or grpo")
#     parser.add_argument("--version", type=str, default=None, help="version of metric")
#     parser.add_argument("--p", type=str, default=None, help="the dir of all ckpts")
#     args = parser.parse_args()

#     ckpt_paths = glob.glob(os.path.join(args.p, "*/merged_hf_model"))

#     eval_dispatch = {
#         "aime_64x": evaluate_model_aime,
#         "aime": evaluate_model_aime,
#         "amc": evaluate_model_amc,
#         "amc_32x": evaluate_model_amc,
#         "gsm8k": evaluate_model_gsm8k,
#         "math": evaluate_model_math,
#         "minerva": evaluate_model_minerva,
#         "olympiad": evaluate_model_olympiad,
#     }

#     metric = []
    
#     for ckpt_path in tqdm(ckpt_paths, desc="Processing", total=len(ckpt_paths)):
#         ckpts = evaluated_ckpts(args)
#         if ckpt_path not in ckpts:
#             params = {
#                 "model_path": ckpt_path, 
#                 "dataset_path": args.eval_dataset_path, 
#                 "batch_size": 512,
#                 "num_samples": None,                           
#                 "gpu_memory_utilization": 0.95,
#                 "save_results": True,
#             }
#             metric.append(eval_dispatch[args.dataset](**params))

#     fname = f"evaluation/{args.version}/{args.dataset}_{args.version}.json"
#     os.makedirs(os.path.dirname(fname), exist_ok=True)
#     with open(fname, "w") as fp:
#         json.dump(metric, fp, indent=2)
#     print(f"Results saved to {fname}")