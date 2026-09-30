# # aime eval
CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
    --eval_dataset_path "/data/daiz/data/SAERL/dataset/aime_eval.json" \
    --dataset "aime" \
    --type "sae" \
    --version "v2" \
    --p "/data/daiz/ckpt/saerl/BasicTest/sae_global_metric_v2/"

# CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
#     --eval_dataset_path "/data/daiz/data/SAERL/dataset/aime_eval.json" \
#     --dataset "aime" \
#     --type "grpo" \
#     --version "v2" \
#     --p "/data/daiz/ckpt/saerl/BasicTest/grpo_v2/"

# CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
#     --eval_dataset_path "/data/daiz/data/SAERL/dataset/aime_eval.json" \
#     --dataset "aime" \
#     --type "deduped_25" \
#     --version "v2" \
#     --p "/data/daiz/ckpt/saerl/BasicTest/deduped_25/"

# # amc eval
# CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
#     --eval_dataset_path "/data/daiz/data/SAERL/dataset/amc_eval.json" \
#     --dataset "amc" \
#     --type "sae" \
#     --version "v2" \
#     --p "/data/daiz/ckpt/saerl/BasicTest/sae_global_metric_v2/"

# # CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
# #     --eval_dataset_path "/data/daiz/data/SAERL/dataset/amc_eval.json" \
# #     --dataset "amc" \
# #     --type "grpo" \
# #     --version "v2" \
# #     --p "/data/daiz/ckpt/saerl/BasicTest/grpo_v2/"

# CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
#     --eval_dataset_path "/data/daiz/data/SAERL/dataset/amc_eval.json" \
#     --dataset "amc" \
#     --type "deduped_25" \
#     --version "v2" \
#     --p "/data/daiz/ckpt/saerl/BasicTest/deduped_25/"

# # # gsm8k
# # CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
# #     --eval_dataset_path "/data/daiz/data/SAERL/dataset/gsm8k_eval.json" \
# #     --dataset "gsm8k" \
# #     --type "sae" \
# #     --version "v2" \
# #     --p "/data/daiz/ckpt/saerl/BasicTest/sae_global_metric_v2/"

# # CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
# #     --eval_dataset_path "/data/daiz/data/SAERL/dataset/gsm8k_eval.json" \
# #     --dataset "gsm8k" \
# #     --type "grpo" \
# #     --version "v2" \
# #     --p "/data/daiz/ckpt/saerl/BasicTest/grpo_v2/"

# # CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
# #     --eval_dataset_path "/data/daiz/data/SAERL/dataset/gsm8k_eval.json" \
# #     --dataset "gsm8k" \
# #     --type "deduped_25" \
# #     --version "v2" \
# #     --p "/data/daiz/ckpt/saerl/BasicTest/deduped_25/"

# # minerva
# CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
#     --eval_dataset_path "/data/daiz/data/SAERL/dataset/minerva_eval.json" \
#     --dataset "minerva" \
#     --type "sae" \
#     --version "v2" \
#     --p "/data/daiz/ckpt/saerl/BasicTest/sae_global_metric_v2/"

# CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
#     --eval_dataset_path "/data/daiz/data/SAERL/dataset/minerva_eval.json" \
#     --dataset "minerva" \
#     --type "grpo" \
#     --version "v2" \
#     --p "/data/daiz/ckpt/saerl/BasicTest/grpo_v2/"

# CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
#     --eval_dataset_path "/data/daiz/data/SAERL/dataset/minerva_eval.json" \
#     --dataset "minerva" \
#     --type "deduped_25" \
#     --version "v2" \
#     --p "/data/daiz/ckpt/saerl/BasicTest/deduped_25/"

# # math
# CUDA_VISIBLE_DEVICES=0,5,6,7 python evaluation/eval.py \
#     --eval_dataset_path "/data/daiz/data/SAERL/dataset/math_eval.json" \
#     --dataset "math" \
#     --type "sae" \
#     --version "v2" \
#     --p "/data/daiz/ckpt/saerl/BasicTest/sae_global_metric_v2/"

CUDA_VISIBLE_DEVICES=7 python evaluation/eval.py \
    --eval_dataset_path "/data/daiz/data/SAERL/dataset/math_eval.json" \
    --dataset "math" \
    --type "grpo" \
    --version "v2" \
    --p "/data/daiz/ckpt/saerl/BasicTest/grpo_v2/"

CUDA_VISIBLE_DEVICES=7 python evaluation/eval.py \
    --eval_dataset_path "/data/daiz/data/SAERL/dataset/math_eval.json" \
    --dataset "math" \
    --type "deduped_25" \
    --version "v2" \
    --p "/data/daiz/ckpt/saerl/BasicTest/deduped_25/"

# olympiad
CUDA_VISIBLE_DEVICES=0 python evaluation/eval.py \
    --eval_dataset_path "/data/daiz/data/SAERL/dataset/olympiad_bench.json" \
    --dataset "olympiad" \
    --type "sae" \
    --version "v2" \
    --p "/data/daiz/ckpt/saerl/BasicTest/sae_global_metric_v2/"

CUDA_VISIBLE_DEVICES=5 python evaluation/eval.py \
    --eval_dataset_path "/data/daiz/data/SAERL/dataset/olympiad_bench.json" \
    --dataset "olympiad" \
    --type "grpo" \
    --version "v2" \
    --p "/data/daiz/ckpt/saerl/BasicTest/grpo_v2/"

CUDA_VISIBLE_DEVICES=6 python evaluation/eval.py \
    --eval_dataset_path "/data/daiz/data/SAERL/dataset/olympiad_bench.json" \
    --dataset "olympiad" \
    --type "deduped_25" \
    --version "v2" \
    --p "/data/daiz/ckpt/saerl/BasicTest/deduped_25/"