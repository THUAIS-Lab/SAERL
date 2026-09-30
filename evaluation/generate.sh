#!/bin/bash

# ================= 配置区域 =================
CKPT_ROOT="/data/daiz/ckpt/saerl/BasicTest/grpo_v2/"
OUTPUT_DIR="/home/daiz/SAERL/eval_results/grpo"
DATA_PATH="/data/daiz/data/SAERL/my_data/amc23/train.parquet"
export CUDA_VISIBLE_DEVICES=4,5,6,7
# ===========================================

mkdir -p "$OUTPUT_DIR"

# 扫描包含 "step" 的文件夹 (兼容 global_step_xxx 或 step_xxx)
CKPT_DIRS=$(find "$CKPT_ROOT" -maxdepth 1 -type d | grep "global_step_100" | sort -V)

if [ -z "$CKPT_DIRS" ]; then
    echo "❌ 未找到任何 Step 目录，请检查路径。"
    exit 1
fi

for ckpt_path in $CKPT_DIRS; do
    step_name=$(basename "$ckpt_path")
    output_file="${OUTPUT_DIR}/${step_name}.parquet"

    # ================= 核心修改逻辑：智能寻找模型路径 =================
    
    # 优先级 1: 检查是否存在 merged_hf_model (你现在的情况)
    if [ -d "${ckpt_path}/merged_hf_model" ]; then
        target_model_path="${ckpt_path}/merged_hf_model"
        echo "🔍 [${step_name}] 发现合并模型: merged_hf_model"

    # 优先级 2: 检查是否存在 actor (旧的/未合并的 checkpoint)
    elif [ -d "${ckpt_path}/actor" ]; then
        target_model_path="${ckpt_path}/actor"
        echo "🔍 [${step_name}] 发现原始 Actor: actor"

    # 优先级 3: 假设当前 step 文件夹本身就是模型 (例如你把合并后的文件直接放在 step_100 里)
    elif [ -f "${ckpt_path}/config.json" ]; then
        target_model_path="${ckpt_path}"
        echo "🔍 [${step_name}] 文件夹本身即为模型"

    else
        echo "⚠️  [${step_name}] 跳过: 未找到有效的模型文件 (merged_hf_model/actor/config.json)"
        continue
    fi
    # =================================================================

    # 断点续跑
    if [ -f "$output_file" ]; then
        echo "⏩ 结果已存在，跳过..."
        continue
    fi

    echo "🚀 开始验证..."
    
    # 注意：参数依然叫 actor.path，这是代码里的变量名，传任何 HF 路径都可以
    python3 -m verl.trainer.main_generation \
        model.path="$target_model_path" \
        data.output_path="$output_file" \
        data.path="$DATA_PATH" \
        data.n_samples=1 \
        rollout.n=8 \
        +trainer.project_name="saerl_eval" \
        +trainer.experiment_name="grpo_v2"

    if [ $? -eq 0 ]; then
        echo "✅ ${step_name} 完成"
    else
        echo "❌ ${step_name} 失败"
    fi
done