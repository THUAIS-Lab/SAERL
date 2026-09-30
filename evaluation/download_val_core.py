import wandb
import json
import os

# ================= 配置区域 =================
ENTITY = "dfse-xi-an-jiaotong-university-" 
PROJECT = "saerl_Qwen2.5-Math-1.5B-Instruct" 

# 这里填你那些 Run 共同的名字 (网页上显示的 Name)
TARGET_RUN_NAME = "grpo"  # <--- 请修改这里
OUTPUT_DIR = "/home/daiz/SAERL/evaluation/qwen2"  # 结果保存在这个文件夹里
EXP_NAME = "grpo_wandb"
# ===========================================

# 1. 创建保存文件夹
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"1. 正在连接 Project: {PROJECT} ...")
api = wandb.Api()

# === 关键步骤：使用 filters 搜索同名 Run ===
print(f"2. 正在搜索 display_name 为 '{TARGET_RUN_NAME}' 的所有 Run ...")
runs = api.runs(
    path=f"{ENTITY}/{PROJECT}",
    filters={"display_name": TARGET_RUN_NAME}
)

if len(runs) == 0:
    print(f"❌ 未找到名为 '{TARGET_RUN_NAME}' 的 Run，请检查拼写。")
    exit()

print(f"   --> 找到了 {len(runs)} 个相关的 Run (IDs: {[r.id for r in runs]})")
print("   --> 开始下载并缝合数据...")

# 用于暂存整理好的数据
# 结构: { "数据集名": [ {step: 10, value: 0.5}, {step: 20, value: 0.6} ] }
dataset_map = {}

for run in runs:
    # 扫描所有历史数据
    scan = run.scan_history()

    for row in scan:
        # 获取 Step (优先使用 global_step，如果没有则用系统 step)
        step = row.get('training/global_step', row.get('_step'))
        if step == None:
            step = 0
        
        # 遍历这一步的所有指标
        for key, value in row.items():
            # 筛选条件：必须是 val-core 且包含 mean@8
            if "val-core" in key and "mean@8" in key:
                # === 核心逻辑：从 Key 中提取数据集名称 ===
                # Key 示例: val-core/Maxwell-Jia/AIME_2024/acc/mean@8
                # 我们按照 "/" 分割
                dataset_name = key.split('/')[2]

                # === 存入字典 ===
                if dataset_name not in dataset_map:
                    dataset_map[dataset_name] = []
                
                # 过滤掉 None 值
                if value is not None:
                    dataset_map[dataset_name].append({
                        "model_path": os.path.join(f"/data/daiz/ckpt/saerl/BasicTest/{TARGET_RUN_NAME}/", f"global_step_{step}/merged_hf_model"),
                        "accuracy": value
                    })

print(f"3. 数据整理完成，共识别出 {len(dataset_map)} 个数据集。")
print("4. 正在写入 JSON 文件...")

for dataset_name, data_list in dataset_map.items():
    if dataset_name == 'AIME_2024':
        file_path = os.path.join(OUTPUT_DIR, f"aime_{EXP_NAME}.json")
    elif dataset_name == 'AMC-23':
        file_path = os.path.join(OUTPUT_DIR, f"amc_{EXP_NAME}.json")
    elif dataset_name == 'MATH-500':
        file_path = os.path.join(OUTPUT_DIR, f"math_{EXP_NAME}.json")
    elif dataset_name == 'minervamath':
        file_path = os.path.join(OUTPUT_DIR, f"minerva_{EXP_NAME}.json")
    elif dataset_name == 'OlympiadBench':
        file_path = os.path.join(OUTPUT_DIR, f"olympiad_{EXP_NAME}.json")
    else:
        file_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{EXP_NAME}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2) # indent=4 让 json 会自动换行缩进，方便人类阅读
        
    print(f"   --> 已保存: {file_path} (共 {len(data_list)} 条记录)")

print("\n✅ 全部完成！请查看 dataset_jsons 文件夹。")