# SAERL

SAERL 是独立的 Python 项目，提供静态梯度提取、SAE 特征筛选、数据选择、子集导出和 GRPO 训练。源码位于 `src/saerl/`；官方 verl 作为外部依赖安装，通过 `import verl` 使用。

verl 固定为原项目对应的 **0.7.0.dev0**，官方提交 **`875788ef6e218d0b98cb64fff919d9fef014f986`**，不会自动升级到新版。版本确认、调用链和局部覆盖的依据见 [上游审计](docs/upstream-audit.md)。

## 1. 安装与启动检查

以下命令均在仓库根目录执行，要求 Python 3.12。开发时使用 editable 安装，修改 `src/saerl/` 后即可生效。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# CPU 特征筛选、selection 和子集导出
python -m pip install -e .

# 检查 CPU 入口
saerl-feature-select --method energy --help
saerl-select --method abs-selected --help
saerl-export-subset --help
```

如果只处理已有提取结果，安装到这里即可。提取和训练需要 CUDA GPU，以及训练依赖：

```bash
# 先安装 Torch，再安装需要读取 Torch/CUDA 信息的 flash-attn
python -m pip install "torch==2.8.0" -c constraints-train.txt
python -m pip install packaging ninja
python -m pip install "flash-attn==2.8.1" --no-build-isolation -c constraints-train.txt
python -m pip install -e '.[train]' -c constraints-train.txt

# 核验官方 verl 的版本和核心源码指纹
python -m saerl.integrations.verl.compat

# 仅打印完整配置，不启动 Ray 或 GPU 任务
saerl-extract --cfg job
saerl-train --cfg job
```

`constraints-train.txt` 来自原 Python 3.12 / CUDA 12.8 环境的主要版本记录，不是所有平台通用的完整依赖锁。Torch、flash-attn 和 vLLM 需要匹配训练节点的 CUDA 环境；如果使用集群提供的 flash-attn wheel，可用该 wheel 替代上面的编译安装命令。全新 CUDA 环境的依赖安装尚未实测。

使用新虚拟环境，并移除指向旧 verl checkout 的 `PYTHONPATH` 设置。启动入口会核验官方依赖，遇到旧 editable 安装或修改过的 verl 会明确报错。

## 2. 准备输入

准备训练和验证 Parquet、Hugging Face 模型目录，以及 SAE decoder 权重目录。路径可以位于项目外。

- 数据使用 verl 的 RL 数据格式，通常包含 `prompt`（消息列表）、`data_source`、`reward_model.ground_truth` 和 `extra_info.index`。
- `extra_info.index` 必须是非空、唯一的整数，作为提取和导出的稳定 sample ID。DeepMath 数据的 `data_source` 使用 `zwhe99/DeepMath-103K`，与 reward 分发函数保持一致。
- SAE 目录包含 `config.json`，其中有 `feature_size`、`hidden_size`、`input_hookpoint`，并包含一个 `.pt` decoder state dict，或 `model.safetensors` / 其分片索引。decoder tensor 支持 `W_dec` 或 `decoder.weight`；维度和 hookpoint 必须与模型匹配。

在已激活的环境中设置路径，后续命令在同一 Bash 会话运行：

```bash
export SAERL_TRAIN_DATA=/data/train.parquet
export SAERL_VAL_DATA=/data/validation.parquet
export SAERL_MODEL_PATH=/models/Qwen2.5-Math-1.5B
export SAERL_SAE_PATH=/models/sae-layer26
export SAERL_WORK_DIR="$PWD/runs/quickstart"
export SAERL_RUN_NAME=extract-001
export SAERL_RUN_DIR="$SAERL_WORK_DIR/extraction/$SAERL_RUN_NAME"
```

下面以单节点两张 GPU、每批 8 个 prompt、每个 prompt 生成 4 个 rollout 为例。根据显存和数据规模调整 batch、长度和 GPU 数量；训练子集至少应满足设定的训练 batch size。

## 3. 提取静态梯度

```bash
saerl-extract \
  data.train_files="$SAERL_TRAIN_DATA" \
  data.val_files="$SAERL_VAL_DATA" \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path="$SAERL_MODEL_PATH" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  'trainer.logger=[console]' \
  static_gradient_extraction.sae_path="$SAERL_SAE_PATH" \
  static_gradient_extraction.output_dir="$SAERL_WORK_DIR/extraction" \
  static_gradient_extraction.run_name="$SAERL_RUN_NAME"
```

结果写入 `$SAERL_RUN_DIR`，包括 `step-*.parquet`、`representation_manifest.json` 和 `resolved_config.yaml`。同一目录不能作为新的提取任务重复使用；重新提取时换一个 `SAERL_RUN_NAME`。

提取捕获的是**层输出梯度**，保留 rollout → reward → GRPO advantage → backward 的计算过程，禁止 optimizer 和 LR scheduler 更新。当前提取模式限定 GRPO、legacy FSDP、Ulysses SP=1，支持尾批和按 step 续跑。

## 4. 特征筛选、selection 和子集导出

已有提取结果时，直接把 `SAERL_RUN_DIR` 指向对应目录，从这里开始：

```bash
saerl-feature-select --method energy \
  --run-dir "$SAERL_RUN_DIR" --top-k 256

saerl-select --method abs-selected \
  --run-dir "$SAERL_RUN_DIR" \
  --feature-indices "$SAERL_RUN_DIR/feature_selection/energy_feature_indices.npy" \
  --device cpu --cosine-threshold 0.7 \
  --output-dir "$SAERL_WORK_DIR/selection"

saerl-export-subset \
  --run-dir "$SAERL_RUN_DIR" \
  --selection-dir "$SAERL_WORK_DIR/selection" \
  --dataset "$SAERL_TRAIN_DATA"
```

特征筛选的 `--top-k` 不应超过 SAE feature size。selection 生成选中 ID、silent ID 和审计表；导出生成：

- `selected_deepmath.parquet`：选中的样本。
- `selected_plus_silent_deepmath.parquet`：选中的样本，加上 reward=0 的 silent 样本。

导出通过 `extra_info.index` 匹配样本，保留原数据的行顺序和 schema。selection 和导出拒绝覆盖已有结果；再次运行时换一个输出目录。

## 5. 在选中子集上启动 GRPO 训练

```bash
saerl-train \
  algorithm.adv_estimator=grpo \
  data.train_files="$SAERL_WORK_DIR/selection/selected_plus_silent_deepmath.parquet" \
  data.val_files="$SAERL_VAL_DATA" \
  data.train_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path="$SAERL_MODEL_PATH" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=10 \
  trainer.default_local_dir="$SAERL_WORK_DIR/checkpoints" \
  'trainer.logger=[console]'
```

checkpoint 写入 `$SAERL_WORK_DIR/checkpoints`。如果要直接在完整数据上训练，将 `data.train_files` 替换成 `"$SAERL_TRAIN_DATA"`。普通训练使用官方 Trainer/Worker。

以上 console 命令分别等价于 `python -m saerl.extract`、`saerl.feature_select`、`saerl.select`、`saerl.export_subset`、`saerl.train`。

## 其他参数

`saerl-select --method` 支持 `abs-selected`、`abs`、`signed-selected`、`signed`、`dense`、`mean-peak`、`top-k`、`coverage`、`raw-abs`、`raw-abs-selected`。`saerl-feature-select --method` 支持 `energy`、`variance`、`prevalence-variance`、`raw-energy`。选定方法后加 `--help` 查看参数。

定序训练使用 `data.saerl.enable=true data.saerl.processed_file=/data/order.pt`。`order.pt` 是**过滤后 dataset 的位置索引 tensor**，与稳定 sample ID 的用途不同。

提取续跑时保持原来的模型、配置、GPU/DP 布局和 `run_name`，在原提取命令后加 `static_gradient_extraction.resume_from_step=N`。已有恢复 step 的 rank 分片会保留，缺失分片会补齐。续跑通过跳过已处理批次实现，不能保证恢复原随机 rollout 序列。

## 开发与验证

```bash
python -m pip install -e '.[test]'
ruff check src tests scripts
pytest

# 安装训练依赖后，检查官方 verl 适配
pytest -m integration
python scripts/check_ray_imports.py

# 在至少两张可用 GPU 上运行端到端 smoke test
python scripts/gpu_smoke.py \
  --dataset "$SAERL_TRAIN_DATA" --model "$SAERL_MODEL_PATH" \
  --sae "$SAERL_SAE_PATH" --output-dir "$SAERL_WORK_DIR/smoke"
```

GPU smoke test 使用数据集前四条样本，要求它们是合法、不同 ID 的短 prompt，且能产生非零 advantage。它检查尾批提取、缺失 rank 分片续跑、selection、导出、一步 GRPO 和 checkpoint 保存。

当前已有 CPU 测试、官方适配和打包验证；真实 GPU/FSDP 流程尚未执行，Ray 子进程训练模块导入检查尚未通过。具体结果见 [验证记录](docs/validation.md)。

## 目录结构

```text
src/saerl/             # 源码、配置、官方 verl 适配
tests/                 # 核心算法和适配测试
scripts/               # Ray/GPU 验证及旧新 selection 对比
docs/                  # 版本审计与验证记录
pyproject.toml         # 包定义、依赖、启动入口
constraints-train.txt  # 原训练环境主要版本约束
LICENSE / NOTICE       # 许可和来源说明
```

旧 verl 源码、旧 recipe、评估结果、baseline、集群脚本和上游 CI 已从当前分支的文件树移除。需要查看旧实现时可切换到 `origin/dev`；本项目的安装和运行不依赖旧目录。
