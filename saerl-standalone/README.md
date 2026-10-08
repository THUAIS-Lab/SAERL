# SAERL

独立的梯度表示、数据选择和 GRPO 训练项目。训练引擎通过 `import verl` 使用官方版本，项目自身不包含 `verl/` 源码。

官方依赖固定为 **0.7.0.dev0 / `875788ef6e218d0b98cb64fff919d9fef014f986`**。该提交与原仓库初始快照的 1175 个文件全部一致；审计和局部覆盖的理由见 [上游审计](docs/upstream-audit.md)。

## 安装

在本目录执行以下命令。使用单独的 Python 3.12 环境，避免旧项目的 editable 安装及 `PYTHONPATH` 遮蔽官方依赖。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# CPU selection / 子集导出；不安装 verl、OpenSAE、vLLM
python -m pip install .
# GPU 训练 / 提取：按集群 CUDA 条件安装 Torch / flash-attn 后执行
python -m pip install '.[train]' -c constraints-train.txt
python -m saerl.integrations.verl.compat
```

`constraints-train.txt` 固定本次检查到的主要版本；它不是所有平台通用的完整依赖锁。训练额外依赖通过官方 Git SHA 安装 verl。入口会核验核心官方文件指纹，发现改过的源码或其他版本时明确报错。

## 核心闭环

路径都由用户指定。模型和 SAE 权重放在项目外；SAE 目录须包含 `config.json` 和 `.pt` 或 safetensors decoder。

```bash
python -m saerl.extract \
  data.train_files=/data/train.parquet \
  data.val_files=/data/validation.parquet \
  actor_rollout_ref.model.path=/models/Qwen2.5-Math-1.5B \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  trainer.n_gpus_per_node=2 \
  static_gradient_extraction.sae_path=/models/sae-layer26 \
  static_gradient_extraction.output_dir=/results/extraction

python -m saerl.feature_select --method energy \
  --run-dir /results/extraction/RUN --top-k 256

python -m saerl.select --method abs-selected \
  --run-dir /results/extraction/RUN \
  --feature-indices /results/extraction/RUN/feature_selection/energy_feature_indices.npy \
  --cosine-threshold 0.7 --output-dir /results/selection

python -m saerl.export_subset \
  --run-dir /results/extraction/RUN --selection-dir /results/selection \
  --dataset /data/train.parquet

python -m saerl.train \
  algorithm.adv_estimator=grpo \
  data.train_files=/results/selection/selected_plus_silent_deepmath.parquet \
  data.val_files=/data/validation.parquet \
  actor_rollout_ref.model.path=/models/Qwen2.5-Math-1.5B \
  actor_rollout_ref.rollout.name=vllm \
  trainer.n_gpus_per_node=2
```

旧的 Parquet 分片、manifest、selection 参数和 `static_gradient_extraction.*` 参数继续适用。导出默认生成 `selected_deepmath.parquet` 和 `selected_plus_silent_deepmath.parquet`；后者只加入 reward=0 的 silent 样本。`--dataset` 未指定时保留旧的 `selection-dir/sorted_deepmath.parquet` 约定。

`saerl.select --method` 支持 `abs-selected`、`abs`、`signed-selected`、`signed`、`dense`、`mean-peak`、`top-k`、`coverage`、`raw-abs`、`raw-abs-selected`。`saerl.feature_select --method` 支持 `energy`、`variance`、`prevalence-variance`、`raw-energy`。选定方法后加 `--help` 查看其原有参数。

普通训练使用官方 Trainer/Worker。只有提取模式使用 SAERL 子类，保留 rollout → reward → advantage → backward 路径，捕获**层输出梯度**，禁止 optimizer 和 LR scheduler 更新。静态提取限定 GRPO、legacy FSDP、Ulysses SP=1；支持尾批、分片和按 step 续跑。

定序训练继续支持 `data.saerl.enable=true data.saerl.processed_file=/data/order.pt`。该文件是**过滤后 dataset 的位置索引 tensor**；它不是稳定 sample ID。子集导出使用 `extra_info.index`，不会把 sample ID 当作行号。

续跑须指定同一 `run_name` 与 `resume_from_step`，保持模型、配置和 DP 布局一致。已有首个恢复 step 的 rank 分片保留，缺失分片补齐；这仍是跳批续跑，不能保证恢复原随机 rollout 序列。

## 验证与独立 Git

```bash
python -m pip install '.[test]'
ruff check src tests scripts
pytest
# 安装 train extra 后运行官方适配测试
pytest -m integration
# 可选：在训练节点检查 Ray 子进程的依赖加载
python scripts/check_ray_imports.py
# 在至少两张可用 GPU 上执行；输入数据须有四个合法、不同 ID 的短 prompt
python scripts/gpu_smoke.py --dataset /data/train.parquet \
  --model /models/Qwen2.5-Math-1.5B --sae /models/sae-layer26 \
  --output-dir /results/saerl-smoke
```

当前对比分支把本目录作为普通文件保存，原实现保留在外层仓库中。独立建仓时只复制本目录的源码、测试、文档、配置和脚本到新目录（排除 `.venv`、构建产物和运行数据），执行 `git init -b main`；不复制外层 `.git`。无需把官方 verl checkout 放进新仓库。
