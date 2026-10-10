# SAERL

SAERL 是独立的 Python 项目，提供梯度提取、Gradient-SAE 数据筛选、GRPO 训练和课程训练。源码直接放在 `src/`，官方 verl 作为外部依赖安装并 import，固定使用原项目对应的 `0.7.0.dev0`。

## 文件功能

```text
run/
  common_qw25.sh             # 三个启动脚本共用的数据、模型路径和训练参数
  extract_grad_qw25.sh       # Qwen2.5-Math-1.5B 静态梯度提取
  grpo_qw25.sh               # 完整训练集上的 GRPO 训练
  sae_qw25.sh                # 按 processed_file 指定的子集和课程顺序训练
src/
  __init__.py               # SAERL 包和版本号
  selection/
    __init__.py             # selection Python 包
    extraction.py           # 聚合层输出梯度的 response/rollout mean，写出结果；提取入口
    sae_projection.py       # 加载 SAE decoder，执行梯度投影和逐 token Top-K
    representations.py      # 定义提取结果的 Parquet 格式，批量读取稀疏 response mean
    feature_selection.py    # Normalized Energy Top-K，选择全数据共享 features；入口
    response_pattern.py     # 在共享 features 上构造 absolute response-mean pattern
    sample_selection.py     # exact cosine greedy，生成 gradient core set；入口
    export_subset.py        # core 与全部零奖励样本合并，导出训练子集；入口
  verl/
    __init__.py             # SAERL 的 verl 接入包
    train.py                # Hydra/Ray 训练入口，调用外部官方 verl
    runner.py               # 训练委托给官方流程；提取时注册 SAERL Trainer/Worker
    curriculum_sampler.py   # 加载 processed_file，按原数据集行号选择样本并保留课程顺序
    extraction_config.py    # 设置单遍静态提取、FSDP 和 step 续跑参数
    trainer.py              # 提取专用循环：rollout → reward → GRPO advantage → backward
    worker.py               # 提取的 Ray RPC、FSDP sharding/offload；不更新 scheduler
    actor.py                # 提取的 loss、backward 和层输出梯度 hook；不更新参数
    reward.py               # DeepMath、AIME、AMC 等数据集的数学奖励评分
  config/
    __init__.py             # 可安装的 Hydra 配置包
    base.yaml               # 共用奖励函数和静态梯度提取参数
    extract.yaml            # 静态梯度提取配置
    train.yaml              # 普通训练配置
    saerl.yaml              # 子集课程训练配置，接入自定义 sampler
evaluation/
  eval.py                   # 使用 Ray/vLLM 评估已合并为 HF 模型的 checkpoint
  eval.sh                   # 各数学评估集的调用示例，使用原有集群路径
pyproject.toml              # 包定义、命令入口、配置打包和依赖声明
requirements-train.txt      # 原训练环境的依赖版本及固定提交的官方 verl
README.md                   # 文件说明、安装和运行方法
LICENSE                     # Apache-2.0 许可证
.gitignore                  # 排除权重、数据、运行产物、缓存和构建产物
```

`src/` 安装后的 Python 包名是 `saerl`，由 `pyproject.toml` 映射；`src/verl/` 属于 `saerl.verl`，官方框架通过 `import verl` 导入。框架接入集中在 `src/verl/`，梯度处理和筛选算法集中在 `src/selection/`。

`.git/` 用于当前分支与历史版本对比；运行环境挂载的 `.agents/`、`.aws/`、`.codex/` 不属于项目，也不进入安装包。

## 安装

使用 Python 3.12，在仓库根目录执行。建议使用新环境，清除指向旧 verl checkout 的 `PYTHONPATH`，避免导入旧仓库。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 先安装 Torch，再安装依赖 Torch/CUDA 的 flash-attn
python -m pip install 'torch==2.8.0' packaging ninja
python -m pip install 'flash-attn==2.8.1' --no-build-isolation
python -m pip install -e '.[train]'
```

`train` extra 直接读取根目录的 `requirements-train.txt`，也可在安装 Torch 和 flash-attn 后执行 `python -m pip install -r requirements-train.txt`、`python -m pip install -e .`。

依赖版本沿用原 Python 3.12 / CUDA 12.8 环境，包括 Torch 2.8.0、vLLM 0.11.0 和 flash-attn 2.8.1；flash-attn 也可安装集群提供的匹配 wheel。全新 CUDA 环境的依赖安装尚未实测。如果只处理已有梯度结果，执行 `python -m pip install -e .` 即可进行 CPU 特征筛选、样本筛选和导出。

## 启动提取和训练

三个脚本共用 `run/common_qw25.sh`。默认保留原来的 DeepMath、Qwen2.5-Math-1.5B、layer26 SAE 和课程索引文件路径，输出改到当前项目的 `runs/qw25/`。在原集群上安装后即可直接运行；迁移环境时，修改公共脚本或通过环境变量覆盖路径。

```bash
# 提取静态梯度，默认使用 2 张 GPU
bash run/extract_grad_qw25.sh

# 完整训练集上的 GRPO，默认使用 4 张 GPU
bash run/grpo_qw25.sh

# 已筛选并排好课程顺序的子集上的 SAERL，默认使用 4 张 GPU
bash run/sae_qw25.sh
```

这三个入口分别启动独立任务，按需要运行。脚本会切换到项目根目录；也可以使用绝对路径从其他目录启动，或直接执行 `./run/sae_qw25.sh`。Python 默认使用已激活环境中的 `python`，也可设置 `PYTHON_BIN=/path/to/python`。

### 路径和参数

| 环境变量 | 用途 / 默认值 |
| --- | --- |
| `SAERL_TRAIN_DATA` | 完整训练 Parquet；默认原 `sorted_deepmath/train.parquet` |
| `SAERL_VAL_DATA` | 验证 Parquet 路径或 Hydra 路径列表；默认 AIME 2024、math500 |
| `SAERL_MODEL_PATH` | Hugging Face 模型目录；默认原 Qwen2.5-Math-1.5B |
| `SAERL_SAE_PATH` | SAE decoder 目录；默认原 OpenSAE layer26，仅提取使用 |
| `SAERL_PROCESSED_FILE` | 筛选并完成课程排序的原数据行号 `.pt`；仅 SAERL 训练使用 |
| `SAERL_WORK_DIR` | 运行输出目录；默认当前项目的 `runs/qw25` |
| `SAERL_RUN_NAME` | 提取任务目录名；默认启动时间戳 |
| `SAERL_EXTRACT_GPUS` / `SAERL_TRAIN_GPUS` | 提取 / 训练 GPU 数；默认 2 / 4 |
| `SAERL_TRAIN_BATCH_SIZE` / `SAERL_PPO_MINI_BATCH_SIZE` | prompt batch / PPO mini batch；默认 128 / 64 |
| `SAERL_ROLLOUT_N` / `SAERL_TOTAL_EPOCHS` | 每个 prompt 的 rollout 数 / 训练 epoch 数；默认 8 / 30 |
| `SAERL_TOKEN_TOP_K` / `SAERL_PROJECTION_CHUNK_SIZE` | 逐 token Top-K / 投影分块大小；默认 128 / 32 |

例如，在其他机器上配置输入路径：

```bash
export SAERL_TRAIN_DATA=/data/deepmath/train.parquet
export SAERL_VAL_DATA='["/data/aime/train.parquet","/data/math500/test.parquet"]'
export SAERL_MODEL_PATH=/models/Qwen2.5-Math-1.5B
export SAERL_SAE_PATH=/models/OpenSAE-Qwen2.5-Math-1.5B-layer26
export SAERL_PROCESSED_FILE=/data/curriculum/deepmath_selected_sorted.pt
export SAERL_WORK_DIR="$PWD/runs/qw25"
```

公共脚本保留原训练设置：prompt 长度 1024、response 长度 3072、rollout 温度 0.6、FSDP、动态 batch 和 gradient checkpointing。GRPO/SAERL 使用学习率 `1e-6`、KL 系数 `0.001`，每 20 step 保存和验证。其他 Hydra 参数直接追加到脚本末尾，会覆盖脚本默认值：

```bash
bash run/grpo_qw25.sh trainer.total_epochs=1 'trainer.logger=[console]'
bash run/sae_qw25.sh data.saerl.processed_file=/data/another_curriculum.pt
bash run/extract_grad_qw25.sh static_gradient_extraction.gradient_feature_topk=64

# 只打印最终配置，不启动 Ray/GPU
bash run/extract_grad_qw25.sh --cfg job
bash run/grpo_qw25.sh --cfg job
bash run/sae_qw25.sh --cfg job
```

默认 WandB 离线记录在项目 `wandb/`；可用 `WANDB_MODE=online` 覆盖。模型 checkpoint 分别写入 `$SAERL_WORK_DIR/checkpoints/grpo` 和 `$SAERL_WORK_DIR/checkpoints/saerl`，默认 `resume_mode=auto` 会从已有 checkpoint 恢复。启动新实验时设置新的 `SAERL_WORK_DIR` 或 `trainer.default_local_dir`。

### processed_file 与 SAERL 训练

`processed_file` 沿用原来直接修改 verl 时的格式：**一维整数 Torch Tensor，元素是 sample 在完整训练数据集中的行号，排列顺序就是课程顺序**。DeepMath 的行号范围是 `0–103021`。例如 `[34226, 78480, 929, ...]` 表示依次训练完整 Parquet 中这些行的样本；文件中的样本已经完成筛选和排序。

`sae_qw25.sh` 读取完整的 `SAERL_TRAIN_DATA`，官方 DataLoader 使用 `SAERLSampler` 按该列表取样。只训练列表指定的样本，保留原始排列及重复项，每个 epoch 重用该顺序。batch 和尾批处理沿用官方训练逻辑。

**`SAERL_TRAIN_DATA` 必须与生成 processed_file 时的完整数据集及行序一致**。这份文件使用原数据行号，因此这里仍传完整 Parquet。SAERL 配置关闭自动长 prompt 过滤和数据截取，以免改变行号；`data.shuffle=false`、`data.dataloader_num_workers=0` 用于保留课程顺序。

该脚本执行 SAERL 的训练阶段；前面的 Gradient-SAE 筛选与课程排序结果由已有 `processed_file` 提供。它不会重新提取、筛选或生成课程排序文件。

### 静态梯度输出与续跑

提取结果写入 `$SAERL_WORK_DIR/extraction/$SAERL_RUN_NAME/`，包括 `step-*.parquet`、`representation_manifest.json`、`resolved_config.yaml`。提取捕获的是**层输出梯度**，执行 rollout → reward → GRPO advantage → backward，保持模型参数不变，也不更新 scheduler。当前提取使用 GRPO、legacy FSDP 和 Ulysses SP=1，支持尾批。

想使用固定任务名或按 step 续跑：

```bash
SAERL_RUN_NAME=extract-001 bash run/extract_grad_qw25.sh
SAERL_RUN_NAME=extract-001 bash run/extract_grad_qw25.sh \
  static_gradient_extraction.resume_from_step=743
```

新提取使用新任务名，续跑保留原模型、参数、GPU 布局和任务名。续跑通过跳过已处理批次实现，不保证恢复原随机 rollout 序列。

## 单独运行 Gradient-SAE 筛选

这部分用于产生新的筛选结果，可独立运行；`sae_qw25.sh` 使用已经准备好的课程索引文件。以下命令使用前面设置的 `SAERL_WORK_DIR` 和 `SAERL_TRAIN_DATA`。

数据按 verl 的 RL Parquet 格式提供 `prompt`、`data_source`、`reward_model.ground_truth`、`extra_info.index`。其中 `extra_info.index` 是提取/导出使用的唯一整数 sample ID；当前 DeepMath 中它与原数据行号一致。SAE 目录提供 `config.json`（`feature_size`、`hidden_size`、`input_hookpoint`）以及 `.pt` 或 safetensors decoder 权重，支持 `W_dec` / `decoder.weight`。

```bash
export SAERL_RUN_DIR="$SAERL_WORK_DIR/extraction/extract-001"
export SAERL_SELECTION_DIR="$SAERL_WORK_DIR/selection"

saerl-feature-select --run-dir "$SAERL_RUN_DIR" --top-k 256
saerl-select \
  --run-dir "$SAERL_RUN_DIR" \
  --feature-indices "$SAERL_RUN_DIR/feature_selection/energy_feature_indices.npy" \
  --device cpu --cosine-threshold 0.7 --output-dir "$SAERL_SELECTION_DIR"
saerl-export-subset \
  --run-dir "$SAERL_RUN_DIR" --selection-dir "$SAERL_SELECTION_DIR" \
  --dataset "$SAERL_TRAIN_DATA"
```

固定算法如下：

1. SAE decoder 方向 L2 归一化；逐 response token 投影后按绝对值保留 Token-TopK，并保留符号。先对该 rollout 的全部有效 response tokens 取平均，再对同一 prompt 的 rollouts 等权平均，得到有符号 `g_i`。
2. Normalized Energy 按 `E_j = Σ_i g_ij² / Σ_f g_if²` 评分，忽略零信号样本；取全数据得分最高的 K 个 features，所有样本共享这组坐标。这里的 K 与逐 token Top-K 是两个参数。
3. 构造 `abs(g_i[selected_features])`，再做 L2 归一化；绝对值在有符号 token/rollout 平均之后取。
4. 按选定 feature 上的 L2 范数降序、sample ID 升序执行 exact cosine greedy；与已保留样本的最大 cosine 达到阈值时视为冗余，得到 gradient core set。
5. core set 与**所有 `reward_mean == 0.0` 的样本**取并集作为最终训练数据，包括零梯度和被判为冗余的零奖励样本。

导出生成 `core_set.parquet`、`training.parquet` 和 `export_manifest.json`，保留原数据行序、schema 和 metadata。导出负责子集成员选择，课程排序由已有排序流程生成 `processed_file`，保存该子集对应的原数据行号。三个命令的 `--run-dir` 可传多个共享 SAE 表示设置的提取目录；`--top-k` 不超过 SAE feature size。

## 评估

`evaluation/` 保留已有评估工具。`eval.py` 查找 checkpoint 目录下的 `*/merged_hf_model`，通过 Ray/vLLM 批量评估并保存结果 JSON；先将训练 checkpoint 合并为 Hugging Face 模型。`eval.sh` 包含各数据集的原集群调用示例，调整 checkpoint 和输出路径后使用。它不是上述三个训练脚本的自动步骤。

## verl 版本、调用链和验证

官方 verl 固定提交为 `875788ef6e218d0b98cb64fff919d9fef014f986`（`0.7.0.dev0`）。原项目审计基于 `origin/dev` 的 `a7a707729b39dbed8c6126ace675bd0ba60d4e2a`；原始快照 `be4da3e` 的 1175 个文件与该官方提交的 Git blob 哈希一致，因此沿用这个版本。

提取调用链：官方 `run_ppo` → `SAERLTaskRunner` → 提取 Trainer → rollout/reward/GRPO advantage → Worker → Actor backward → 层输出梯度 hook → SAE 投影 → 聚合 → Parquet。该版本缺少完整提取回调及可替换的 Trainer 工厂，因此使用 SAERL 的局部子类适配，保留官方 loss 缩放、entropy/KL、dispatch、sharding 和 offload。

GRPO 和 SAERL 训练使用官方 Trainer/Worker；SAERL 通过官方 `create_rl_sampler` 的 `data.sampler` 扩展接口加载课程 sampler，保持原 `torch.load(processed_file).tolist()` 的取样语义，无需修改安装后的官方源码。

奖励入口为 `src/verl/reward.py:compute_score`：DeepMath、MinervaMath、OlympiadBench 使用 mathruler，AIME 2024 和 AMC-23 使用官方 `math_reward`，其他来源使用官方默认评分；保留 SymPy 的 10 秒超时。

已有 22 项 CPU/官方 verl 回归验证覆盖梯度表示、loss/梯度一致、禁止参数/scheduler 更新、共享 feature energy、exact cosine greedy、全部零奖励并集、导出、尾批及续跑。editable/wheel 安装、五个命令入口、Hydra 配置、包构建及脱离 Git 重建已验证。

本次新增 10 项启动脚本与课程 sampler 测试，验证路径覆盖、参数传递、旧 `.pt` 索引加载、官方 RL 数据集取样顺序和 StatefulDataLoader 断点恢复；三个脚本的真实 Hydra 配置及新 wheel 构建也已通过。真实 GPU/FSDP backward 与分布式 Ray 训练尚未执行。验证工具保存在临时目录，不加入项目。

SAERL includes adaptations of verl (Copyright Bytedance Ltd., SGLang Team, ModelBest Inc.) and a DeepMath scorer (Copyright Huawei Technologies Co., Ltd.), licensed under Apache-2.0. 许可证全文见 `LICENSE`，相关源码保留原有版权声明。
