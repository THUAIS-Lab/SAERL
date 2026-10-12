# SAERL

SAERL 是独立的 Python 项目，当前实现“梯度提取 → Gradient-SAE selection → 子集 RL 训练”。`sae.sh` 读取外部准备的 `processed_file`，训练已经筛选、排序好的样本；curriculum 排序模块尚未合并。

官方 verl 作为外部 Python 依赖安装，固定提交为 `875788ef6e218d0b98cb64fff919d9fef014f986`，对应安装版本号 `0.7.0.dev0`。配置环境时通过 pip 安装该提交，本项目的适配代码通过 `import verl` 加载已安装的包。

## 文件功能

```text
scripts/
  common.sh                 # 共用输入路径、GRPO、rollout、优化器、KL、日志及 checkpoint 设置
  extract.sh                # 单遍静态梯度提取入口，配置 SAE 和提取输出
  select.sh                 # feature 选择、core set 筛选、零奖励合并及子集导出
  grpo.sh                   # 普通 GRPO 训练入口，可训练完整数据集或筛选子集
  sae.sh                    # 按外部 processed_file 指定的原数据行号及顺序训练
src/
  __init__.py               # 包版本号
  selection/
    __init__.py             # Python 包标记
    extraction.py           # 定义提取 schema，聚合 raw gradient 和 SAE response-mean 并写入 Parquet；提取入口
    sae_projection.py       # 读取和归一化 SAE decoder，分块计算 raw 梯度统计及 Token-TopK mean
    feature_selection.py    # 提取结果读取、Normalized Energy Top-K，共享 feature 选择入口
    sample_selection.py     # pattern 构造、core set 筛选、零奖励合并及最终训练子集导出
  verl/
    __init__.py             # Python 包标记
    train.py                # Hydra/Ray 入口及 TaskRunner 扩展；处理提取目录和 step 续跑
    trainer.py              # 单遍提取循环、尾批 padding、日志和 step 续跑
    worker.py               # 提取的 Ray RPC、FSDP sharding/offload 和指标收集；不更新 scheduler
    actor.py                # 静态 GRPO loss/backward 和层输出梯度 hook；不更新模型参数
    reward.py               # 数学奖励评分分发及 SymPy 超时处理
    curriculum_sampler.py   # 读取 processed_file；保留行号顺序及重复项
  config/
    __init__.py             # Python 包标记
    base.yaml               # 官方 PPO defaults、奖励入口和提取参数
    extract.yaml            # 单遍静态提取、legacy FSDP、SP=1、rollout 数和 worker 配置传递
    train.yaml              # 普通 RL 训练配置
    saerl.yaml              # 接入 processed_file sampler 的训练配置
pyproject.toml              # Python 包映射、依赖、配置资源和命令入口
requirements.txt            # 训练依赖版本及官方 verl 固定提交
diff.txt                    # 文件变更清单，运行流程不读取
test.sbatch                 # 本地 Slurm 作业脚本，当前申请一张 GPU 并启动 select.sh
a.py                        # 本地 Parquet 结果对照脚本，运行入口不调用
README.md                   # 文件说明和运行步骤
LICENSE                     # Apache-2.0 许可证
.gitignore                  # 数据、权重、运行产物、缓存和构建产物的忽略规则
```

`src/` 安装后的包名是 `saerl`，由 `pyproject.toml` 映射。`src/verl/` 对应 `saerl.verl`，用于框架接入；`src/selection/` 对应 `saerl.selection`，用于梯度处理和筛选。先安装项目，再运行脚本或 `python -m saerl...`。

`test.sbatch`、`a.py` 和 `diff.txt` 是本地作业或对照文件，不进入 Python 包。`test.sbatch` 中的工作目录、环境激活和启动路径需要按运行机器调整；`.vscode/` 是本地编辑器配置。

selection 按梯度提取、feature selection、sample selection 三个阶段组织。梯度提取分为 `extraction.py`（提取结果聚合与写出）和 `sae_projection.py`（SAE 投影）；`feature_selection.py` 负责 Normalized Energy Top-K；`sample_selection.py` 完成 pattern 构造、exact cosine greedy、全部零奖励样本合并和最终子集导出。提取结果的加载与稀疏读取放在 `feature_selection.py`，供后续步骤复用。

## 安装

提取和 RL 训练使用 Python 3.12，依赖固定为 Torch 2.8.0、vLLM 0.11.0、flash-attn 2.8.1 等版本，沿用原 CUDA 12.8 环境。在项目根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'torch==2.8.0' packaging ninja
python -m pip install 'flash-attn==2.8.1' --no-build-isolation
python -m pip install -e '.[train]'
```

`train` extra 读取根目录的 `requirements.txt`。也可以在安装 Torch 和 flash-attn 后执行 `python -m pip install -r requirements.txt`、`python -m pip install -e .`。使用集群提供的 flash-attn wheel 时，选择与上述 Torch/CUDA 环境匹配的版本。避免将旧 verl checkout 加入 `PYTHONPATH`。

本项目直接加载 OpenSAE 格式的预训练 decoder，投影和 Token-TopK 使用 Torch 实现；运行上述流程无需额外安装 OpenSAE 源码包。

### 安装固定 verl 版本

本项目使用的基线来自本仓库 `main` 的 `be4da3e7711ced82677c4e7174367e7938ea0b54`。已将该快照与[官方固定提交](https://github.com/verl-project/verl/commit/875788ef6e218d0b98cb64fff919d9fef014f986)逐文件比较：1175 个对应文件的 Git blob 哈希一致。`dev` 的 `a7a707729b39dbed8c6126ace675bd0ba60d4e2a` 包含本地训练、奖励和提取改动；当前项目通过 `saerl.verl` 适配官方基线。

`requirements.txt` 使用完整 Git commit 的直接依赖，`pyproject.toml` 的 `train` extra 从该文件读取。因此 `pip install -e '.[train]'` 和 `pip install -r requirements.txt` 使用同一提交，不跟随上游 main/dev 分支更新，也不使用仅固定 `verl==0.7.0.dev0` 的方式；不同源码提交可能使用相同版本号。

如果沿用以前直接开发 verl 的环境，需要先替换旧的 editable verl。以下命令只替换 verl，训练环境的其他依赖仍需按上面的安装步骤准备：

```bash
python -m pip uninstall -y verl
python -m pip install --no-deps \
  'verl @ git+https://github.com/verl-project/verl.git@875788ef6e218d0b98cb64fff919d9fef014f986'
```

`import verl` 只加载当前环境中的包，不会下载或安装依赖；版本固定在 pip 安装阶段完成。安装后如需手动查看提交信息和导入位置，可以运行：

```bash
python -c "import importlib.metadata as m; print(m.distribution('verl').read_text('direct_url.json'))"
python -c "import importlib.util; print(importlib.util.find_spec('verl').origin)"
```

第一条输出中的 `vcs_info.commit_id` 应为上述完整提交，第二条应指向当前环境安装的 verl。旧 checkout 出现在 `PYTHONPATH` 或当前工作目录下时可能覆盖已安装的包；移除旧路径并从本项目根目录启动即可。SAERL 自身可以 editable 安装，verl 使用上述固定 Git 依赖安装。

如果只对已有提取结果进行单卡 GPU selection 和导出，安装基础依赖及 Torch 即可：

```bash
python -m pip install -e . 'torch==2.8.0'
```

分步运行并使用 `--device cpu` 时，可省略 Torch，只安装 `python -m pip install -e .`。

## 输入与公共参数

提前准备模型、训练/验证 Parquet 和预训练 SAE decoder。本项目读取现成的 SAE 权重。

训练数据使用 verl 的 RL Parquet 格式，包括 `prompt`、`data_source`、`reward_model.ground_truth` 和 `extra_info.index`。提取默认从 `index` 或 `extra_info.index` 读取 sample ID；selection 导出通过唯一整数 `extra_info.index` 匹配源数据。当前 DeepMath 中 sample ID 与完整源数据行号一致。

SAE 目录需要 `config.json`，包含 `feature_size`、`hidden_size` 和 `input_hookpoint`，以及一个 `.pt` checkpoint 或 `model.safetensors` / 分片 safetensors 权重。decoder 支持 `W_dec` 或 `decoder.weight`，形状为 `[feature_size, hidden_size]`。hookpoint 需要匹配模型的 transformer layer；提取时也可用 `static_gradient_extraction.hookpoint` 覆盖。

将下面的 `path/to/...` 替换为实际路径。相对路径以项目根目录为基准：

```bash
export SAERL_TRAIN_DATA="path/to/train.parquet"
export SAERL_VAL_DATA="path/to/val.parquet"
export SAERL_MODEL_PATH="path/to/model"
export SAERL_SAE_PATH="path/to/sae"
export SAERL_WORK_DIR="$PWD/runs"
```

| 环境变量 | 用途及默认值 |
| --- | --- |
| `SAERL_TRAIN_DATA` / `SAERL_VAL_DATA` | 训练 / 验证 Parquet 路径；默认 `path/to/train.parquet` / `path/to/val.parquet` |
| `SAERL_MODEL_PATH` | 模型目录；默认 `path/to/model` |
| `SAERL_SAE_PATH` | SAE decoder 目录；默认 `path/to/sae`，仅提取使用 |
| `SAERL_PROCESSED_FILE` | 外部样本行号 Tensor；默认 `path/to/processed.pt`，仅 `sae.sh` 使用 |
| `SAERL_WORK_DIR` | 输出根目录；默认项目根目录下的 `runs/` |
| `SAERL_RUN_NAME` | 提取任务名；默认启动时间戳 |
| `SAERL_EXTRACT_GPUS` / `SAERL_TRAIN_GPUS` | 提取 / 训练 GPU 数；默认 `4` / `4` |
| `SAERL_TRAIN_BATCH_SIZE` / `SAERL_PPO_MINI_BATCH_SIZE` | prompt batch / PPO mini batch；默认 `128` / `64` |
| `SAERL_ROLLOUT_N` / `SAERL_TOTAL_EPOCHS` | 每个 prompt 的 rollout 数 / RL 训练轮数；默认 `8` / `30` |
| `SAERL_TOKEN_TOP_K` | 每个 response token 保留的 SAE features 数；默认 `128` |
| `SAERL_PROJECTION_CHUNK_SIZE` | 提取脚本的投影分块大小；默认 `4096` |
| `PYTHON_BIN` | Python 解释器；默认已激活环境中的 `python` |
| `WANDB_MODE` / `WANDB_DIR` | 日志模式 / 保存目录；默认 `offline` / 项目根目录下的 `wandb/` |

`scripts/common.sh` 使用 GRPO、FSDP、动态 batch、gradient checkpointing，prompt/response 最大长度分别为 `1024` / `3072`，rollout 温度为 `0.6`。actor 学习率为 `1e-6`，actor KL loss 使用 `low_var_kl`、系数 `0.001`；提取脚本覆盖为关闭 actor KL loss。公共脚本同时设置 `algorithm.kl_ctrl.kl_coef=0.001`，该系数在启用 `algorithm.use_kl_in_reward` 后用于 reward KL；当前沿用官方默认关闭状态。

配置按以下顺序合并，同一个参数后面的赋值覆盖前面的赋值：

```text
官方 verl YAML → SAERL YAML → common.sh 的 COMMON_ARGS → 各脚本专用参数 → 命令末尾的 Hydra 参数
```

代码读取未配置的参数时才使用相应默认值。提取启动前还会根据实际 `rollout.n` 设置 `num_rollouts`，组装 `run_dir`，并保存最终配置。`select.sh` 复用公共脚本的数据路径和 Python 解释器，通过环境变量设置筛选参数。上表描述脚本默认值；直接调用 Python 提取入口时，`base.yaml` 的投影分块默认值是 `32`。

## 当前流程快速运行

安装环境并设置上述输入路径后：

```bash
export SAERL_RUN_NAME=extract-001
export SAERL_RUN_DIR="$SAERL_WORK_DIR/extraction/$SAERL_RUN_NAME"
export SAERL_SELECTION_DIR="path/to/selection_output"

bash scripts/extract.sh
bash scripts/select.sh
bash scripts/grpo.sh data.train_files="$SAERL_SELECTION_DIR/subset.parquet"
```

这三步分别提取梯度、导出筛选子集、训练子集。已有完整数据的梯度文件时可以直接从 `select.sh` 开始；已有外部筛选和排序结果时，使用后文的 `sae.sh`。

## 1. 梯度提取

```bash
export SAERL_RUN_NAME=extract-001
bash scripts/extract.sh
```

提取执行单遍 `rollout → reward → GRPO advantage → actor backward`，捕获配置层的**输出梯度**，随即完成 SAE 投影和聚合。模型参数、optimizer 和 scheduler 不进行更新。当前提取使用 legacy FSDP、async rollout 和 Ulysses SP=1；尾批通过 padding 补齐，写出时排除补齐样本。

`extract.yaml` 设置关闭 critic、数据和 actor shuffle，actor 的 `ppo_epochs=1`。提取脚本覆盖公共参数，关闭 actor KL loss 和普通 checkpoint 恢复；reward KL 沿用官方默认关闭状态。`StaticGradientTrainer` 单遍读取数据，最多执行一个 dataloader 长度，也可以用 `trainer.total_training_steps` 限制提取步数。

`extract.yaml` 的保存和验证频率默认是 `-1`，`common.sh` 在命令行中将它们覆盖为 `20`；静态提取循环不调用验证或 checkpoint 保存，因此提取仍只写梯度和配置文件。`src/verl/train.py` 在创建 worker 前解析配置，将实际提取配置传给 worker。

提取脚本默认过滤超过 `data.max_prompt_length` 的 prompts。需要覆盖每一行时，可关闭过滤，并设置足够的 prompt 长度：

```bash
bash scripts/extract.sh data.filter_overlong_prompts=False data.max_prompt_length=4096
```

prompt 长度按实际数据调整；默认 `data.truncation=error`。

输出目录为 `$SAERL_WORK_DIR/extraction/$SAERL_RUN_NAME/`：

| 文件 | 内容 |
| --- | --- |
| `step-XXXXXXXX-rank-XXXXX.parquet` | 每个 prompt 一行的 `raw_grad_vals`、稀疏 SAE 表示、sample ID、平均奖励和零信号标记 |
| `representation_manifest.json` | SAE、hookpoint、Token-TopK、raw gradient 布局和聚合方式 |
| `resolved_config.yaml` | 实际提取配置 |

`raw_grad_vals` 保存 float32、`4H` 的原始梯度统计，`H` 为目标层 hidden size：

```text
[prompt_mean, prompt_signed_peak, response_mean, response_signed_peak]
```

每个 rollout 分别在有效 prompt / response tokens 上计算有符号均值，以及每个 hidden 维度绝对值最大的梯度并保留符号；绝对值相同时优先正值。然后对同一 prompt 的 rollouts 按通道等权平均。`raw_grad_vals[2 * H:3 * H]` 即 raw response-mean。raw gradient 使用捕获的层输出梯度，在恢复 GradScaler 缩放后直接聚合，不经过 SAE 投影或归一化。它与 SAE 表示保存在同一 Parquet 中，当前 selection 仍读取 SAE response-mean 字段。

提取分别处理有效 prompt 和 response tokens。每个 chunk 先用 float32 梯度计算 raw 统计，再转为 decoder 的 bfloat16 dtype 做 SAE 投影，并以 float32 累加。写出的 SAE 表示只保存 response 的 Token-TopK mean 和 feature IDs，不计算或保存 SAE TopK peak。

使用新的任务名启动新提取。按 step 续跑时保留原模型、提取参数、GPU 布局和任务名：

```bash
bash scripts/extract.sh static_gradient_extraction.resume_from_step=743
```

将 `743` 替换为第一个未完整写出的 step。续跑跳过此前的 batch，从该 step 开始处理；配置另存为 `resolved_config.resume-step-00000743.yaml`。起始 step 已存在的 rank 分片会保留，其余缺失分片继续写入。续跑通过跳过批次实现，不恢复原随机 rollout 状态。

## 2. Gradient-SAE selection

复用完整数据集的提取结果，一次执行共享 feature 选择、core set 筛选、全部零奖励样本合并和子集导出：

```bash
export SAERL_RUN_DIR="$SAERL_WORK_DIR/extraction/$SAERL_RUN_NAME"
export SAERL_SELECTION_DIR="path/to/selection_output"
bash scripts/select.sh
```

如果梯度已提前准备好，直接指定输入和输出路径即可：

```bash
SAERL_TRAIN_DATA="path/to/train.parquet" \
SAERL_RUN_DIR="path/to/gradient_run" \
SAERL_SELECTION_DIR="path/to/selection_output" \
bash scripts/select.sh
```

梯度目录需要包含 `representation_manifest.json` 和 `step-*.parquet`；`SAERL_TRAIN_DATA` 传用于提取的完整源数据。输出目录中最终的 `subset.parquet` 是 core set 与全部零奖励样本的并集。

| selection 环境变量 | 用途及脚本默认值 |
| --- | --- |
| `SAERL_RUN_DIR` | 已提取的梯度目录；默认 `path/to/gradient_run` |
| `SAERL_SELECTION_DIR` | 所有筛选和导出结果的保存目录；默认 `path/to/selection_output` |
| `SAERL_FEATURE_TOP_K` | 全数据共享的 feature 数；默认 `256` |
| `SAERL_COSINE_THRESHOLD` | cosine 冗余阈值；默认 `0.7` |
| `CUDA_VISIBLE_DEVICES` | 用于 cosine 计算的可见 GPU；仅使用第一张，Slurm 作业沿用调度器分配 |
| `SAERL_SELECTION_READ_BATCH_SIZE` | 稀疏梯度读取 batch 大小；默认 `4` |
| `SAERL_SIMILARITY_BLOCK_SIZE` | cosine 计算分块大小；默认 `64` |

例如覆盖 feature 数并选择一张 GPU：

```bash
SAERL_FEATURE_TOP_K=128 CUDA_VISIBLE_DEVICES=0 bash scripts/select.sh
```

需要汇总多个梯度目录时，将目录作为脚本位置参数传入；它们共同拟合一组 features，并在同一个空间中筛选样本：

```bash
bash scripts/select.sh path/to/gradient_run_a path/to/gradient_run_b
```

也可以分步运行同一流程：

```bash
saerl-feature-select \
  --run-dir "$SAERL_RUN_DIR" \
  --top-k 256 \
  --output "$SAERL_SELECTION_DIR/feature_indices.npy"

CUDA_VISIBLE_DEVICES=0 saerl-select \
  --run-dir "$SAERL_RUN_DIR" \
  --feature-indices "$SAERL_SELECTION_DIR/feature_indices.npy" \
  --dataset "$SAERL_TRAIN_DATA" \
  --device cuda \
  --cosine-threshold 0.7 \
  --output "$SAERL_SELECTION_DIR/subset.parquet"
```

selection 使用以下算法：

1. 将 SAE decoder 方向 L2 归一化。逐 response token 投影，按绝对值保留 Token-TopK，保留原符号；TopK 中绝对值不超过 `gradient_zero_eps`（默认 `1e-12`）的值也置零，其余坐标记为零。对全部有效 response tokens 取平均，再对同一 prompt 的 rollouts 等权平均，得到有符号表示 `g_i`。
2. 按 `E_j = Σ_i (g_ij² / Σ_f g_if²)` 计算 Normalized Energy，忽略零信号样本。取全数据得分最高的 K 个 features，所有样本共享这组坐标；同分时优先较小 feature ID。最终 feature IDs 按 ID 升序保存。
3. 在选定 features 上构造 `abs(g_i[selected_features])`，再做 L2 归一化。绝对值在有符号 token/rollout 均值之后取。
4. 按选定 features 上的 L2 范数降序、sample ID 升序执行 exact cosine greedy。与已保留样本的最大 cosine 达到阈值时判为冗余，其余保留为 gradient core set。
5. 将 core set 与提取结果中**所有 `reward_mean == 0.0` 的样本**取并集，作为最终训练子集，包括零梯度和被判为冗余的零奖励样本。

`SAERL_TOKEN_TOP_K` 是每个 token 的保留数量，超过 decoder feature size 时自动截到 feature size；`SAERL_FEATURE_TOP_K` 是全数据共享 feature 数，要求在 `1` 到 feature size 之间。这两个参数分别设置。

Python feature 选择入口默认 `--top-k=256`。Python 样本筛选入口默认 `--cosine-threshold=0.98`、`--read-batch-size=4`、`--similarity-block-size=64`、`--device=cuda`；`select.sh` 显式将 cosine 阈值设为 `0.7`。exact cosine 计算使用可见 GPU 列表中的第一张卡 `cuda:0`，需要安装 Torch；feature 统计、Parquet 读取和导出在 CPU 上执行。单独调用 Python 样本筛选入口时，可传 `--device cpu` 使用基础依赖进行 CPU 筛选。

两个 Python 命令的 `--run-dir` 都支持多个目录，要求使用相同 SAE、hookpoint 和投影设置。`saerl-select` 必须传入完整源数据集 `--dataset`，直接保存合并后的训练子集。`select.sh` 将所有结果写入 `SAERL_SELECTION_DIR`；单独调用 Python 入口且省略 `--output` 时，feature IDs 写入第一个梯度目录的 `selection/feature_indices.npy`，训练子集写入 `selection/training.parquet`。

| 输出位置 | 文件及内容 |
| --- | --- |
| `$SAERL_SELECTION_DIR/` | `feature_indices.npy`：全数据共享的 feature IDs |
| `$SAERL_SELECTION_DIR/` | `subset.parquet`：gradient core set 与全部零奖励样本合并后的训练子集 |

筛选阶段只保存上述两个文件，feature 数、core set 数、零奖励样本数及最终子集大小输出到终端。

导出保留源 Parquet 的行序、schema 和 metadata，不按 greedy 选中顺序重排行。`subset.parquet` 包含最终训练子集；脚本不会额外写 sample indices、pattern、energy scores 或筛选 manifest。

### 使用 Slurm 提交 selection

当前 `test.sbatch` 申请一张 GPU，并调用 `scripts/select.sh`。按运行机器调整其中的路径与环境后，可以在提交时传入数据和输出目录：

```bash
sbatch --export=ALL,SAERL_TRAIN_DATA=path/to/train.parquet,SAERL_RUN_DIR=path/to/gradient_run,SAERL_SELECTION_DIR=path/to/selection_output test.sbatch
```

selection 使用作业分配的第一张可见 GPU；作业内沿用 Slurm 设置的 `CUDA_VISIBLE_DEVICES` 即可。

## 3. RL 训练

### 普通 GRPO

完整训练集上的 GRPO：

```bash
bash scripts/grpo.sh
```

对 selection 导出的子集进行 GRPO：

```bash
bash scripts/grpo.sh data.train_files="$SAERL_SELECTION_DIR/subset.parquet"
```

第二种方式完成当前已实现的“梯度提取 → selection → 子集 RL 训练”流程。

### 按外部 processed_file 训练

`processed_file` 需要准备为**一维整数 Torch Tensor，保存样本在完整训练数据集中的行号，排列顺序就是训练取样顺序**。索引范围为 `0` 到实际完整数据集行数减一。sampler 通过 `torch.load(...).tolist()` 读取列表，保留原顺序及重复项，每个 epoch 复用该列表。

```bash
export SAERL_PROCESSED_FILE="path/to/processed.pt"
bash scripts/sae.sh
```

这里的 `SAERL_TRAIN_DATA` 必须仍指向生成索引时的**完整源 Parquet，且行序一致**；通过索引列表限定要训练的子集。`sae.sh` 覆盖公共参数，将长 prompt 过滤设为 `False`，保留原数据行号。`saerl.yaml` 设置 `train_max_samples=-1`、`shuffle=false`、`dataloader_num_workers=0`，接入读取索引的 `SAERLSampler`。prompt 最大长度应能容纳选中样本。

`sae.sh` 只训练文件指定的样本。完整 SAERL 所需的 curriculum 排序模块尚未合并；已有外部排序结果时，可通过这个入口进行训练。

普通 GRPO 和 `sae.sh` 使用官方训练 DataLoader 的 `drop_last=True`：每个 epoch 不足一个 prompt batch 的尾部样本会被丢弃。`sae.sh` 的列表长度需要至少达到一个 batch；希望每轮遍历全部列表时，应让列表长度能被实际 batch size 整除（通常为 `data.train_batch_size`，配置 `data.gen_batch_size` 时使用后者），并满足 verl 的 GPU 数和 PPO mini batch 约束。静态提取使用 `drop_last=False` 并补齐尾批。

### checkpoint、验证和参数覆盖

GRPO 与 `sae.sh` 默认训练 `30` 个 epoch，训练前验证，并且每 `20` step 保存和验证。checkpoint 分别写入 `$SAERL_WORK_DIR/checkpoints/grpo` 和 `$SAERL_WORK_DIR/checkpoints/saerl`；`resume_mode=auto` 自动从已有 checkpoint 恢复。启动新实验时设置新的 `SAERL_WORK_DIR` 或 `trainer.default_local_dir`。

提取循环不执行验证或 checkpoint 保存，关闭普通训练恢复，续跑使用专门的 `static_gradient_extraction.resume_from_step`。默认 WandB 离线记录在项目 `wandb/`，可通过 `WANDB_MODE=online` 覆盖。

提取和 RL 训练入口可在命令末尾追加 Hydra 参数；筛选脚本通过上面的环境变量配置：

```bash
bash scripts/grpo.sh trainer.total_epochs=1 'trainer.logger=[console]'
bash scripts/extract.sh static_gradient_extraction.gradient_feature_topk=64
bash scripts/sae.sh \
  data.saerl.processed_file="path/to/other_processed.pt"
```

只查看合成配置、不启动 Ray/GPU：

```bash
bash scripts/extract.sh --cfg job
bash scripts/grpo.sh --cfg job
bash scripts/sae.sh --cfg job
```

安装后有四个 Python 命令入口：`saerl-extract`、`saerl-train`、`saerl-feature-select`、`saerl-select`。前两个直接使用 Hydra 配置，脚本中的 `COMMON_ARGS` 由 `.sh` 入口传入。更新命令入口后，已安装环境可在项目根目录执行 `python -m pip install -e . --no-deps --no-build-isolation` 刷新安装信息。

## verl 接入与许可证

普通 RL 训练使用官方 TaskRunner、Trainer 和 Worker；`sae.sh` 通过官方 `data.sampler` 扩展接口加载行号列表。`SAERLSampler` 实现初始化、索引迭代和长度计算，并继承官方要求的 `AbstractSampler`；它只按现有列表取样，不动态更新课程排序。

提取调用链为官方 `run_ppo` → `SAERLTaskRunner` → `StaticGradientTrainer` → `StaticGradientWorker` → `StaticGradientActor` → 层输出梯度 hook → SAE 投影及聚合 → Parquet。

奖励入口为 `saerl.verl.reward.compute_score`：DeepMath、MinervaMath、OlympiadBench 使用 mathruler，AIME 2024、AMC-23 使用官方 `math_reward`，其他数据来源使用官方默认评分；SymPy 等价性判断保留 10 秒超时。

本次核对通过了 CPU selection 小样本验证（feature energy、exact cosine greedy、全部零奖励并集、导出行序和 metadata）、wheel 构建及固定官方 YAML 的配置合成。当前环境完整训练入口的导入检查在 flash-attn 初始化阶段超时；未执行真实 GPU/FSDP backward、分布式 Ray 训练或全新 CUDA 环境安装。

SAERL includes adaptations of verl (Copyright Bytedance Ltd., SGLang Team, ModelBest Inc.) and a DeepMath scorer (Copyright Huawei Technologies Co., Ltd.), licensed under Apache-2.0. 许可证全文见 `LICENSE`，相关源码保留原有版权声明。
