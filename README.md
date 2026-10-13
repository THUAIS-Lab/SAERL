# SAERL

SAERL 是独立的 Python 项目，当前实现“梯度提取 → Gradient-SAE selection → 子集 RL 训练”。`sae.sh` 读取外部准备的 `processed_file`，训练已经筛选、排序好的样本；curriculum 排序模块尚未合并。

官方 verl 作为外部 Python 依赖安装，固定提交为 `875788ef6e218d0b98cb64fff919d9fef014f986`，对应安装版本号 `0.7.0.dev0`。配置环境时通过 pip 安装该提交，本项目的适配代码通过 `import verl` 加载已安装的包。

## 文件功能

```text
scripts/
  install_train.sh          # 调用固定提交的官方 verl 脚本，配置 FSDP + vLLM 训练环境
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
a.py                        # 本地 Parquet 结果对照脚本，运行入口不调用
README.md                   # 文件说明和运行步骤
LICENSE                     # Apache-2.0 许可证
.gitignore                  # 数据、权重、运行产物、缓存和构建产物的忽略规则
```

`src/` 安装后的包名是 `saerl`，由 `pyproject.toml` 映射。`src/verl/` 对应 `saerl.verl`，用于框架接入；`src/selection/` 对应 `saerl.selection`，用于梯度处理和筛选。完成环境安装后，通过 `scripts/` 下的脚本运行。

`a.py` 和 `diff.txt` 是本地对照文件，不进入 Python 包；`.vscode/` 是本地编辑器配置。

selection 按梯度提取、feature selection、sample selection 三个阶段组织。梯度提取分为 `extraction.py`（提取结果聚合与写出）和 `sae_projection.py`（SAE 投影）；`feature_selection.py` 负责 Normalized Energy Top-K；`sample_selection.py` 完成 pattern 构造、exact cosine greedy、全部零奖励样本合并和最终子集导出。提取结果的加载与稀疏读取放在 `feature_selection.py`，供后续步骤复用。

## 环境安装

使用 Conda 创建 Python 3.12 环境，然后通过 `scripts/install_train.sh` 调用固定 verl 提交的[官方依赖安装脚本](https://github.com/verl-project/verl/blob/875788ef6e218d0b98cb64fff919d9fef014f986/scripts/install_vllm_sglang_mcore.sh)，安装 FSDP + vLLM 训练环境及本项目。适用于 Linux x86_64 / NVIDIA CUDA 12.8；需要可用的 `git`、`curl`、`wget`，并能访问 PyTorch wheel 源与 GitHub。

在本项目根目录执行以下三条命令：

```bash
conda create -n saerl python=3.12 pip -y
conda activate saerl
bash scripts/install_train.sh
```

脚本依次安装 Torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0 的 CUDA 12.8 wheel，以 `USE_MEGATRON=0 USE_SGLANG=0` 运行官方脚本，再安装固定提交的 verl 和本项目，最后执行 `pip check`。从安装开始即用 `requirements.txt` 约束依赖版本，并提前安装 NumPy 1.26.4 和旧开发环境使用的 filelock 3.25.2，保持原评分超时实现的依赖一致；完成后即可运行提取、筛选和训练脚本。

`requirements.txt` 中的完整 Git commit 固定了 verl 版本，`pyproject.toml` 的 `train` extra 从该文件读取依赖；安装入口下载的官方脚本也使用同一提交。该提交对应本仓库 main 的源码基线，不跟随上游分支更新。上述环境安装已包含 verl 的安装，完成后 `import verl` 直接加载该包，无需再单独安装 verl。

官方脚本写死了 `MAX_JOBS=32` 和 flash-attn 的 FALSE ABI；安装入口将前者改为默认 `2`（可通过 `MAX_JOBS` 覆盖），并按实际 Torch ABI 选择官方 flash-attn wheel。其余依赖安装步骤沿用原脚本。

安装入口会输出 `[1/5]` 到 `[5/5]` 的步骤日志。`[3/5]` 是导入 Torch 并读取 ABI，此时还未开始安装 vLLM。

以后开启新终端时，激活同一环境：

```bash
conda activate saerl
```

本项目直接加载 OpenSAE 格式的预训练 decoder，投影和 Token-TopK 使用 Torch 实现，无需额外安装 OpenSAE 源码包。

## 输入与公共参数

提前准备模型、训练/验证 Parquet 和预训练 SAE decoder。本项目读取现成的 SAE 权重。

训练数据使用 verl 的 RL Parquet 格式，包括 `prompt`、`data_source`、`reward_model.ground_truth` 和 `extra_info.index`。提取默认从 `index` 或 `extra_info.index` 读取 sample ID；selection 导出通过唯一整数 `extra_info.index` 匹配源数据。当前 DeepMath 中 sample ID 与完整源数据行号一致。

SAE 目录需要 `config.json`，包含 `feature_size`、`hidden_size` 和 `input_hookpoint`，以及一个 `.pt` checkpoint 或 `model.safetensors` / 分片 safetensors 权重。decoder 支持 `W_dec` 或 `decoder.weight`，形状为 `[feature_size, hidden_size]`。hookpoint 需要匹配模型的 transformer layer；提取时也可用 `static_gradient_extraction.hookpoint` 覆盖。

运行前直接修改脚本中的 `path/to/...` 路径和参数默认值。相对路径以项目根目录为基准：

| 修改的脚本 | 需要配置的内容 |
| --- | --- |
| `scripts/common.sh` | 完整训练/验证 Parquet、模型路径、输出根目录，以及公共 batch、rollout、优化器和日志参数 |
| `scripts/extract.sh` | SAE 权重路径、提取任务名、GPU 数、Token-TopK 和投影分块大小 |
| `scripts/select.sh` | 提取结果目录、筛选输出目录、共享 feature 数、cosine 阈值和计算分块大小 |
| `scripts/grpo.sh` | 训练 GPU 数、epoch 数、checkpoint 目录；训练子集时在 `COMMON_ARGS` 后增加 `data.train_files` 参数，指向筛选导出的 `subset.parquet` |
| `scripts/sae.sh` | `processed_file` 路径、训练 GPU 数、epoch 数和 checkpoint 目录 |

`common.sh` 中的训练数据路径保持为完整源 Parquet，供提取、selection 导出和按原行号取样使用。普通 GRPO 可以在 `grpo.sh` 中通过 `data.train_files` 覆盖为筛选子集。

`scripts/common.sh` 使用 GRPO、FSDP、动态 batch、gradient checkpointing，prompt/response 最大长度分别为 `1024` / `3072`，rollout 温度为 `0.6`。actor 学习率为 `1e-6`，actor KL loss 使用 `low_var_kl`、系数 `0.001`；提取脚本覆盖为关闭 actor KL loss。公共脚本同时设置 `algorithm.kl_ctrl.kl_coef=0.001`，该系数在启用 `algorithm.use_kl_in_reward` 后用于 reward KL；当前沿用官方默认关闭状态。

配置按以下顺序合并，同一个参数后面的赋值覆盖前面的赋值：

```text
官方 verl YAML → SAERL YAML → common.sh 的 COMMON_ARGS → 各脚本专用参数 → 命令末尾的 Hydra 参数
```

代码读取未配置的参数时才使用相应默认值。提取启动前还会根据实际 `rollout.n` 设置 `num_rollouts`，组装 `run_dir`，并保存最终配置。`select.sh` 复用公共脚本的数据路径和 Python 解释器，筛选参数直接在该脚本中修改。提取脚本的投影分块默认值是 `4096`，会覆盖 `base.yaml` 中的 `32`。

## 当前流程快速运行

修改好上述脚本后，在项目根目录依次执行：

```bash
bash scripts/extract.sh
bash scripts/select.sh
bash scripts/grpo.sh
```

这三步分别提取梯度、导出筛选子集、训练子集。`select.sh` 的提取结果目录应对应 `extract.sh` 的输出任务；`grpo.sh` 的 `data.train_files` 应指向 `select.sh` 导出的 `subset.parquet`。已有提取结果时可以从 `select.sh` 开始；已有外部筛选和排序结果时使用 `sae.sh`。

## 1. 梯度提取

修改 `extract.sh` 中的 SAE 路径和任务名后执行：

```bash
bash scripts/extract.sh
```

提取执行单遍 `rollout → reward → GRPO advantage → actor backward`，捕获配置层的**输出梯度**，随即完成 SAE 投影和聚合。模型参数、optimizer 和 scheduler 不进行更新。当前提取使用 legacy FSDP、async rollout 和 Ulysses SP=1；尾批通过 padding 补齐，写出时排除补齐样本。

`extract.yaml` 设置关闭 critic、数据和 actor shuffle，actor 的 `ppo_epochs=1`。提取脚本覆盖公共参数，关闭 actor KL loss 和普通 checkpoint 恢复；reward KL 沿用官方默认关闭状态。`StaticGradientTrainer` 单遍读取数据，最多执行一个 dataloader 长度，也可以用 `trainer.total_training_steps` 限制提取步数。

`extract.yaml` 的保存和验证频率默认是 `-1`，`common.sh` 在命令行中将它们覆盖为 `20`；静态提取循环不调用验证或 checkpoint 保存，因此提取仍只写梯度和配置文件。`src/verl/train.py` 在创建 worker 前解析配置，将实际提取配置传给 worker。

提取脚本默认过滤超过 `data.max_prompt_length` 的 prompts。需要覆盖每一行时，在 `extract.sh` 的专用参数中将 `data.filter_overlong_prompts` 设为 `False`，并根据数据调整 `data.max_prompt_length`；默认 `data.truncation=error`。

输出目录由 `common.sh` 的输出根目录和 `extract.sh` 的任务名决定，默认位于 `runs/extraction/<任务名>/`：

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

使用新的任务名启动新提取。按 step 续跑时，在 `extract.sh` 的提取参数中增加 `static_gradient_extraction.resume_from_step=743`，再运行同一脚本；将 `743` 替换为第一个未完整写出的 step，并保留原模型、提取参数、GPU 布局和任务名。

续跑跳过此前的 batch，从该 step 开始处理；配置另存为 `resolved_config.resume-step-00000743.yaml`。起始 step 已存在的 rank 分片会保留，其余缺失分片继续写入。续跑通过跳过批次实现，不恢复原随机 rollout 状态。

## 2. Gradient-SAE selection

修改 `select.sh` 中的提取结果目录和筛选输出目录后执行：

```bash
bash scripts/select.sh
```

该脚本一次完成共享 feature 选择、core set 筛选、全部零奖励样本合并和子集导出。提取结果目录需要包含 `representation_manifest.json` 和 `step-*.parquet`；`common.sh` 中的训练数据路径应指向用于提取的完整源 Parquet。

`select.sh` 默认选择 `256` 个共享 features，cosine 阈值为 `0.7`，稀疏梯度读取 batch 大小为 `4`，cosine 计算分块大小为 `64`；需要调整时直接修改该脚本。汇总多个提取目录时，在脚本的 `RUN_DIRS` 数组中填写这些目录，它们共同拟合一组 features 并在同一空间中筛选；各目录需要使用相同 SAE、hookpoint 和投影设置。

selection 使用以下算法：

1. 将 SAE decoder 方向 L2 归一化。逐 response token 投影，按绝对值保留 Token-TopK，保留原符号；TopK 中绝对值不超过 `gradient_zero_eps`（默认 `1e-12`）的值也置零，其余坐标记为零。对全部有效 response tokens 取平均，再对同一 prompt 的 rollouts 等权平均，得到有符号表示 `g_i`。
2. 按 `E_j = Σ_i (g_ij² / Σ_f g_if²)` 计算 Normalized Energy，忽略零信号样本。取全数据得分最高的 K 个 features，所有样本共享这组坐标；同分时优先较小 feature ID。最终 feature IDs 按 ID 升序保存。
3. 在选定 features 上构造 `abs(g_i[selected_features])`，再做 L2 归一化。绝对值在有符号 token/rollout 均值之后取。
4. 按选定 features 上的 L2 范数降序、sample ID 升序执行 exact cosine greedy。与已保留样本的最大 cosine 达到阈值时判为冗余，其余保留为 gradient core set。
5. 将 core set 与提取结果中**所有 `reward_mean == 0.0` 的样本**取并集，作为最终训练子集，包括零梯度和被判为冗余的零奖励样本。

`extract.sh` 中的 Token-TopK 是每个 token 的保留数量，超过 decoder feature size 时自动截到 feature size；`select.sh` 中的共享 feature 数要求在 `1` 到 feature size 之间。这两个参数分别在对应脚本中修改。

exact cosine 计算使用可见 GPU 中的第一张卡；feature 统计、Parquet 读取和导出在 CPU 上执行。

| 输出位置 | 文件及内容 |
| --- | --- |
| `select.sh` 配置的输出目录 | `feature_indices.npy`：全数据共享的 feature IDs |
| `select.sh` 配置的输出目录 | `subset.parquet`：gradient core set 与全部零奖励样本合并后的训练子集 |

筛选阶段只保存上述两个文件，feature 数、core set 数、零奖励样本数及最终子集大小输出到终端。

导出保留源 Parquet 的行序、schema 和 metadata，不按 greedy 选中顺序重排行。`subset.parquet` 包含最终训练子集；脚本不会额外写 sample indices、pattern、energy scores 或筛选 manifest。

## 3. RL 训练

### 普通 GRPO

在 `grpo.sh` 中配置训练数据和训练参数后执行：

```bash
bash scripts/grpo.sh
```

默认使用 `common.sh` 配置的完整训练集。训练 selection 导出的子集时，在 `grpo.sh` 的训练命令中、`COMMON_ARGS` 后增加 `data.train_files` 参数，值设为筛选输出目录中的 `subset.parquet`，再运行同一脚本。

### 按外部 processed_file 训练

`processed_file` 需要准备为**一维整数 Torch Tensor，保存样本在完整训练数据集中的行号，排列顺序就是训练取样顺序**。索引范围为 `0` 到实际完整数据集行数减一。sampler 通过 `torch.load(...).tolist()` 读取列表，保留原顺序及重复项，每个 epoch 复用该列表。

修改 `sae.sh` 中的 `processed_file` 路径后执行：

```bash
bash scripts/sae.sh
```

这里 `common.sh` 中的训练数据路径必须仍指向生成索引时的**完整源 Parquet，且行序一致**；通过索引列表限定要训练的子集。`sae.sh` 覆盖公共参数，将长 prompt 过滤设为 `False`，保留原数据行号。`saerl.yaml` 设置 `train_max_samples=-1`、`shuffle=false`、`dataloader_num_workers=0`，接入读取索引的 `SAERLSampler`。prompt 最大长度应能容纳选中样本。

`sae.sh` 只训练文件指定的样本。完整 SAERL 所需的 curriculum 排序模块尚未合并；已有外部排序结果时，可通过这个入口进行训练。

普通 GRPO 和 `sae.sh` 使用官方训练 DataLoader 的 `drop_last=True`：每个 epoch 不足一个 prompt batch 的尾部样本会被丢弃。`sae.sh` 的列表长度需要至少达到一个 batch；希望每轮遍历全部列表时，应让列表长度能被实际 batch size 整除（通常为 `data.train_batch_size`，配置 `data.gen_batch_size` 时使用后者），并满足 verl 的 GPU 数和 PPO mini batch 约束。静态提取使用 `drop_last=False` 并补齐尾批。

### checkpoint 和验证

GRPO 与 `sae.sh` 默认训练 `30` 个 epoch，训练前验证，并且每 `20` step 保存和验证。checkpoint 分别写入输出根目录下的 `checkpoints/grpo` 和 `checkpoints/saerl`；`resume_mode=auto` 自动从已有 checkpoint 恢复。启动新实验时，修改 `common.sh` 的输出根目录或对应训练脚本中的 `trainer.default_local_dir`。

提取循环不执行验证或 checkpoint 保存，关闭普通训练恢复，续跑使用专门的 `static_gradient_extraction.resume_from_step`。默认 WandB 离线记录在项目 `wandb/`，需要在线记录时，在 `common.sh` 中将 WandB 模式改为 `online`。

训练轮数、GPU 数和各流程专用参数在对应 `.sh` 脚本中修改；公共参数在 `common.sh` 中修改。配置合并顺序见前文，脚本传入的参数会覆盖 YAML 中的同名设置。

## verl 接入与许可证

普通 RL 训练使用官方 TaskRunner、Trainer 和 Worker；`sae.sh` 通过官方 `data.sampler` 扩展接口加载行号列表。`SAERLSampler` 实现初始化、索引迭代和长度计算，并继承官方要求的 `AbstractSampler`；它只按现有列表取样，不动态更新课程排序。

提取调用链为官方 `run_ppo` → `SAERLTaskRunner` → `StaticGradientTrainer` → `StaticGradientWorker` → `StaticGradientActor` → 层输出梯度 hook → SAE 投影及聚合 → Parquet。

奖励入口为 `saerl.verl.reward.compute_score`：DeepMath、MinervaMath、OlympiadBench 使用 mathruler，AIME 2024、AMC-23 使用官方 `math_reward`，其他数据来源使用官方默认评分。SymPy 等价性判断沿用 dev 中的 verl 子进程超时实现，保留 10 秒超时，超时评分为 0。

已通过 CPU selection 小样本验证（feature energy、exact cosine greedy、全部零奖励并集、导出行序和 metadata）、wheel 构建及固定官方 YAML 的配置合成。完整 GPU 梯度提取及分布式 RL 训练仍需实际运行验证。

SAERL includes adaptations of verl (Copyright Bytedance Ltd., SGLang Team, ModelBest Inc.) and a DeepMath scorer (Copyright Huawei Technologies Co., Ltd.), licensed under Apache-2.0. 许可证全文见 `LICENSE`，相关源码保留原有版权声明。
