# SAERL

SAERL 是独立的 Python 项目，提供静态梯度提取、SAE 特征筛选、数据选择、子集导出和 GRPO 训练。源码直接位于 `src/`；官方 verl 作为外部依赖安装，通过 `import verl` 使用。

项目只保留这一条数据选择流程：**Token-TopK response-mean Gradient-SAE → Normalized Energy Top-K 全局共享 features → absolute response-mean pattern → exact cosine greedy core set → 合并全部 zero-reward samples → GRPO 训练**。

verl 固定为原项目对应的 **0.7.0.dev0**，官方提交 **`875788ef6e218d0b98cb64fff919d9fef014f986`**，不会自动升级到新版。版本确认、调用链和局部覆盖的依据见下文“verl 版本与适配说明”。

## 1. 安装与启动检查

以下命令均在仓库根目录执行，要求 Python 3.12。开发时使用 editable 安装，修改 `src/` 后即可生效。Python 包名仍为 `saerl`，由 `pyproject.toml` 将它映射到 `src/`；启动命令和 `import saerl` 保持不变。

本次调整了命令对应的模块路径，已有环境需重新执行 `python -m pip install -e .` 更新命令入口。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# CPU 特征筛选、selection 和子集导出
python -m pip install -e .

# 检查 CPU 入口
saerl-feature-select --help
saerl-select --help
saerl-export-subset --help
```

如果只处理已有提取结果，安装到这里即可。提取和训练需要 CUDA GPU，以及训练依赖：

```bash
# 先安装 Torch，再安装需要读取 Torch/CUDA 信息的 flash-attn
python -m pip install "torch==2.8.0"
python -m pip install packaging ninja
python -m pip install "flash-attn==2.8.1" --no-build-isolation
python -m pip install -e '.[train]'

# 仅打印完整配置，不启动 Ray 或 GPU 任务
saerl-extract --cfg job
saerl-train --cfg job
```

根目录的 `requirements-train.txt` 是训练依赖清单，保留原环境的版本，包括固定提交的官方 verl。`pyproject.toml` 直接读取它生成 `train` extra，因此上面的 `pip install -e '.[train]'` 会安装清单中的依赖。也可以在预装 Torch 和 flash-attn 后使用常见的安装方式：

```bash
python -m pip install -r requirements-train.txt
python -m pip install -e .
```

这份清单来自原 Python 3.12 / CUDA 12.8 环境，不是所有平台通用的完整依赖锁。Torch、flash-attn 和 vLLM 需要匹配训练节点的 CUDA 环境；如果使用集群提供的 flash-attn wheel，可用该 wheel 替代上面的编译安装命令。全新 CUDA 环境的依赖安装尚未实测。

使用新虚拟环境，并移除指向旧 verl checkout 的 `PYTHONPATH` 设置。官方 verl 的提交由训练依赖清单固定，运行时直接 import，不再扫描安装包源码或核验指纹。

## 2. 准备输入

准备训练和验证 Parquet、Hugging Face 模型目录，以及 SAE decoder 权重目录。路径可以位于项目外。

- 数据使用 verl 的 RL 数据格式，通常包含 `prompt`（消息列表）、`data_source`、`reward_model.ground_truth` 和 `extra_info.index`。
- `extra_info.index` 必须是非空、唯一的整数，作为提取和导出的稳定 sample ID。DeepMath 数据的 `data_source` 使用 `zwhe99/DeepMath-103K`。
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

SAE decoder 的每个 feature 方向使用 L2 归一化。每个 response token 的梯度投影到 decoder 后，按绝对值保留 `gradient_feature_topk` 个 features，并保留其原始符号，其他坐标视为零。先除以该 rollout 的全部有效 response token 数，再对同一 prompt 的 rollouts 等权平均，得到有符号的 response-mean 向量 `g_i`。提取只保存 response mean、稳定 ID、平均奖励和零信号标记。

## 4. 特征筛选、selection 和子集导出

已有提取结果时，直接把 `SAERL_RUN_DIR` 指向对应目录，从这里开始：

```bash
saerl-feature-select \
  --run-dir "$SAERL_RUN_DIR" --top-k 256

saerl-select \
  --run-dir "$SAERL_RUN_DIR" \
  --feature-indices "$SAERL_RUN_DIR/feature_selection/energy_feature_indices.npy" \
  --device cpu --cosine-threshold 0.7 \
  --output-dir "$SAERL_WORK_DIR/selection"

saerl-export-subset \
  --run-dir "$SAERL_RUN_DIR" \
  --selection-dir "$SAERL_WORK_DIR/selection" \
  --dataset "$SAERL_TRAIN_DATA"
```

Normalized Energy 使用 `E_j = Σ_i g_ij² / Σ_f g_if²` 对 features 评分，忽略零信号及 response-mean 范数不大于 `gradient_zero_eps` 的样本；取全数据得分最高的 `--top-k` 个 features，分数相同优先较小 feature ID。输出一个 `energy_feature_indices.npy` 和评分摘要 JSON，所有样本使用同一组 features。这里的 `--top-k` 是全局 feature 数，与提取时逐 token 的 `gradient_feature_topk` 不同。

在共享 features 上构造 `p_i = abs(g_i[selected_features])`，随后做 L2 归一化。**绝对值在 token 和 rollout 的有符号平均之后取**。greedy 按投影后 L2 范数降序、sample ID 升序处理；与已保留样本的最大 exact cosine 达到 `--cosine-threshold` 时判为冗余，否则加入 core set。CPU 和可选 CUDA 均计算完整 cosine，不使用 ANN。零信号或投影后为零的样本不能参与 cosine 选择。

`--top-k` 不应超过 SAE feature size。三个命令的 `--run-dir` 都可接受多个互不重复的提取目录；必须使用相同 SAE 坐标、提取设置和同一组输入 runs。selection 生成 core ID、silent ID 和审计表；silent 文件用于审计，不参与最终并集规则。导出生成：

- `core_set.parquet`：greedy 选出的 gradient core set。
- `training.parquet`：core set 与**全部 `reward_mean == 0.0` 样本**的去重并集，包括 silent、投影后为零，以及有梯度但被判为冗余的零奖励样本。
- `export_manifest.json`：core、零奖励、重叠及最终训练样本数。

`reward_mean` 是同一 prompt 所有 rollout 的奖励均值。对当前 0/1 数学评分，均值为零意味着所有 rollout 都未得分；silent 并不等于 zero reward，奖励为正的 silent 样本不会自动加入训练集。

导出通过 `extra_info.index` 匹配样本，保留原数据的行顺序、schema 和 metadata。特征筛选、selection 和导出可重复运行，会覆盖同名结果。

## 5. 在选中子集上启动 GRPO 训练

```bash
saerl-train \
  algorithm.adv_estimator=grpo \
  data.train_files="$SAERL_WORK_DIR/selection/training.parquet" \
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

每个命令直接对应所属功能模块，也可以通过 Python 模块启动：

| 命令 | Python 模块入口 |
| --- | --- |
| `saerl-extract` | `python -m saerl.selection.extraction` |
| `saerl-feature-select` | `python -m saerl.selection.feature_selection` |
| `saerl-select` | `python -m saerl.selection.sample_selection` |
| `saerl-export-subset` | `python -m saerl.selection.export_subset` |
| `saerl-train` | `python -m saerl.verl.train` |

## 其他参数

特征和样本选择已经固定为上述方案，没有 `--method` 分发参数。使用各命令的 `--help` 查看剩余参数。

奖励入口只有 `src/verl/reward.py:compute_score`，直接接受官方 custom reward 参数。保留原评分规则：DeepMath、MinervaMath、OlympiadBench 使用 mathruler；AIME 2024 和 AMC-23 使用官方 `math_reward`；其他来源交给官方默认评分。SymPy 的 10 秒超时保留，避免奖励进程卡住。

提取续跑时保持原来的模型、配置、GPU/DP 布局和 `run_name`，在原提取命令后加 `static_gradient_extraction.resume_from_step=N`。已有恢复 step 的 rank 分片会保留，缺失分片会补齐。续跑通过跳过已处理批次实现，不能保证恢复原随机 rollout 序列。

旧版四通道提取结果中的 response-mean 列仍可用于选择和导出。继续提取时使用新版创建的 run，避免混合不同表示格式。

## verl 版本与适配说明

官方依赖固定为 `875788ef6e218d0b98cb64fff919d9fef014f986`（版本 `0.7.0.dev0`）。审计源为原项目 `origin/dev` 的 `a7a707729b39dbed8c6126ace675bd0ba60d4e2a`；原始快照 `be4da3e` 的 **1175 个文件**与该官方提交的 Git blob 哈希全部一致，零修改、零本地新增文件。官方版本因此沿用原项目实际使用的提交。

调用链为：官方 `run_ppo` → `SAERLTaskRunner` → Trainer → rollout → reward → GRPO advantage → Worker → Actor backward → 层输出梯度 hook → SAE decoder 投影 → rollout 聚合 → Parquet 分片。普通训练使用官方 Trainer/Worker；提取模式通过 SAERL 子类接入，不修改安装后的官方源码。

| 原项目修改位置 | 独立 SAERL 的接入方式 |
| --- | --- |
| `main_ppo.py` | 注入 `SAERLTaskRunner`，复用官方数据、reward 和资源初始化 |
| `ray_trainer.py` | 提取专用 Trainer，保留尾批、稳定 ID、step 续跑和提取循环 |
| `fsdp_workers.py` | Worker 子类共享官方模型和 optimizer 对象，提取时禁止 scheduler 更新 |
| `dp_actor.py` | Actor 子类复用官方 forward，在 backward 后消费层输出梯度，禁止 optimizer 更新 |
| `static_gradient_sae.py` | 独立梯度表示模块，用等价的 Torch TopK 替代 OpenSAE 包依赖 |
| reward 分发及 `deepmath.py` | 合并为一个模块，通过官方 custom reward 接口加载 |
| 原 trainer 配置 | SAERL YAML 通过 Hydra 搜索路径组合官方配置 |

该版本没有覆盖完整提取流程的回调，也缺少可替换的 Trainer 工厂，因此提取模式使用 SAERL 的 Runner、Trainer、Worker 和 Actor。自定义 Trainer 只执行一遍 rollout → reward → GRPO advantage → backward，不包含普通训练的 critic 更新、验证、checkpoint 和其他 advantage 算法。普通训练委托给官方 `TaskRunner.run`。Actor 保留官方 loss 缩放、entropy/KL，Worker 保留 dispatch、sharding 和 offload 语义，相关源码保留来源版权说明。

SAERL includes adaptations of verl (Copyright Bytedance Ltd., SGLang Team, ModelBest Inc.) and a DeepMath scorer (Copyright Huawei Technologies Co., Ltd.), licensed under Apache-2.0. 许可证全文保留在 `LICENSE`，相关源码保留原有版权声明。

## 验证状态

当前代码已通过 22 项 CPU/官方 verl 回归用例，覆盖 signed Token-TopK response mean 与原实现一致、AMP 缩放和 padding 排除、greedy 与原实现及逐样本 exact cosine 参考一致、跨 run 的共享 feature energy、全部零奖励并集和 Parquet 保真导出。框架回归覆盖官方 Actor loss/梯度一致、禁止 optimizer/scheduler 更新、同步/异步 rollout 的尾批分组、续跑时跳过已完成批次，以及普通训练委托给官方流程。

editable 和 wheel 安装后的五个命令、Hydra 配置组合及外部官方 verl 导入均通过检查，wheel 的 CPU 特征选择、core 筛选和零奖励导出流程也已跑通。wheel/源码包构建及脱离 Git 的源码包重建通过；配置和固定训练依赖均正确打包。

验证脚本保存在临时目录，不进入项目。源码包包含根目录的依赖清单，无需 MANIFEST.in。

上述验证使用临时 Python 3.12 环境安装官方 verl，并共享原环境的第三方包。真实 GPU/FSDP backward 和分布式 Ray 提取训练流程尚未执行。

当前文件树只保留安装和运行所需内容。可选文档、测试和验证脚本已移除；历史版本仍可从当前 Git 分支的提交记录中查看。

## 目录结构

```text
src/
  __init__.py
  selection/                  # SAERL 算法和对应命令入口
    extraction.py             # 接收层输出梯度，聚合 response/rollout mean，写出提取结果
    sae_projection.py         # 加载 SAE decoder，执行梯度投影和逐 token Top-K
    representations.py        # 定义提取结果格式，按批读取稀疏 response-mean 分片
    feature_selection.py      # Normalized Energy Top-K，全数据共享 feature 集合
    response_pattern.py       # 投影到选定 features，构造 absolute response-mean pattern
    sample_selection.py       # exact cosine greedy，保存 core ID 和筛选结果
    export_subset.py          # core 与全部零奖励样本合并，导出最终训练 Parquet
  verl/                       # 与官方 verl 的连接层
    train.py                  # Hydra/Ray 启动入口
    runner.py                 # 普通训练调用官方流程；提取时注册 SAERL Worker/Trainer
    extraction_config.py      # 设置 GRPO、单遍提取、FSDP 和 step 续跑配置
    trainer.py                # 编排 rollout、reward、advantage、backward
    worker.py                 # Ray RPC、FSDP sharding/offload；不更新 scheduler
    actor.py                  # 官方 actor loss、backward、层输出梯度 hook；不更新参数
    reward.py                 # 数据集数学奖励函数
  config/                     # Hydra 运行配置
    base.yaml                 # 共用 reward 入口和梯度提取参数
    extract.yaml              # GRPO 静态提取模式，组合官方配置
    train.yaml                # 普通训练模式，组合官方配置
pyproject.toml         # 包定义、依赖、启动入口和配置资源打包
requirements-train.txt # 训练依赖及固定版本；供 train extra 和 pip -r 共用
README.md              # 安装、启动、版本审计和验证状态
LICENSE                # Apache-2.0 许可证全文；来源说明合并在 README 中
.gitignore             # 忽略缓存、权重、运行数据和构建产物
```

`.git/` 保留当前分支和历史，供版本管理及旧新对比使用。

当前执行环境自动挂载的 `.agents/`、`.aws/`、`.codex/` 是空只读目录，不属于 SAERL，已由 `.gitignore` 排除，也不会进入安装包。

算法链为 `extraction → sae_projection → feature_selection → response_pattern → sample_selection → export_subset`。`representations.py` 专门负责这条链之间的提取数据格式和批量读取。框架接入统一放在 `saerl.verl`，官方依赖仍通过 `import verl` 使用；层输出 hook 属于 Actor 接口适配，捕获后的梯度处理全部在 `selection/`。各功能模块直接提供命令入口，根目录不再保留重复入口文件，也无需通用 `utils` 目录。

各目录的 `__init__.py` 用于声明可安装、可导入的 Python 包；`src/__init__.py` 同时提供 SAERL 版本号。

运行时保留算法需要的共享 SAE 坐标、合法 Top-K/阈值、稳定样本 ID 和 rollout 分组检查。删除逐行 schema/空值扫描、数组哈希、导出后重新读回校验、官方源码指纹扫描，以及 CUDA 内存预判和 OOM 自动回退。`--device auto` 按 CUDA 是否可用选择执行设备；显存不足时可调小 `--similarity-block-size` 或使用 CPU。

配置 YAML 由 `pyproject.toml` 的 package-data 配置纳入 wheel 和源码包。根目录的 `requirements-train.txt` 作为动态依赖定义文件进入源码包；构建 wheel 时，它的依赖及固定版本写入安装元数据。添加新的 Python 子包时，需要在 `tool.setuptools.packages` 中注册对应的 `saerl.*` 名称。
