# SAERL

SAERL 是独立的 Python 项目，提供静态梯度提取、SAE 特征筛选、数据选择、子集导出和 GRPO 训练。源码直接位于 `src/`；官方 verl 作为外部依赖安装，通过 `import verl` 使用。

项目只保留这一条数据选择流程：**Token-TopK response-mean Gradient-SAE → Normalized Energy Top-K 全局共享 features → absolute response-mean pattern → exact cosine greedy core set → 合并全部 zero-reward samples → GRPO 训练**。

verl 固定为原项目对应的 **0.7.0.dev0**，官方提交 **`875788ef6e218d0b98cb64fff919d9fef014f986`**，不会自动升级到新版。版本确认、调用链和局部覆盖的依据见下文“verl 版本与适配说明”。

## 1. 安装与启动检查

以下命令均在仓库根目录执行，要求 Python 3.12。开发时使用 editable 安装，修改 `src/` 后即可生效。Python 包名仍为 `saerl`，由 `pyproject.toml` 将它映射到 `src/`；启动命令和 `import saerl` 保持不变。

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
python -m pip install "torch==2.8.0" -c src/config/constraints-train.txt
python -m pip install packaging ninja
python -m pip install "flash-attn==2.8.1" --no-build-isolation -c src/config/constraints-train.txt
python -m pip install -e '.[train]' -c src/config/constraints-train.txt

# 核验官方 verl 的版本和核心源码指纹
python -m saerl.verl.compat

# 仅打印完整配置，不启动 Ray 或 GPU 任务
saerl-extract --cfg job
saerl-train --cfg job
```

`src/config/constraints-train.txt` 来自原 Python 3.12 / CUDA 12.8 环境的主要版本记录，不是所有平台通用的完整依赖锁。Torch、flash-attn 和 vLLM 需要匹配训练节点的 CUDA 环境；如果使用集群提供的 flash-attn wheel，可用该 wheel 替代上面的编译安装命令。全新 CUDA 环境的依赖安装尚未实测。

使用新虚拟环境，并移除指向旧 verl checkout 的 `PYTHONPATH` 设置。启动入口会核验官方依赖，遇到旧 editable 安装或修改过的 verl 会明确报错。

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

每个 response token 的梯度投影到 SAE decoder 后，按绝对值保留 `gradient_feature_topk` 个 features，并保留其原始符号，其他坐标视为零。先除以该 rollout 的全部有效 response token 数，再对同一 prompt 的 rollouts 等权平均，得到有符号的 response-mean 向量 `g_i`。新提取只保存 response mean、稳定 ID、平均奖励和零信号标记，不再计算或保存 raw gradient、dense SAE、prompt 或 peak 分支。

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

导出通过 `extra_info.index` 匹配样本，保留原数据的行顺序、schema 和 metadata；不把 ID 当作行号。特征筛选、selection 和导出拒绝覆盖已有结果；再次运行时换一个输出目录。

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

Python 模块入口为 `python -m saerl.feature_select`、`python -m saerl.select`、`python -m saerl.export_subset` 和 `python -m saerl.train`。提取与训练共享 `src/train.py`，`saerl-extract` 等价于 `python -m saerl.train --config-name extract`。

## 其他参数

特征和样本选择已经固定为上述方案，没有 `--method` 分发参数。使用各命令的 `--help` 查看剩余参数。

奖励入口只有 `src/verl/reward.py:compute_score`，直接接受官方 custom reward 参数。保留原评分规则：DeepMath、MinervaMath、OlympiadBench 使用 mathruler；AIME 2024 和 AMC-23 使用官方 `math_reward`；其他来源交给官方默认评分。SymPy 的 10 秒超时保留，避免奖励进程卡住。

提取续跑时保持原来的模型、配置、GPU/DP 布局和 `run_name`，在原提取命令后加 `static_gradient_extraction.resume_from_step=N`。已有恢复 step 的 rank 分片会保留，缺失分片会补齐。续跑通过跳过已处理批次实现，不能保证恢复原随机 rollout 序列。

旧版四通道提取结果仍可用于 feature selection、selection 和导出，但不能与新版 response-only 分片混用续跑；需要继续提取时使用新的 `run_name`。

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

该版本没有覆盖完整提取流程的回调，也缺少可替换的 Trainer 工厂，因此部分编排、训练循环、Actor/Worker 更新方法在 SAERL 中局部覆盖。覆盖保留官方 loss 缩放、entropy/KL、dispatch、sharding 和 offload 语义，并保留来源版权说明。核心官方文件指纹存于 `src/verl/upstream.json`，启动时核验。

## 验证状态

选择算法和梯度表示已通过 22 项 CPU/官方 verl 回归用例，包括 response-mean 与旧实现一致、AMP 缩放和 padding 排除、不同 block size 的 greedy 与旧实现及逐样本 exact cosine 参考一致、全数据共享 feature energy、旧/新提取格式输入、全部零奖励并集和 Parquet 保真导出，以及官方 Actor loss/梯度一致、禁止 optimizer/scheduler 更新、尾批和续跑配置。

当前目录组织下另重跑 8 项相关回归并通过，覆盖框架适配、梯度处理模块接入、reward 加载和完整 CPU 选择导出流程。editable 和 wheel 安装后的五个启动命令、配置组合、模块入口和奖励加载也通过验证，官方 `verl` 始终解析到外部安装。wheel 与源码包均包含配置、版本约束和官方指纹资源，无需 MANIFEST.in。测试和安装验证脚本保存在临时目录，不进入项目。

上述验证使用临时 Python 3.12 环境安装官方 verl，并共享原环境的第三方包，不代表全新 CUDA 环境已完成依赖解析。真实 GPU/FSDP backward、两张 GPU 的完整提取训练流程尚未执行；Ray 子进程加载训练模块的检查曾超时，仍待训练节点验证。

当前文件树只保留安装和运行所需内容。可选文档、测试和验证脚本已移除；历史版本仍可从当前 Git 分支的提交记录中查看。

## 目录结构

```text
src/
  train.py             # GRPO 训练与梯度提取的命令入口
  feature_select.py    # 全局 SAE feature 筛选入口
  select.py            # gradient core set 选择入口
  export_subset.py     # 合并零奖励样本并导出训练数据
  verl/                # 官方 verl 适配：启动、Trainer、Worker、Actor、reward、版本校验
  selection/           # 梯度 SAE 表示、Normalized Energy、投影、exact cosine、数据校验
  config/              # Hydra 配置和 constraints-train.txt 版本约束
pyproject.toml         # 包定义、依赖、启动入口和配置资源打包
README.md              # 安装、启动、版本审计和验证状态
LICENSE / NOTICE       # 许可和来源说明
.gitignore             # 忽略缓存、权重、运行数据和构建产物
```

`.git/` 保留当前分支和历史，供版本管理及旧新对比使用。

当前执行环境自动挂载的 `.agents/`、`.aws/`、`.codex/` 是空只读目录，不属于 SAERL，已由 `.gitignore` 排除，也不会进入安装包。

框架接入统一通过 `saerl.verl` 使用，官方依赖仍通过 `import verl` 使用。梯度 hook 和 backward 编排属于框架适配，放在 `verl/`；捕获后的梯度投影、response-mean 聚合和 feature/sample 选择放在 `selection/`。命令入口负责接收参数并串联这些功能。

`src/config/constraints-train.txt` 保留原训练环境的主要版本约束，安装命令会使用它。配置 YAML、版本约束和官方源码指纹均由 `pyproject.toml` 的 package-data 配置纳入 wheel 和源码包，无需单独的 MANIFEST 文件。添加新的 Python 子包时，需要在 `tool.setuptools.packages` 中注册对应的 `saerl.*` 名称。
