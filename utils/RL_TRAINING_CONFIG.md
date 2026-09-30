# DeepMath：GRPO / DAPO RL 训练配置

依据：[GRPO baseline](grpo_baseline/qwen1_5b/deepmath_grpo_1k.sh)、[SAE 排序 GRPO](sae/qwen1_5b/sorted_deepmath128_qw25b.sh)、[DAPO](recipe/dapo/deepmath128_qwen2_5.sh)。以下是脚本的实际设置；注明“默认”的值来自仓库配置。

## GRPO 配置

| 超参数 | 配置 | 说明 |
| --- | --- | --- |
| 基座模型 / 优势估计 | Qwen2.5-Math-1.5B / `grpo` | — |
| 训练数据 / batch | `sorted_deepmath/train.parquet` / 128 prompts | 每题采样 8 条，约 1024 条轨迹/批 |
| SAE 排序变体 | `data.shuffle=False`，启用 `data.saerl` | 使用预处理排序文件；baseline 默认 `shuffle=True` |
| Prompt / response 上限 | 1024 / 3072 tokens | `data.truncation=error` |
| 超长 prompt 过滤 | baseline 开启；SAE 排序变体未开启 | SAE 变体未显式设种子 |
| 学习率 / mini-batch | `1e-6` / 64 | `ppo_epochs=1`（默认） |
| PPO 裁剪 | `clip_ratio_low/high=0.2/0.2`，`clip_ratio_c=3` | 仓库默认值 |
| KL 正则 | actor KL loss 开启，系数 `0.001`，类型 `low_var_kl` | reward 内 KL 未启用；脚本中的 `kl_ctrl.kl_coef=0.001` 不生效于 reward |
| 采样 | vLLM，`n=8`，温度 `0.6` | GPU memory utilization `0.85` |
| 动态 batch / token 上限 | 开启 / 每卡 `32768` tokens | 同时启用 remove padding 和 gradient checkpointing |
| 训练轮数 / 硬件 | 30 epochs / 单节点 4 GPU | 脚本顶部“8 卡”注释与实际参数不符 |
| 验证 / 保存 | 每 20 步 | baseline 训练前验证；SAE 排序变体不做训练前验证 |

## DAPO 配置

| 超参数 | 配置 | 说明 |
| --- | --- | --- |
| 基座模型 / 优势估计 | Qwen2.5-Math-1.5B / `grpo` | DAPO 使用 GRPO 组内优势估计 |
| 训练数据 / batch | `my_data/deepmath/train.parquet` / 128 prompts | 每题采样 8 条，约 1024 条轨迹/批 |
| 动态采样 | `gen_batch_size=384` | 按 `acc` 丢弃全对或全错的回答组，补采至 128 个有效 prompt，最多 10 批 |
| Prompt / response 上限 | 1024 / 3072 tokens | 超长 prompt 过滤开启，`truncation=error` |
| 学习率 / mini-batch | `1e-6` / 64 | warmup 10 步，weight decay `0.1`，grad clip `1.0` |
| PPO 裁剪 | `clip_ratio_low/high=0.2/0.28`，`clip_ratio_c=10` | 非对称裁剪 |
| KL 正则 | actor KL loss 与 reward 内 KL 均关闭 | 系数均为 0 |
| 损失聚合 | `token-mean` | 按有效 token 聚合策略损失 |
| 超长回答惩罚 | 开启，buffer 512 tokens，penalty factor `1.0` | 超过 2560 tokens 后线性增加惩罚，至 3072 tokens 达最大值 |
| 采样 | vLLM，`n=8`，温度 `0.6` | GPU memory utilization `0.85` |
| 动态 batch / token 上限 | 开启 / actor 每卡 `32768` tokens | 同时启用 remove padding 和 gradient checkpointing |
| 训练轮数 / 硬件 | 30 epochs / 单节点 4 GPU | — |
| 验证 / 保存 | 每 10 步 | 训练前验证开启 |

验证集均为 AIME 2024 与 MATH-500，验证 batch size 为 512；验证采样温度为 0.6，每题 8 条。
