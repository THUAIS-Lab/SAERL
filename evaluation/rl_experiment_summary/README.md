# RL 实验总览与论文实验部分建议

本报告只盘点 `evaluation/final_results` 已归档的 RL 评测结果，不将训练脚本存在视为实验已经完成。生成日期：2026-09-15。

## 1. 核心判断

论文最有证据的主线是：**Gradient-SAE 驱动的数据选择与训练顺序设计，在 DeepMath 上改善 RL 的学习效果；完整方法在 1.5B 的三个训练 seed 下稳定优于 GRPO，部分收益可扩展到 7B、较大训练 batch 和 DAPO。**

实验不能支持“完整方法在所有设置中最优”“所有模块都有稳定叠加收益”或“跨训练数据集普遍有效”。silent 样本保留的消融最清晰；selection 与 mix 的互补性仍是开放问题。Math8k 的结果应作为局限性诚实报告。

## 2. 已跑过哪些实验

共 697 个 JSON：694 个逐数据集结果文件 + 3 个 avg128 派生平均文件。按模型、训练设置、运行标签去重，共 **58 个运行标签**，不是 58 个独立随机 seed，也不代表每个训练任务均已结束。

| 模型与设置 | 运行标签数 | 结果覆盖 | 用途 |
|---|---:|---|---|
| Qwen2.5-Math-1.5B / DeepMath128 | 41 | 每个运行均有六集 mean 和 pass 文件 | 主结果、消融、seed、算法扩展 |
| Qwen2.5-Math-7B / DeepMath128 | 13 | 12 个六集完整；完整方法 run2 缺 Minerva | 模型规模扩展 |
| Qwen2.5-Math-1.5B / DeepMath512 | 2 | GRPO、v4mix8，各六集双指标 | batch 扩展 |
| Qwen2.5-Math-1.5B / Math8k128 | 2 | LIMR、v4mix8，各六集双指标 | 训练数据变化与 LIMR 对比 |

1.5B 的 41 个运行可完整分为：

| 组别 | 全部运行标签 |
|---|---|
| GRPO（3） | `grpo`, `grpo_seed0`, `grpo_seed999` |
| 外部/难度基线（4） | `gainrl`, `gradalign`, `difficulty_baseline`, `difficulty_mean16` |
| AdaRFT（7） | `adarft_grpo`, `adarft_grpo_100`, `adarft_grpo_100_run2`, `adarft_grpo_100_mean128`, 后者的 `run2/run3/run4` |
| DAPO（3） | `dapo`, `dapo128`, `dapo_v4mix8` |
| 数据选择（4） | `sel_v1`, `sel_v2`, `sel_v1_plus_silent`, `sel_v1_plus_silent2` |
| 完整组合（4） | `sel_v1_plus_silent_v4mix8`, 其 `run2/seed0/seed999` |
| mix 与排序变体（10） | `v4mix4`, `v4mix8`, `v4mix8_seed0`, `v4mix16`, `v4mix16_run2`, `v4mix32`, `v4r`, `v5r`, `v6r`, `v5mix8` |
| 表示/来源消融（6） | `v4mix8_promptonly`, `emb_full_v4mix8`, `emb_sel_v1_plus_silent_v4mix8`, `raw_grad_sel_v1_plus_silent`, `llama_v4mix8`, `ood_difficulty_v4mix8` |

7B 的全部 13 个标签见 [全量表](all_runs.md)。版本号、`silent2`、`run2`、AdaRFT 的 `100/mean128` 后缀不能仅凭名字解释为特定超参数或随机种子；论文中的正式命名需要绑定实际运行配置。`llama_v4mix8` 归档的被训练模型仍是 Qwen1.5B，不能据此声称训练了 Llama backbone。

## 3. 统计口径：写论文前先固定

- 根据 `evaluation/eval.py` 的 `n=8`、`calculate_mean.py` 和 `calculate_pass.py`，mean 为全部回答的正确率（8 次采样估计的单次成功率），pass 为每题 8 次中至少一次正确的比例。**pass128 不是 pass@128；128/512 对应训练 batch 设置。**
- 题目数由 pass JSON 的 total 对应为 AIME 30、AMC 40、GSM8K 1319、MATH 500、Minerva 272、Olympiad 673。mean 的 total 为它们的 8 倍。
- 本报告默认沿用已有多 seed 报告：AIME 用 pass，其余五集用 mean，六集等权平均，称为 Mixed Avg。这是异质指标的汇总，正文必须在表头/表注明确。全量表同时提供统一六集 mean 的 Mean Avg，避免结论依赖指标切换。
- DeepMath128 在 step ≤900 内取各数据集峰值；DeepMath512 和 Math8k 在 step ≤300 内取峰值。独立峰值平均不等于某个 checkpoint 的实际整体表现。
- [全量表](all_runs.md)额外给出六集**共同 step**上 mean 宏平均的峰值及 step，便于检查统一 checkpoint 表现；它仍是测试集选峰值，不等于独立验证集选模型。
- 已逐项重算现有 DeepMath128 ≤900、DeepMath512 ≤300 Markdown 峰值表，与原始 JSON 无数值不一致；694 个逐集 JSON 未发现重复 step，accuracy 与 correct/total 一致。原始采样过程与真实训练完成状态不能只由汇总 JSON 验证。

## 4. 正文建议一：主效果与训练随机性

### 4.1 最适合主张的多 seed 结果

1.5B，seed={0,42,999}，沿用现有 `evaluation/multiseed` 的映射；无后缀为42，排除 run2。单位为百分数，± 为样本标准差。

| 指标 | GRPO | 完整 SAERL | 提升（百分点） |
|---|---:|---:|---:|
| AIME pass | 31.11 ± 1.92 | 36.67 ± 3.33 | +5.56 |
| AMC mean | 53.65 ± 0.65 | 55.31 ± 0.94 | +1.67 |
| GSM8K mean | 81.34 ± 0.79 | 85.84 ± 0.21 | +4.50 |
| MATH mean | 71.82 ± 0.59 | 72.75 ± 0.38 | +0.93 |
| Minerva mean | 26.16 ± 0.86 | 28.48 ± 0.55 | +2.31 |
| Olympiad mean | 34.96 ± 0.43 | 36.00 ± 0.49 | +1.04 |
| Mixed Avg | **49.84 ± 0.31** | **52.51 ± 0.69** | **+2.67** |

现有 Welch 差值95%区间为 [1.21,4.13] 个百分点；不能声称六个单项都统计显著。区间反映测试集选峰值后的训练间变异，三个 seed 的估计有限，也未校正测试集挑 checkpoint 的偏差。固定上游 SAE/数据产物时，应称“RL training seeds”，不能称完整 pipeline 独立重跑。

### 4.2 基线与模型规模（单个无后缀运行）

| 方法 | 1.5B Mixed Avg | 1.5B Mean Avg | 7B Mixed Avg | 7B Mean Avg |
|---|---:|---:|---:|---:|
| GRPO | 49.48 | 46.77 | 60.02 | 56.20 |
| GainRL | 49.52 | 46.18 | 61.37 | 56.58 |
| difficulty_baseline | 50.57 | 47.65 | 60.74 | 56.99 |
| difficulty_mean16 | 52.24 | 48.83 | 61.97 | 57.10 |
| v4mix8 | 52.48 | 48.18 | 62.02 | 57.58 |
| sel_v1_plus_silent | 52.53 | 48.99 | 61.80 | 57.36 |
| 完整 SAERL | 51.91 | 48.71 | 61.87 | 57.99 |

解释：完整方法相对 GRPO 在两种模型规模、两种宏平均下均有收益。**1.5B 默认运行完整方法低于 selection-only 和 mix-only 的 Mixed Avg；7B 完整方法的 Mixed Avg 略低于 v4mix8 与 difficulty_mean16，但统一 mean 宏平均更高。**因此结论是有效且具有竞争力，不能写全面领先。单 seed 基线与多 seed 完整方法的均值不要无说明地混在同一列比较。

AdaRFT 多个配置的全部结果在全量表；不要从 run1–4 按测试表现挑一行代表方法。GradAlign 只有200步、dapo128只有700步，应在共同预算内比较，不能直接把其截短峰值用于证明900步方法胜出。

## 5. 正文建议二：组件消融——最值得讲 silent 保留

1.5B，≤900步，默认运行：

| 方法 | Mixed Avg | Mean Avg |
|---|---:|---:|
| GRPO | 49.48 | 46.77 |
| sel_v1 | 49.40 | 46.21 |
| sel_v2 | 50.80 | 46.42 |
| sel_v1_plus_silent | **52.53** | **48.99** |
| sel_v1_plus_silent2 | 52.40 | 48.58 |
| v4mix8 | 52.48 | 48.18 |
| 完整组合 | 51.91 | 48.71 |
| 完整组合 run2 | 52.58 | 49.04 |

**可讲的机制假设：当前 checkpoint 下梯度信号为零，不意味着这些题在后续训练没有价值。**保留 silent 后相对 sel_v1，Mixed Avg 增加3.12点，Mean Avg 增加2.78点，六个 mean 单项均增加。该结果与保留潜在有用训练样本的解释一致，但尚不是动态机制的直接证明。

完整组合并未稳定超过 selection-only；建议用 `GRPO / selection+silent / mix / selection+silent+mix` 的四格消融，明确报告混合效果。优先补齐四格相同 seed，才能讨论交互和互补。

## 6. 正文建议三：表示选择与 mix 参数

### 表示消融

| 对照 | Mixed Avg | Mean Avg | 对应参照 |
|---|---:|---:|---|
| raw_grad_sel_v1_plus_silent | 50.77 | 46.95 | SAE selection+silent：52.53 / 48.99 |
| emb_sel_v1_plus_silent_v4mix8 | 51.82 | 48.00 | 完整组合：51.91 / 48.71 |
| emb_full_v4mix8 | 50.27 | 47.28 | v4mix8：52.48 / 48.18 |
| v4mix8_promptonly | 50.88 | 47.20 | v4mix8：52.48 / 48.18 |
| llama_v4mix8 | 50.41 | 47.28 | v4mix8：52.48 / 48.18 |
| ood_difficulty_v4mix8 | 50.85 | 47.10 | v4mix8：52.48 / 48.18 |

SAE 表示相对 raw-gradient 选择、embedding 表示、prompt-only 版本的表现支持表示设计有价值。但比较前要核对选择样本数量、阈值、数据顺序；否则无法把收益完全归因于 SAE。本报告对 `llama/ood` 只按运行标签描述，确切表示来源与难度来源要回查产物 manifest。

### mix 强度与排序版本

| 版本 | Mixed Avg | Mean Avg |
|---|---:|---:|
| v4mix4 | 51.66 | 47.49 |
| v4mix8 | **52.48** | **48.18** |
| v4mix16 | 50.53 | 46.22 |
| v4mix16_run2 | 50.38 | 46.56 |
| v4mix32 | 50.66 | 47.05 |
| v4r | 50.19 | 46.24 |
| v5r | 50.15 | 46.82 |
| v6r | 50.86 | 47.04 |
| v5mix8 | 49.79 | 46.87 |

当前默认运行以 mix8 最好，可用折线/柱状图说明参数并非越大越好。mix8 的 seed0 只有49.78 Mixed Avg，表明其单独收益有明显运行变异；不要用此单 seed 扫描宣称8是普适最优。版本定义需要对照生成排序文件的配置，而不是由文件名猜测。

## 7. 正文建议四：效率与可扩展性

### 7.1 学习效率：可以讲达到同等分数的训练步数

由原始 JSON 重算，目标为1.5B GRPO在≤900步的各集峰值；AIME用pass，其余用mean。以下为首次达到目标的评测step。

| 方法 | AIME | AMC | GSM8K | MATH | Minerva | Olympiad |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 40 | 680 | 840 | 800 | 780 | 800 |
| v4mix8 | 20 | 580 | 520 | 620 | 840 | 760 |
| selection+silent | 20 | 540 | 380 | 440 | 480 | 600 |
| 完整组合 | 160 | 640 | 480 | 700 | 540 | 720 |

selection+silent 在 GSM8K 上为840/380≈2.21倍步数效率；完整组合为840/480=1.75倍。完整组合 AIME 达标反而更晚，不要概括为所有任务更快。优先展示 GSM8K、MATH、Minerva 的学习曲线和统一宏均值曲线。

已有 `acc_speedup` 中10000表示未达到，不是真实步数；部分列目标设置为0，没有解释价值。没有 GPU时间、实际生成token数、预处理成本时，只能称训练步数效率，不能称端到端加速。DAPO有额外采样/过滤，同step也未必同计算量。

### 7.2 扩展实验

| 设置与窗口 | 基线 Mixed / Mean Avg | 对照方法 Mixed / Mean Avg | 结论 |
|---|---:|---:|---|
| DeepMath512，≤300 | GRPO 49.55 / 45.52 | v4mix8 50.58 / 47.32 | 较大batch仍有收益；六集mean均提高，AIME pass下降 |
| DAPO，≤900 | dapo 52.23 / 48.27 | dapo_v4mix8 52.63 / 48.89 | 有小幅额外收益；Olympiad下降，其余五集mean提高 |
| Math8k，≤300 | LIMR 50.02 / 45.99 | v4mix8 49.45 / 45.49 | 当前未超过LIMR，应作为边界结果 |

512 batch和128 batch不能直接比较相同步数：900×128=115200个名义prompt位置，300×512=153600，且尚未计入rollout和动态过滤成本。Math8k有的v4mix8曲线延伸至395/520/560步，但本报告统一截到300；不能拿其更长训练峰值对比LIMR300步。

7B 的训练脚本引用1.5B生成的选择/排序产物（`sae/qwen7b/sorted_dm128a.sh`），这可作为跨目标模型规模复用策略的线索；提交论文前用实际运行配置确认，脚本可能被后续编辑。

## 8. 推荐的论文实验章节与图表

1. **Experimental Setup**：模型、训练数据、六个benchmark、每题8采样、训练batch/rollout、预算、基线配置、checkpoint选择协议。
2. **Main Results**：两模型规模的主结果；1.5B多seed表作为最强证据，说明7B为单运行结果。
3. **Component Analysis**：四格selection/mix消融，加silent保留对照；将“为什么不能直接丢掉无信号样本”作为分析重点。
4. **Representation and Mixing Analysis**：raw gradient / embedding / prompt-only；mix4/8/16/32敏感性。可将版本开发过程v4r/v5r/v6r移至附录。
5. **Learning Efficiency and Extensions**：达标步数与学习曲线；batch512、DAPO扩展。Math8k/LIMR放补充结果与局限性。

推荐正文配置：主结果表、组件消融表、表示消融表；学习曲线图、mix参数图。附录提供58个运行标签全表、checkpoint覆盖、多seed逐项结果、AdaRFT配置和Math8k结果。

### 按优先级补什么

- **优先级1：公平性与统计口径。**补齐强基线 `difficulty_mean16` 和 selection-only 的同seed结果；固定主方法和seed映射，勿从run2挑最好者。确认数据重排前后题目集合一致、评测与训练数据隔离；训练脚本将AIME/MATH作为val时，解释它们是否参与方法/超参数选择。
- **优先级2：完整方法的必要性。**四格消融都跑相同seed，确认mix究竟增强峰值、早期效率还是稳定性；目前三者不能混为一谈。
- **优先级3：效率证据。**补真实GPU-hours、采样token与离线SAE/排序成本，或收敛到同目标分数的总成本。
- **优先级4：覆盖补全。**7B完整方法run2缺Minerva、Olympiad只到740步；GradAlign到200、dapo128到700。补齐前明确缺失/截短。Math8k扩展完整方法前，不主张跨训练数据集普适提升。

## 9. 可复查产物与来源

- [all_runs.md](all_runs.md)：58个运行标签的六集峰值、两种宏平均、共同checkpoint宏平均和覆盖边界。
- [inventory.csv](inventory.csv)：694个逐集JSON的checkpoint数量、最小/最大step、峰值step、窗口末步分数、原始路径。
- [summarize.py](summarize.py)：标准库脚本；仓库根目录执行 `python evaluation/rl_experiment_summary/summarize.py` 重建上述两份文件。README是人工分析，不由脚本自动重写。
- 指标实现：`evaluation/eval.py`、`evaluation/calculate_mean.py`、`evaluation/calculate_pass.py`。
- 历史峰值与速度实现：`evaluation/find_max_and_step.py`、`evaluation/find_min_step.py`。
- 多seed细节：`evaluation/multiseed/multiseed_reporting.md`、`evaluation/multiseed/peak_before900/table.md`。
- 配置线索：`sae/qwen1_5b/`、`sae/qwen7b/`、`recipe/dapo/deepmath128_qwen2_5*.sh`、`recipe/static_gradient_sae/raw_grad_sel/README.md`。
