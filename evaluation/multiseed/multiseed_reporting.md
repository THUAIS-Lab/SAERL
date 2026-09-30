# 多 seed 峰值实验报告

## 为每个 seed 指定 run

编辑同目录的 `runs.example.json`，将每个方法和 seed 对应的值改成 Markdown 表中 **Method 列的完整行名**，不要加数据集前缀或 `.json` 后缀。例如，若你确认 `sel_v1_plus_silent_v4mix8_run2` 是要用于 seed42 的运行，将 SAERL 的 `"42"` 值改成该名称即可。脚本不从run2这样的后缀推断真实seed，映射由你提供。

```bash
python evaluation/multiseed/multiseed_analysis.py \
  --runs-json evaluation/multiseed/runs.example.json \
  --output-dir evaluation/multiseed/peak_selected_runs
```

提供配置时必须完整列出 GRPO 和 SAERL 各自的所有seed（默认0、42、999；可用 --seeds 更改）。配置优先于 --baseline-tag、--saerl-tag 和 --unsuffixed-seed；不提供配置则保留原命名规则。每个seed选定的run统一用于六个数据集。脚本在写结果之前检查缺失seed、重复run和两个输入表中不存在的run。

`summary.json` 的 runs 字段和 `table.md` 的 Selected runs 表记录最终映射，`per_seed.csv` 的 source_row 记录每个分数的run。配置中的seed归属需与实际训练记录一致；若按测试表现挑选重复run，报告中也应说明该选择规则。

## 统计口径

直接读取用户指定的两份 Markdown 表中完整精度的数值，不使用 step900 单点准确率：

- AIME：`evaluation/final_results/Qwen2.5-Math-1.5B/pass128/acc_pass128_before_900.md`，pass128。
- AMC、GSM8K、MATH、Minerva、Olympiad：`evaluation/final_results/Qwen2.5-Math-1.5B/mean128/acc_mean128_before_900.md`，mean128。
- 表头明确为 `max step ≤ 900`，因此包含900步。
- GRPO 行为 grpo、grpo_seed0、grpo_seed999；完整 SAERL 行为 sel_v1_plus_silent_v4mix8 及其 _seed0、_seed999。无后缀对应42，排除run2和其他消融。

每个方法、seed、数据集使用该数据集在窗口内的峰值，再跨seed计算均值及样本标准差（ddof=1）。宏平均先对每个seed的六个数据集峰值等权平均，再跨seed统计。各数据集峰值可能来自不同checkpoint，所以宏平均是“六集峰值的平均”，不是单个模型checkpoint的六集表现，也不是先求六集均值再取其峰值。

## 运行和文件

依赖 numpy、scipy。在仓库根目录运行：

```bash
python evaluation/multiseed/multiseed_analysis.py
```

可通过 --mean-table、--pass-table、--saerl-tag、--baseline-tag、--output-dir 指定输入输出。默认输出到 `evaluation/multiseed/peak_before900/`：

- table.md：均值±SD、差值及其95% CI。
- summary.json：上述统计、各方法自身95% CI、逐seed分数、指标映射。
- per_seed.csv：36个分数及指标、seed、来源表和行名。

`provisional_step900/` 是旧口径历史结果，已被本次结果替代，不应用于本次论文表格。

## 统计解释

方法均值区间：mean ± t(0.975,n−1) × SD/√n；n=3时t临界值约4.303。差值默认采用独立两样本Welch t区间；只有明确配对实验设计时才用 --paired，相同数字seed本身不足以证明配对设计。区间依赖近似正态假设，三个seed下估计不稳定。区间为逐项95%，没有多重比较校正。

当前宏平均 GRPO 49.84 ± 0.31，SAERL 52.51 ± 0.69，提升2.67个百分点，Welch 95% CI [1.21,4.13]。六个数据集的跨seed平均峰值均提高，但并非每个数据集的差值区间都排除0，不能声称六项均显著。

这是测试集峰值统计，区间反映该选峰值流程下跨seed的变异，不校正测试集checkpoint选择造成的乐观偏差。论文应明确说明选峰值规则，并保持方法之间训练预算、评测频率和候选checkpoint数量可比。不能将这里的区间解释为独立验证集选定模型后的无偏泛化区间。

## 论文与rebuttal

主表两行方法、七列结果，标注mean ± sample SD，n=3，seeds={0,42,999}。附录列出逐seed分数和差值95% CI。

表注可用：

> Results are reported as mean ± sample standard deviation over three training seeds (0, 42, and 999). For each seed and benchmark, we use the peak score among evaluated checkpoints at training steps ≤ 900. AIME uses the pass128 metric, while the other five benchmarks use mean128. The macro-average is computed over the six benchmark-specific peaks within each seed before aggregation across seeds. Peaks on different benchmarks may correspond to different checkpoints.

这里保留用户指标标签pass128/mean128；论文若要写成pass@k等正式名称，应按实际评测定义确认k，而不是仅凭目录名推断。

Rebuttal模板（重复范围按真实实验填写）：

> Thank you for highlighting the variability of RL training. We repeated [the RL training stage / the entire pipeline] for both GRPO and SAERL using three seeds (0, 42, and 999). We now report the mean and sample standard deviation of benchmark-specific peak scores within the first 900 training steps, individual-seed results, and approximate 95% confidence intervals. AIME uses pass128, and the other five benchmarks use mean128. SAERL improves the mean peak score on all six benchmarks, with a macro-average of 52.51 ± 0.69 compared with 49.84 ± 0.31 for GRPO, a gain of 2.67 percentage points (Welch 95% CI: [1.21, 4.13]). These intervals describe variation across training seeds under our peak-selection protocol; they do not correct for test-set checkpoint selection, and estimates remain imprecise with only three runs.

若SAE或数据选择等上游产物固定，仅更换GRPO训练seed，请明确“RL training seeds with upstream artifacts held fixed”，不要声称所有随机阶段独立重跑。

## 逐题bootstrap

这两份Markdown和汇总JSON不能恢复逐题信息，当前未计算逐题bootstrap。需要原始题目ID和正确性结果：mean指标按题聚合回答准确率；pass指标按其实际定义聚合（不能用mean代替pass）。同题多个回答作为一个簇，两个方法及各seed按共同题目索引重采样。对已选峰值checkpoint做bootstrap得到的是条件于已选择checkpoint的区间；若评估完整选峰值流程，必须保留全部候选checkpoint的逐题结果，并在每次重采样后重新取峰值。题目bootstrap不能代替训练seed统计，也不会自动消除测试集选择偏差。
