# Global Feature Selection for Response-Mean Representation

## 1. 范围与结论

本目录实现了三种 **全局、无监督、逐 feature（univariate）** 的选择方法：

| 方法 | 实现 | 排序依据 | 主要作用 |
|---|---|---|---|
| Normalized Energy Top-K | `select_energy_features.py` | sample 归一化后的平均平方值 | 保留总体归一化能量最大的 features |
| Normalized Variance Top-K | `select_variance_features.py` | sample 归一化后的跨样本方差 | 保留最能区分不同 samples 的 features |
| Prevalence-Filtered Variance Top-K | `select_prevalence_variance_features.py` | 先过滤低流行度 feature，再按方差排序 | 抑制由极少数 samples 产生的尖峰 feature |

三个脚本当前都只读取稀疏的 Token-TopK `response_mean`：

```text
topk_response_feat_ids
topk_response_feat_mean
zero_signal_token_topk
```

它们输出一组对所有 samples 共享的 SAE feature IDs，而不是让每个 sample 单独选维度。
这保证下游不同 samples 的第 $k$ 个坐标具有相同语义。

当前建议是：

1. 将 **Dense response mean + Energy/Variance** 作为线性表示基线，但要另写或复用支持
   Dense 输入的统计代码；本目录三个脚本并不处理 Dense；
2. 将 **Token-TopK response mean + Energy** 作为最小稀疏基线；
3. 将 **Token-TopK response mean + Prevalence-Filtered Variance** 作为主方案；
4. 同时报告普通 Variance，判断收益来自 variance criterion，还是主要来自 prevalence
   regularization。

一句话概括三者的差异：**Energy 选择“信号强”的维度，Variance 选择“样本间变化大”的
维度，Prevalence-Filtered Variance 选择“在足够多样本中反复显著、且样本间变化大”的
维度。**

## 2. Response-mean representation 从哪里来

### 2.1 Dense response mean 是线性基线

设 L2 归一化后的 SAE decoder 为：

$$
D\in\mathbb{R}^{F\times H},
$$

response token gradient 为 $g_{irt}\in\mathbb{R}^{H}$，其中 $r$ 是 rollout，$t$ 是
response token。Dense response mean 为：

$$
m_i^{\mathrm{dense}}
=
\frac{1}{R}\sum_{r=1}^{R}
\frac{1}{T_{ir}}\sum_{t=1}^{T_{ir}}
g_{irt}D^\top.
$$

因为 projection 和 mean 都是线性的：

$$
m_i^{\mathrm{dense}}=\bar g_iD^\top.
$$

选择全局 feature set $G$ 后：

$$
m_i^{\mathrm{dense}}[G]=\bar g_iD_G^\top.
$$

对应的内积为：

$$
m_i[G]m_j[G]^\top
=
\bar g_iD_G^\top D_G\bar g_j^\top.
$$

所以 Dense SAE representation 本质上是在原始平均 gradient 空间中使用
$D_G^\top D_G$ 定义的线性度量。它没有调用 SAE encoder，也没有引入非线性；SAE 在这里
只提供从 activation 数据中学习到的 decoder directions。论文中应把它与 raw gradient、
PCA 和随机投影比较，避免把 Dense 分支描述成“SAE latent activation”。

### 2.2 Token-TopK response mean 引入 token-dependent 非线性

令 $H_{k_{\mathrm{token}}}(\cdot)$ 表示对每个 token 的投影按绝对值选择 Top-K，保留原始
符号。Token-TopK response mean 为：

$$
m_i^{\mathrm{topk}}
=
\frac{1}{R}\sum_{r=1}^{R}
\frac{1}{T_{ir}}\sum_{t=1}^{T_{ir}}
H_{k_{\mathrm{token}}}\!\left(g_{irt}D^\top\right).
$$

一般有：

$$
\frac{1}{T}\sum_t H_{k_{\mathrm{token}}}(g_tD^\top)
\neq
H_{k_{\mathrm{token}}}\!\left(\frac{1}{T}\sum_t g_tD^\top\right).
$$

原因是每个 token 的 Top-K mask 不同。这个 token-dependent hard thresholding 过滤了 Dense
投影中大量微弱尾部，同时保留每个 token 与 decoder dictionary 对齐最强的 directions。
经过 token 和 rollout pooling 后，一个 sample 的 support 可以大于
$k_{\mathrm{token}}$，不同 samples 的 support 也可以不同；但 feature IDs 始终位于同一个
全局 SAE 坐标系。

这里有两个不要混淆的 K：

- $k_{\mathrm{token}}$：extraction 阶段每个 token 保留多少个投影；
- $K_{\mathrm{global}}$：本目录 feature selection 最终在全数据上保留多少个共享 features。

下文用 $K$ 表示 $K_{\mathrm{global}}$。

## 3. 三种方法共享的输入与预处理

### 3.1 有效样本集合

对 sample $i$，将 sparse IDs/values 还原为逻辑向量
$m_i\in\mathbb{R}^{F}$。没有出现在 `topk_response_feat_ids` 中的 feature 值严格视为 0。
代码实际使用的有效集合是：

$$
\mathcal A
=
\left\{
i:\ \texttt{zero\_signal\_token\_topk}_i=\mathrm{false},
\ \lVert m_i\rVert_2>0
\right\},
\qquad N=|\mathcal A|.
$$

也就是说，三个脚本都会：

1. 跳过 extraction 标记为 zero-signal 的 sample；
2. 再防御性地跳过 `response_mean` 范数恰好为 0 的 sample；
3. 用最终实际进入统计的 sample 数作为 `sample_count`。

`zero_signal_token_topk` 是由 prompt/response 的 mean/peak 四个 channels 共同决定的
extraction-level 标记，而这里仅使用 `response_mean`。因此 flag 为 false 但
`response_mean` 范数为 0 不一定是坏数据，也可能表示信号只存在于 prompt 或 signed-peak
channel。论文统计中应将它记录为 `response_mean_zero` 或 branch-zero，而不是误报为
source-silent。脚本 stdout 中的 `samples` 因此可能合理地小于
`zero_signal_token_topk=False` 的行数。

### 3.2 Sample-level L2 normalization

对每个有效 sample 做：

$$
z_i=\frac{m_i}{\lVert m_i\rVert_2}.
$$

代码对零范数先跳过，因此除法时没有额外加 $\epsilon$。归一化后：

$$
\sum_{f=1}^{F}z_{if}^2=1.
$$

这样每个 sample 给 feature ranking 的总平方质量相同，选择依据是 gradient pattern，而不
是原始 gradient norm。否则少数大范数 samples 会支配全局 ranking。

这一归一化只用于 **拟合 feature set**。当前下游 selected-feature 脚本仍从原始
$m_i[G]$ 计算投影后 signal strength，再为 cosine 计算做 row normalization。因此论文中
应区分：

- feature selection statistic 使用 $z_i$；
- 下游 sample 强度可以使用 $\lVert m_i[G]\rVert_2$；
- 下游 cosine pattern 使用 $m_i[G]/\lVert m_i[G]\rVert_2$。

### 3.3 稀疏统计中的隐式零

实现只对当前 row 中出现的 IDs 更新 `value_sum`、`squared_sum` 或 `score`。没有出现的
feature 不执行加法，等价于向该 feature 加 0。因此后面的均值和方差分母仍然是全部 $N$
个有效 samples，而不是该 feature 的非零次数。

这是正确的总体语义：一个 feature 在大部分 samples 中缺失，本身就是其跨样本分布的一
部分。不能只在非零样本上计算 variance，否则会完全丢失稀疏度信息。

## 4. 方法 A：Normalized Energy Top-K

### 4.1 定义

对每个 feature $f$ 定义归一化能量：

$$
S_f^{\mathrm{energy}}
=
\frac{1}{N}\sum_{i\in\mathcal A}z_{if}^2.
$$

最终选择：

$$
G_K^{\mathrm{energy}}
=
\operatorname{TopK}_f\left(S_f^{\mathrm{energy}},K\right).
$$

代码实际累加的是 $\sum_i z_{if}^2$，没有除以 $N$；因为所有 features 的分母相同，
ranking 和上式完全一致。又因为每个 $z_i$ 都是单位向量：

$$
\sum_f S_f^{\mathrm{energy}}=1.
$$

所以 $S_f^{\mathrm{energy}}$ 还能解释为 feature $f$ 平均承担的 sample-normalized
gradient energy 份额。

### 4.2 它偏好什么

Energy 会同时偏好：

- 在许多 samples 中出现的 feature；
- 只在部分 samples 中出现、但相对幅度很大的 feature；
- 对所有 samples 都近似同方向、同幅度的公共 feature。

它不关心符号，因为使用平方值。feature 的符号仍保存在下游 representation 中，只是
不影响该阶段的 ranking。

### 4.3 优点与局限

优点是只有 $K$ 一个主要超参数，稳定、容易解释，并直接优化平均 retained energy：

$$
\frac{1}{N}\sum_i\sum_{f\in G}z_{if}^2.
$$

局限是“能量大”不等于“区分性强”。例如 $z_{if}=0.1$ 对所有 samples 都成立时，该
feature 有非零 energy，却不能区分任何两条 sample。若下游目标是 cosine redundancy 或
coverage diversity，这类共同背景方向可能浪费维度预算。

因此它最适合充当简单、强且易复现的 baseline。

## 5. 方法 B：Normalized Variance Top-K

### 5.1 定义与实现

先计算 signed feature mean：

$$
\mu_f=\frac{1}{N}\sum_{i\in\mathcal A}z_{if},
$$

再计算 population variance：

$$
S_f^{\mathrm{var}}
=
\frac{1}{N}\sum_{i\in\mathcal A}(z_{if}-\mu_f)^2
=
\frac{1}{N}\sum_i z_{if}^2-\mu_f^2.
$$

选择：

$$
G_K^{\mathrm{var}}
=
\operatorname{TopK}_f\left(S_f^{\mathrm{var}},K\right).
$$

代码用 `value_sum` 和 `squared_sum` 单遍 streaming 得到二阶矩，使用分母 $N$ 而不是
$N-1$。对所有 features 使用相同 $N$ 时，这不会改变排序。

### 5.2 与 Energy 的精确关系

两者满足：

$$
S_f^{\mathrm{energy}}
=
S_f^{\mathrm{var}}+\mu_f^2.
$$

因此区别只来自 dataset-level signed mean：

- 若某 feature 在所有 samples 中都稳定为正或稳定为负，$|\mu_f|$ 大，Energy 高但
  Variance 低；
- 若某 feature 在不同 samples 中正负翻转，$\mu_f\approx 0$，Energy 与 Variance 几乎
  相同；
- 若绝大多数 signed projections 的均值接近 0，两种方法的 Top-K 会高度重合。

一个简单例子是：

```text
feature a: [0.1,  0.1,  0.1,  0.1]  -> energy > 0, variance = 0
feature b: [0.1, -0.1,  0.1, -0.1]  -> energy = variance > 0
```

Variance 会优先选 `b`，因为它能够描述样本差异；Energy 可能同时保留 `a` 和 `b`。

### 5.3 优点与局限

Variance 更接近“为区分 samples 选择坐标”的目标，并自动压低公共偏置方向。它仍然是
无监督方法，不使用 reward、difficulty、label 或下游 selection 结果。

它的主要局限有两个：

1. 稀有尖峰也会产生较大方差。若 feature 以概率 $p$ 取值 $a$、其余为 0，则其方差约为
   $p(1-p)a^2$；很大的 $a$ 可能补偿很小的 $p$；
2. 按单个 feature 排序不考虑 feature-feature correlation，可能同时选择多个高度相关的
   SAE decoder directions。

另外，**按 variance 选 feature 不等于下游已经中心化**。当前 selector 只保存 feature
IDs，不保存 $\mu_G$；现有下游也使用未中心化的 signed coordinates。若论文方法要写成
$z_i[G]-\mu_G$，必须额外保存训练集均值并在下游实际应用。按当前实现，准确表述应是：
“variance-ranked features with uncentered signed-cosine downstream comparison”。

## 6. 方法 C：Prevalence-Filtered Variance Top-K

### 6.1 为什么需要 prevalence filter

Token-TopK 的 support 天然稀疏。普通 variance 无法区分“在很多 samples 中稳定出现的
差异”与“只被一两个异常 sample 激活的尖峰”。后者可能在统计上方差很高，但用于构造
共享全局子空间时泛化和覆盖较差。

Prevalence-Filtered Variance 先提出一个硬约束：feature 必须在足够多 samples 的
**sample 内显著集合**中出现；只在满足约束的候选中比较 variance。

### 6.2 显著集合与 prevalence

先忽略 row 中数值恰好为 0 的 coordinates，定义：

$$
T_i
=
\operatorname{TopL}_f\left(|m_{if}|\right).
$$

因为 $z_i=m_i/\lVert m_i\rVert_2$ 只乘了一个 sample 内正常数，用 $|m_{if}|$ 或
$|z_{if}|$ 做 sample 内 Top-L 排序完全等价。若 sample 的非零 feature 少于 $L$，则把
所有非零 features 都计入 $T_i$。

feature 的显著出现次数与 prevalence 为：

$$
c_f=\sum_{i\in\mathcal A}\mathbf 1(f\in T_i),
\qquad
P_f=\frac{c_f}{N}.
$$

候选集合为：

$$
\mathcal C
=
\{f:c_f\ge p_{\min}N\}.
$$

由于 $c_f$ 是整数，代码中的比较等价于：

$$
c_f\ge\left\lceil p_{\min}N\right\rceil.
$$

例如 $N=339,p_{\min}=0.02$ 时，feature 至少要在 7 个 samples 的 Top-L 中出现。

### 6.3 候选内按 variance 排序

方差仍按第 5 节在全部 $N$ 个有效 samples 上计算，包括隐式零。最终选择：

$$
G_K^{\mathrm{prev-var}}
=
\operatorname{TopK}_{f\in\mathcal C}
\left(S_f^{\mathrm{var}},K\right).
$$

因此它不是把 prevalence 和 variance 加权求和，而是：

```text
硬约束：P_f >= p_min
软排序：在合格 candidates 中最大化 variance
```

这使方法具有清晰的两阶段解释：**reliability gate + discriminability ranking**。

### 6.4 三个超参数分别控制什么

- $L$（`--salient-top-l`）：每个 sample 给多少个 feature 投 prevalence 票。$L$ 越大，
  越多 features 容易通过筛选，prevalence 约束越弱；
- $p_{\min}$（`--min-prevalence`）：至少多大比例的有效 samples 必须把该 feature 视为
  显著。越大越强调稳定复现，但越容易删除稀有模式；
- $K$（`--top-k`）：最终共享空间的维数，控制 coverage、存储和下游计算成本。

必须满足 $|\mathcal C|\ge K$。否则脚本会报：

```text
Only C features satisfy min_prevalence, fewer than top_k=K
```

调参时建议先固定 $L$，画出 candidate count 随 $p_{\min}$ 的曲线，再在有充分余量的候选
集合上选择 $K$。不建议为了勉强得到 $K$ 个 features 而持续降低 $p_{\min}$，因为这会让
主方法退化为普通 Variance。

当前脚本默认值也是现阶段的推荐起点：

$$
L=128,\qquad p_{\min}=0.02,\qquad K=256.
$$

已有探索性记录中，339 个有效 `response_mean` samples 在该配置下产生约 460 个
candidates，足以选出 256 个 features。这个 candidate 数依赖数据和 extraction 配置，
正式实验应从本次运行的 stdout/manifest 重新报告。

### 6.5 优点与局限

该方法保留“反复显著且具有区分性”的 features，对 sample-specific spike 更稳健，特别
适合 Token-TopK 稀疏表示。

它也会有意过滤真正稀有但语义重要的 feature。因此 prevalence 是 robustness prior，
不是“重要性”的充分定义。论文中应报告稀有模式上的性能或至少做 $p_{\min}=0$ 的消融，
不能只报告总体均值。

Dense 表示几乎所有 features 都非零，因此不能用简单的 `value != 0` 定义 Dense
prevalence。若要把该思想用于 Dense，应仍使用 sample 内 Top-L rank，而不是非零频率；
本目录目前没有对应 Dense 实现。

## 7. 三种方法的直观对比

| 情况 | Energy | Variance | Prevalence-Filtered Variance |
|---|---|---|---|
| 所有 samples 都有稳定同号的大值 | 容易选中 | 会被均值项压低 | 通过 prevalence，但仍被方差压低 |
| 许多 samples 中正负或幅度不同 | 容易选中 | 容易选中 | 若反复进入 Top-L，容易选中 |
| 极少数 samples 中出现大尖峰 | 可能选中 | 可能选中 | 可能在排序前就被过滤 |
| 很常见但始终很弱 | 可能因累计能量选中 | 方差通常低 | prevalence 可能通过，但方差通常低 |
| 稀有且任务关键 | 可能选中 | 可能选中 | 有被误删风险 |
| 主要超参数 | $K$ | $K$ | $L,p_{\min},K$ |

更形式化地说：

- Energy 直接最大化平均 retained energy，是 reconstruction/coverage-oriented criterion；
- Variance 最大化逐坐标的边际变化，是 discrimination-oriented criterion；
- Prevalence-Filtered Variance 在 discrimination 前加入跨样本复现约束，是
  robustness-oriented criterion。

三者都是逐坐标打分，均没有显式处理 decoder directions 或 sample coordinates 之间的
相关性。若选出的 features 高度冗余，可以在 Top-M candidates 后增加 decoder cosine
去冗余、mRMR、PCA/whitening，或用下游指标验证是否真的需要更复杂方法。

## 8. 如何选择全局维数 K：sample coverage

固定 $K$ 最简单。若希望给 $K$ 一个数据驱动解释，可以检查每个 sample 在共享子空间中
保留的归一化平方能量：

$$
C_i(K)
=
\sum_{f\in G_K}z_{if}^2
=
\frac{\lVert m_i[G_K]\rVert_2^2}{\lVert m_i\rVert_2^2},
\qquad 0\le C_i(K)\le 1.
$$

可选择满足下式的最小 $K$：

$$
Q_{0.05}\left(\{C_i(K)\}_{i\in\mathcal A}\right)\ge\tau.
$$

例如 $\tau=0.05$ 表示至少 95% 的有效 samples 在共享 feature set 中保留不低于 5% 的
原始 response-mean 平方能量。这是最低信号/coverage 要求，不是重构质量要求。

每一种 ranking 都应单独计算 coverage，不能用 Energy 的覆盖数字替代 Prevalence-Filtered
Variance。Dense 和 Token-TopK 也必须分别拟合 $G_K$ 并分别评估。

现有探索性记录为：

| Representation | Global K | p5 coverage | median coverage |
|---|---:|---:|---:|
| Token-TopK | 256 | 6.25% | 8.54% |
| Token-TopK | 512 | 7.53% | 10.14% |
| Dense | 256 | 0.36% | 0.45% |
| Dense | 4096 | 4.93% | 5.30% |

另有记录显示 Energy 与 Variance 的 Top-256 重合为：

- Dense response mean：249 / 256；
- Token-TopK response mean：254 / 256。

这与 $S^{\mathrm{energy}}=S^{\mathrm{var}}+\mu^2$ 且 signed mean 接近 0 的分析一致，也暗示
“Energy 换成 Variance”本身可能不是主要增益来源；更值得检验的是 token-level Top-K 和
prevalence filter。

以上数字是当前目录已有的探索性结果，不是 selector 自动生成的完整实验报告。用于论文
前应绑定具体 run、数据 split、SAE checkpoint、$k_{\mathrm{token}}$ 和 selector 参数重新
生成，不能把它们当作跨数据集结论。

## 9. 实现细节与复现命令

### 9.1 Streaming 算法

三个脚本都按文件名字典序读取 `step-*.parquet`，再按 Arrow batch/row streaming 处理，
不 materialize $N\times F$ dense matrix。

设第 $i$ 个 sparse row 的 support 大小为 $s_i$：

- Energy/Variance 的 row accumulation 约为 $O(\sum_i s_i)$；
- Prevalence-Filtered Variance 还会对每个 row 的非零值做完整稳定排序，约为
  $O(\sum_i s_i\log s_i)$，因此当前实现通常最慢；
- 三者的全局 accumulator memory 均为 $O(F)$；
- 最后 Energy/Variance 对 $F$ 个 scores 排序，prevalence 版本只对 candidate scores 排序。

当前实现用稳定降序排序。分数完全相同时，较小 feature ID 会先出现。选出 Top-K 后又对
IDs 做一次升序排序，因此保存的 `.npy` 表示 **feature set**，不保留 score ranking 顺序。

### 9.2 命令

```bash
python recipe/static_gradient_sae/feature/select_energy_features.py \
  --run-dir RUN_DIR \
  --top-k 256

python recipe/static_gradient_sae/feature/select_variance_features.py \
  --run-dir RUN_DIR \
  --top-k 256

python recipe/static_gradient_sae/feature/select_prevalence_variance_features.py \
  --run-dir RUN_DIR \
  --top-k 256 \
  --salient-top-l 128 \
  --min-prevalence 0.02
```

默认输出分别为：

```text
RUN_DIR/feature_selection/energy_feature_indices.npy
RUN_DIR/feature_selection/variance_feature_indices.npy
RUN_DIR/feature_selection/prevalence_variance_feature_indices.npy
```

所有输出都是一维 `int64` NumPy array，按 feature ID 升序排列。

### 9.3 当前实现的复现缺口

selector 目前只保存 indices，并在 stdout 打印 sample/candidate count；它没有保存：

- 每个 feature 的 score、mean 或 prevalence；
- 按 score 排序的完整 feature ranking；
- 参数、输入 shard hash、manifest hash、git commit 和运行时间；
- coverage、bootstrap stability 或不同方法间的 overlap。

正式论文实验建议给每次 selection 增加 JSON/Parquet manifest，至少记录上述信息。尤其是
Variance 若要在下游中心化，必须保存训练 split 上拟合的 $\mu_G$。

三个脚本还依赖 extraction contract：每个 row 的 IDs 与 values 等长、IDs 严格递增且不
重复、所有值有限、IDs 位于 $[0,F)$。下游 loader 会做较完整校验，但 feature selectors
本身没有逐项验证；正式批量实验前最好加入同样的数据校验。

## 10. 与下游 sample selection 的边界

选择 feature set 后，对 sample $i$ 的投影为：

$$
x_i=m_i[G].
$$

若使用 signed cosine，则：

$$
u_i=\frac{x_i}{\lVert x_i\rVert_2}.
$$

feature ranking 中使用平方/绝对值不表示下游应该丢弃符号。当前 representation 是 signed
gradient projection；正负方向可能具有不同功能含义。除非专门做 absolute-pattern
消融，主实验应保留符号。

还需要明确 projected-zero sample 的处理。原则上有三种状态：

- `silent`：extraction 已标记 `zero_signal_token_topk=True`；
- `active`：原始有效且在 $G$ 上有足够 signal/coverage；
- `out_of_global_subspace`：原始有效，但投影后为零或 coverage 过低。

从语义上，`out_of_global_subspace` 不等于原始 gradient silent。为了不因全局 feature
selection 系统性丢掉稀有模式，可以将它单独审计并直接保留。

但是，当前 `select_topk_response_mean_selected_features_v1_2_2.py` 会把投影范数不高于
`zero_eps` 的原始有效 sample 计入 `projected_zero_count`，并放入其 silent set。这与上面的
推荐语义不同。论文实验必须明确选择一种策略，并让文档、代码和统计口径一致；至少要把
source-silent 与 projected-zero 分开报告。

## 11. 论文写作 takeaway 与实验方向

### 11.1 推荐的主叙事

不要把主要贡献写成“用 variance 做 feature selection”，因为它是标准的单变量过滤思想，
且当前 Energy/Variance Top-K 高度重合。更有说服力的主线是：

> 使用 SAE decoder 作为共享的过完备 gradient dictionary；先通过 token-conditioned
> hard Top-K 得到稀疏、带符号的 sample representation，再用 prevalence constraint
> 选择跨样本稳定复现的候选，最后用 variance 保留具有区分性的全局坐标。

这个叙事把方法拆成三个可检验的设计：

1. **dictionary**：SAE decoder directions 是否优于 raw/PCA/random directions；
2. **token-conditioned sparsification**：Token-TopK 是否优于 Dense；
3. **global reliability + discriminability**：prevalence gate 与 variance ranking 是否优于
   单纯 energy。

方法名可以用 **Prevalence-Constrained Variance Selection (PCVS)**，比
“prevalence-filtered variance Top-K”更适合在图表中使用。应把它表述为简单、可解释的
无监督 selector，而不是声称它学习了新的 feature extractor。

### 11.2 最小但完整的 baseline ladder

建议保持相同最终维数 $K$、相同下游 selection algorithm 和阈值，只替换表示/feature
selector：

1. Raw response-mean gradient；
2. PCA 或 random projection；
3. Dense SAE response mean + Energy；
4. Token-TopK response mean + Energy；
5. Token-TopK response mean + Variance；
6. Token-TopK response mean + PCVS。

其中 3→4 检验 token-level nonlinear sparsification，4→5 检验 variance，5→6 检验
prevalence constraint。这样的递进比只比较最终方法和一个弱 baseline 更容易支持因果归因。

### 11.3 建议报告的指标

除了最终训练 reward/accuracy，至少报告：

- retained sample 数、去重率与下游训练效果；
- $C_i(K)$ 的 p5/median/p95，而不只报告平均 coverage；
- source-silent、projected-zero/out-of-global-subspace 的数量；
- selected feature prevalence/variance 分布；
- 不同 random seed 或 bootstrap samples 下 feature set 的 Jaccard stability；
- Energy、Variance、PCVS 的 Top-K overlap；
- feature selection 与下游 selection 的运行时间、峰值内存；
- 若强调可解释性，展示稳定高 prevalence feature 的案例，而不是只展示最大值样本。

### 11.4 关键消融

建议优先做：

- $k_{\mathrm{token}}\in\{32,64,128,256\}$：验证 token sparsity；
- $K\in\{128,256,512,1024\}$：给出性能—coverage—成本曲线；
- $L\in\{64,128,256\}$ 与 $p_{\min}\in\{0,0.01,0.02,0.05\}$：验证 prevalence
  robustness，其中 $p_{\min}=0$ 退化为普通 Variance；
- signed vs absolute downstream cosine：验证方向信息是否有用；
- 是否做 decoder-cosine 去冗余或 whitening：检验 univariate selection 的相关性局限；
- feature selector 在 train split 拟合并冻结到 held-out split：验证 feature set stability，
  避免把数据集特定 ranking 写成普适结论。

### 11.5 可以落在论文结论里的谨慎表述

若实验支持，可以写：

> Token-level sparsification concentrates gradient mass into a compact shared SAE coordinate
> system. A prevalence constraint removes sample-specific spikes, while variance ranking
> preserves coordinates that differentiate examples. Together, they provide a more compact
> and stable subspace for redundancy-aware data selection than dense energy ranking.

如果 PCVS 与普通 Variance 的下游结果接近，则更准确的 takeaway 是：prevalence filter
提高了 feature-set stability/可解释性，但主要性能来自 Token-TopK representation，而不
应把无显著差异写成 selection criterion 的性能提升。
