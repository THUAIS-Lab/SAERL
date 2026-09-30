# Energy-Guided Absolute Gradient Core-Set Selection

## 1. 方法概述

简要来说，该方法先根据归一化后的 response-mean Gradient-SAE 能量，选出一组全数据共享的
Top-K features。随后在这些 features 上取梯度绝对值，按投影梯度范数从大到小遍历样本，并
用 cosine threshold 去除可由已保留样本代表的冗余样本，得到 gradient core set。最后加入
zero-reward samples，以保护当前模型尚未解决的困难样本。

本文档给出最终的数据选择流程，由以下三个阶段组成：

1. 使用 **Normalized Energy Top-K** 从 Token-TopK response-mean Gradient-SAE 表示中选择
   一组全数据共享的 SAE features；
2. 在选定 features 上构造 absolute response-mean pattern，并使用 exact cosine greedy
   selection 得到非冗余 gradient core set；
3. 将 core set 与 zero-reward samples 合并，作为最终训练数据集。

为了便于论文写作，本文将该流程称为：

> **Energy-Guided Absolute Gradient Core-Set Selection（EAGCS）**。

其核心思想是将“结构冗余”和“当前模型是否能够求解”分开处理：

- Gradient-SAE representation 和 absolute cosine 用于判断两个 samples 是否具有相似的
  gradient feature composition；
- reward 只在最后的数据集组装阶段使用，用于保护当前策略完全失败的 samples；
- 一个 sample 只有在没有进入 gradient core set，且不满足困难样本保护条件时才会被删除。

该方法是 deterministic、unsupervised-with-respect-to-feature-selection 的 filter。Energy
feature selection 和 cosine redundancy selection 都不使用 reward；reward 仅用于最后的
hard-example safeguard。

## 2. 问题定义与记号

设候选数据集为：

$$
\mathcal D=\{d_i\}_{i=1}^{N_{\mathrm{all}}}.
$$

每个 sample 对应：

- sample ID：$i$；
- $R$ 个 rollout；
- response token gradients；
- sample-level reward mean：$r_i$；
- Token-TopK response-mean Gradient-SAE representation：
  $m_i\in\mathbb R^F$。

设 SAE decoder feature 数为 $F$，token-level Top-K 为
$k_{\mathrm{token}}$，最终保留的全局 feature 数为 $K$，cosine redundancy threshold 为
$\tau$。需要区分：

- $k_{\mathrm{token}}$ 控制 extraction 阶段每个 token 的稀疏投影；
- $K$ 控制 sample selection 阶段使用的共享低维空间；
- $\tau$ 控制 core set 的冗余粒度。

## 3. Token-TopK response-mean Gradient-SAE 表示

设 decoder matrix 为：

$$
D\in\mathbb R^{F\times H},
$$

并对 decoder 的每个 feature direction 做 L2 normalization。sample $i$ 的第 $r$ 个
rollout、第 $t$ 个 response token gradient 为：

$$
g_{irt}\in\mathbb R^H.
$$

首先将 gradient 投影到 SAE decoder dictionary：

$$
p_{irt}=g_{irt}D^\top\in\mathbb R^F.
$$

令 $H_{k_{\mathrm{token}}}$ 表示按绝对值保留 Top-K coordinates，同时保存原始符号：

$$
\tilde p_{irt}
=
H_{k_{\mathrm{token}}}(p_{irt}).
$$

对 response tokens 和 rollouts 分别取算术平均：

$$
m_i
=
\frac{1}{R}\sum_{r=1}^{R}
\frac{1}{T_{ir}}\sum_{t=1}^{T_{ir}}
\tilde p_{irt}.
$$

$m_i$ 是带符号的逻辑 $F$ 维向量，但以 sparse IDs/values 保存。未出现在 sample support
中的 feature 值为 0。

这里使用的是 gradient 对 SAE decoder directions 的 signed projection，不是 SAE encoder
forward latent activation。论文中应将其称为 **Gradient-SAE decoder-coordinate
representation** 或 **SAE-dictionary gradient representation**。

## 4. 阶段一：Normalized Energy 全局 feature selection

### 4.1 Feature-fitting 样本集合

Energy selector 使用：

$$
\mathcal A
=
\left\{
i:\ \texttt{zero\_signal\_token\_topk}_i=\mathrm{false},
\ \lVert m_i\rVert_2>0
\right\},
\qquad N=|\mathcal A|.
$$

对每个有效 sample 做 row L2 normalization：

$$
z_i=\frac{m_i}{\lVert m_i\rVert_2}.
$$

因此每个 sample 的总归一化平方能量均为 1：

$$
\sum_{f=1}^{F}z_{if}^2=1.
$$

这一 normalization 使 feature selection 对 sample 的整体 gradient scale 不敏感，避免
少数大范数 samples 对全局 feature ranking 产生平方级支配。

### 4.2 Normalized Energy score

对每个 SAE feature $f$ 定义：

$$
e_f
=
\frac{1}{N}\sum_{i\in\mathcal A}z_{if}^2
=
\frac{1}{N}\sum_{i\in\mathcal A}
\frac{m_{if}^2}{\lVert m_i\rVert_2^2}.
$$

选择得分最大的 $K$ 个全局 features：

$$
G_K
=
\operatorname{TopK}_f(e_f,K),
\qquad |G_K|=K.
$$

实现中累加的是 $\sum_i z_{if}^2$，没有显式除以 $N$。由于分母对所有 features 相同，
不会改变 ranking。

### 4.3 优化解释

对于任意 feature set $G$，其平均相对能量保留率为：

$$
\operatorname{Coverage}(G)
=
\frac{1}{N}\sum_{i\in\mathcal A}
\frac{\lVert m_i[G]\rVert_2^2}{\lVert m_i\rVert_2^2}
=
\sum_{f\in G}e_f.
$$

因此，在逐 feature 可加的目标下，Normalized Energy Top-K 精确求解：

$$
G_K
=
\arg\max_{G:\,|G|=K}
\frac{1}{N}\sum_{i\in\mathcal A}
\frac{\lVert m_i[G]\rVert_2^2}{\lVert m_i\rVert_2^2}.
$$

即：在固定 $K$ 维预算下，最大化所有有效 samples 的平均相对 retained energy。

### 4.4 为什么 Energy 与 absolute downstream representation 一致

下游对 feature values 取绝对值，而：

$$
|z_{if}|^2=z_{if}^2.
$$

因此 absolute transform 不改变 Energy ranking。与 signed variance 不同，Energy 不会把
只由正负翻转产生的变化错误解释为 absolute-pattern distinction。

Energy 的主要局限是它优化 average coverage，而不是直接优化 pairwise cosine distortion；
此外，它是逐 feature criterion，不显式处理不同 SAE decoder directions 之间的相关性。

## 5. 阶段二：Absolute selected-feature representation

对每个 sample，将原始 signed response mean 投影到共享 feature set $G_K$，再逐元素取
绝对值：

$$
x_i
=
|m_i[G_K]|
\in\mathbb R_{\ge0}^{K}.
$$

定义 selected-feature signal strength：

$$
M_i=\lVert x_i\rVert_2
=
\lVert m_i[G_K]\rVert_2.
$$

绝对值不改变 L2 norm。sample 在以下条件下进入 active set：

$$
\mathcal V
=
\left\{
i:\ \texttt{zero\_signal\_token\_topk}_i=\mathrm{false},
\ M_i>\epsilon
\right\}.
$$

对 active sample 构造单位 pattern：

$$
u_i=\frac{x_i}{M_i}.
$$

这里 $M_i$ 和 $u_i$ 承担不同作用：

- $M_i$ 表示 sample 在选定全局子空间中的 gradient signal strength，用于确定 greedy
  processing order；
- $u_i$ 只描述 absolute feature composition，用于计算 scale-invariant similarity。

对 selected features 取绝对值表示本方法忽略每个 decoder direction 上的正负方向，只
比较 gradient magnitude allocation pattern。两个向量的相似度是：

$$
s_{ij}=u_i^\top u_j
=
\cos\!\left(|m_i[G_K]|,|m_j[G_K]|\right),
\qquad 0\le s_{ij}\le1.
$$

需要强调：

$$
\cos(|a|,|b|)\neq|\cos(a,b)|
$$

通常成立。本方法是先逐坐标取绝对值，再计算 cosine。

## 6. 阶段三：Strength-Ordered Greedy Redundancy Selection

### 6.1 确定性处理顺序

将 active samples 按以下顺序排序：

1. $M_i$ 降序；
2. 若 $M_i$ 相同，则 `sample_id` 升序。

记该顺序为 $\pi(1),\ldots,\pi(|\mathcal V|)$。该顺序保证更强的 selected-feature
gradient signal 优先成为 cluster-like representative，并使算法在相同输入下可复现。

### 6.2 Greedy core-set rule

初始化 core set：

$$
\mathcal C=\varnothing.
$$

依次处理 $i=\pi(t)$。若 $\mathcal C$ 非空，计算当前 sample 与所有已保留 samples 的最大
absolute cosine：

$$
c_i^*
=
\max_{j\in\mathcal C}u_i^\top u_j.
$$

决策规则为：

$$
\begin{cases}
i\in\mathcal C,
& \mathcal C=\varnothing\ \text{or}\ c_i^*<\tau,\\
i\in\mathcal R,
& c_i^*\ge\tau,
\end{cases}
$$

其中 $\mathcal R$ 是 redundancy set。对 redundant sample，记录：

$$
\operatorname{rep}(i)
=
\arg\max_{j\in\mathcal C}u_i^\top u_j.
$$

由于 representative 总是更早被处理，通常满足：

$$
M_{\operatorname{rep}(i)}\ge M_i.
$$

因此 active sample 的冗余条件可以写为：

$$
\boxed{
M_j\ge M_i
{}\land{}
\cos\!\left(|m_i[G_K]|,|m_j[G_K]|\right)\ge\tau
\Longrightarrow
i\text{ is redundant with respect to }j.
}
$$

这里的 `redundant` 是相对于给定 SAE checkpoint、gradient layer、$k_{\mathrm{token}}$、
$G_K$ 和 $\tau$ 的 representation-level redundancy，不应被表述为文本语义完全重复。

## 7. 阶段四：Reward-aware 最终数据集组装

### 7.1 Reward mean 的定义

设第 $r$ 个 rollout 的 reward 为 $y_{ir}$。代码保存：

$$
r_i
=
\frac{1}{R}\sum_{r=1}^{R}y_{ir}.
$$

若每个 rollout reward 为二值 $y_{ir}\in\{0,1\}$，则：

$$
r_i\in
\left\{0,\frac1R,\frac2R,\ldots,1\right\}.
$$

当前已检查的 extraction run 使用 $R=8$，因此 `reward_mean` 一般不只取 0 和 1，还可能
取 $1/8,2/8,\ldots,7/8$。

定义 zero-reward set：

$$
\mathcal H_0=\{i:r_i=0\}.
$$

它表示当前策略在本次采样的所有 rollouts 中均未得到成功 reward。它是 observed failure
set，而不是对样本固有难度的无噪声估计。

### 7.2 用户指定的最终集合

按照“代码保留的 core samples 与所有 reward=0 samples 组成最终数据集”的定义：

$$
\boxed{
\mathcal D_{\mathrm{final}}
=
\mathcal C\cup\mathcal H_0.
}
$$

集合并集会自动去除同时属于 core set 和 zero-reward set 的重复 IDs。

对应的删除集合为：

$$
\mathcal D_{\mathrm{drop}}
=
\mathcal D\setminus\mathcal D_{\mathrm{final}}
=
(\mathcal D\setminus\mathcal C)\cap\{i:r_i>0\}.
$$

这带来一个清晰的策略解释：

- 所有 zero-reward samples 都被保护，即使它们在 absolute gradient space 中与其他样本
  相似；
- 非零 reward samples 只有在进入 gradient core set 时才保留；
- 一个样本只有同时满足“未进入 core set”和“当前策略至少一次获得正 reward”时才删除。

在 sample-level reward 严格二值、即 $r_i\in\{0,1\}$ 的特殊情况下：

$$
\mathcal D_{\mathrm{drop}}
=
(\mathcal D\setminus\mathcal C)\cap\{i:r_i=1\}.
$$

此时可以准确写成：

> We remove solved examples only when they are not required by the gradient-pattern core set,
> while retaining all zero-reward examples as hard-example safeguards.

不建议写成“删除所有 redundant samples 和所有 reward=1 samples”，因为这是两个集合的
并集语义，会错误暗示：

- redundant 但 reward=0 的 samples 也被删除；
- 已进入 core set 但 reward=1 的 samples 也被删除。

更准确的中文表述是：

> 最终数据集由 absolute-gradient core set 与全部 zero-reward samples 的并集组成；因此
> 仅删除未进入 core set 且当前策略已经表现出求解能力的样本。

### 7.3 如果论文只想声称删除 reward=1 的已学会样本

当 $R>1$ 时，用户指定的 $\mathcal C\cup\mathcal H_0$ 还会删除未进入 core set、但
$0<r_i<1$ 的部分成功 samples。因此它不能被描述为“只删除 reward=1 samples”。

如果希望删除条件严格限定为 fully solved，即 $r_i=1$，应改用：

$$
\mathcal D_{\mathrm{final}}^{\mathrm{fully\ solved}}
=
\mathcal C\cup\{i:r_i<1\},
$$

从而：

$$
\mathcal D_{\mathrm{drop}}^{\mathrm{fully\ solved}}
=
(\mathcal D\setminus\mathcal C)\cap\{i:r_i=1\}.
$$

更一般地，可以在 validation set 上选择 learned threshold $\gamma$：

$$
\mathcal D_{\mathrm{final}}(\gamma)
=
\mathcal C\cup\{i:r_i<\gamma\},
$$

$$
\mathcal D_{\mathrm{drop}}(\gamma)
=
(\mathcal D\setminus\mathcal C)\cap\{i:r_i\ge\gamma\}.
$$

$\gamma=0^+$ 对应只保护 zero-reward samples；$\gamma=1$ 对应保护所有非 fully-solved
samples。论文必须报告采用哪一种定义。

## 8. 完整算法

```text
Input:
    sparse Token-TopK response means {m_i}
    extraction zero-signal flags {q_i}
    reward means {r_i}
    global feature budget K
    cosine redundancy threshold tau
    numerical zero threshold epsilon

Stage A: fit global features
    A <- {i: q_i is false and ||m_i||_2 > 0}
    for i in A:
        z_i <- m_i / ||m_i||_2
    for feature f:
        e_f <- sum_i z_if^2
    G_K <- K features with largest e_f

Stage B: build absolute patterns
    for every sample i:
        x_i <- |m_i[G_K]|
        M_i <- ||x_i||_2
    V <- {i: q_i is false and M_i > epsilon}
    for i in V:
        u_i <- x_i / M_i

Stage C: greedy gradient core set
    order V by descending M_i, then ascending sample_id
    C <- empty set
    for i in order:
        if C is empty or max_{j in C} dot(u_i, u_j) < tau:
            add i to C
        else:
            mark i redundant and record its most similar representative

Stage D: hard-example safeguard
    H_0 <- {i: r_i = 0}
    D_final <- C union H_0

Output:
    global feature IDs G_K
    gradient core set C
    redundant-to-representative mapping
    final dataset D_final
```

## 9. 实现对应关系

### 9.1 Global feature selection

实现：

```text
recipe/static_gradient_sae/feature/select_energy_features.py
```

示例：

```bash
python recipe/static_gradient_sae/feature/select_energy_features.py \
  --run-dir RUN_DIR \
  --top-k 256
```

默认输出：

```text
RUN_DIR/feature_selection/energy_feature_indices.npy
```

### 9.2 Absolute gradient core-set selection

实现：

```text
recipe/static_gradient_sae/selection_v1/
select_topk_response_mean_abs_selected_features_v1_2_3.py
```

示例：

```bash
python recipe/static_gradient_sae/selection_v1/\
select_topk_response_mean_abs_selected_features_v1_2_3.py \
  --run-dir RUN_DIR \
  --feature-indices RUN_DIR/feature_selection/energy_feature_indices.npy \
  --cosine-threshold 0.7
```

主要输出：

```text
selected_sample_ids.parquet
silent_sample_ids.parquet
selection_audit.parquet
selection_manifest.json
```

其中 `selection_audit.parquet` 将 samples 标记为 `selected`、`redundant` 或 `silent`，并为
redundant samples 保存 representative sample ID 与 cosine。

### 9.3 Final dataset assembly 的当前实现差异

当前：

```text
recipe/static_gradient_sae/save_deepmath_selection_subsets.py
```

实际构造的是：

$$
\mathcal C
\cup
(\mathcal Q\cap\mathcal H_0),
$$

其中 $\mathcal Q$ 是 `silent_sample_ids.parquet` 中的 silent set。该脚本只补回
**zero-reward silent samples**，没有补回 zero-reward redundant samples。

因此，若最终论文方法定义为：

$$
\mathcal D_{\mathrm{final}}=\mathcal C\cup\mathcal H_0,
$$

则当前 dataset assembly 代码尚不完全一致。组装时必须从 extraction shards 的所有
sample IDs 中读取 `reward_mean==0`，再与 `selected_sample_ids.parquet` 做 union，而不能只
在 `silent_sample_ids.parquet` 内过滤 reward。

在完成该实现对齐前，不能声称现有 `selected_plus_silent_deepmath.parquet` 保留了所有
zero-reward samples。

## 10. 超参数与选择原则

### 10.1 Global feature budget K

$K$ 控制表示压缩率和 projected coverage。建议报告：

$$
K\in\{128,256,512,1024\}.
$$

对每个 $K$ 计算：

$$
C_i(K)
=
\frac{\lVert m_i[G_K]\rVert_2^2}{\lVert m_i\rVert_2^2},
$$

并报告 p5、median、p95 coverage 与 projected-zero count。$K=256$ 可作为当前起点，但
最终选择应由 coverage—retention—downstream performance 曲线支持。

### 10.2 Cosine threshold τ

- 较低 $\tau$：冗余判定更宽松，删除更多 samples，core set 更小；
- 较高 $\tau$：只删除高度相似 samples，core set 更大；
- 代码默认值为 0.98，但实际论文实验应报告明确使用的值，不能只依赖默认参数。

建议至少比较：

$$
\tau\in\{0.70,0.75,0.80,0.90,0.95,0.98\}.
$$

最终阈值应根据目标数据压缩率和 held-out training performance 选择，而不是在最终测试集
上调参。

### 10.3 Reward safeguard

必须明确报告以下哪一种规则：

- zero-only safeguard：$\mathcal C\cup\{r_i=0\}$；
- all-unsolved safeguard：$\mathcal C\cup\{r_i<1\}$；
- threshold safeguard：$\mathcal C\cup\{r_i<\gamma\}$。

若 $r_i$ 来自有限次数随机 rollouts，应报告 $R$、sampling temperature、reward verifier
和随机种子。单次 observed reward 不应被直接解释为样本的固有难度标签。

## 11. 计算复杂度

设 sparse response-mean 的总非零数为 $\mathrm{nnz}$，active sample 数为 $N_a$：

- Energy streaming accumulation：$O(\mathrm{nnz})$；
- 全局 feature ranking：当前完整排序约为 $O(F\log F)$；
- selected-feature projection：$O(\mathrm{nnz})$，输出 dense $N\times K$ matrix；
- exact greedy cosine：最坏约为 $O(N_a|\mathcal C|K)$；
- selected-feature dense storage：$O(N_aK)$；
- reward-aware union：约为 $O(N\log N)$ 或 hash-based $O(N)$。

该流程避免 materialize $N\times F$ dense Token-TopK representation，但 exact cosine
selection 仍可能是主要计算瓶颈。

## 12. 方法局限与论文中需要避免的过度表述

### 12.1 Absolute transform 丢弃方向信息

如果 feature 正负方向具有不同的优化含义，absolute cosine 会把它们视为相同。必须通过
signed cosine baseline 验证这种不变性是否有利。

### 12.2 Greedy selection 依赖处理顺序

方法按 projected strength 排序，因此代表样本偏向强 gradient signal。它不是全局最优
minimum dominating set，也不保证最小 core set。

### 12.3 Energy 只优化平均 coverage

Normalized Energy 可能牺牲少数稀有 samples 的 coverage，也可能选择相互相关的 decoder
features。应报告低分位 coverage，而不只报告平均值。

### 12.4 Redundancy 是 representation-relative

`redundant` 只表示两个 samples 在当前 absolute selected-feature gradient space 中的
cosine 超过阈值，不代表题目文本、答案或学习价值完全相同。

### 12.5 Reward 不是无噪声的 learned/unlearned 标签

`reward_mean=1` 更准确地表示当前策略在所采样的 $R$ 个 rollouts 中全部成功；
`reward_mean=0` 表示全部失败。有限采样、解码温度和 verifier error 都会产生不确定性。

论文中建议使用：

- “currently solved under the selection policy”；
- “observed zero-reward examples”；
- “all sampled rollouts failed/succeeded”。

避免直接写成：

- “the model has permanently learned the sample”；
- “zero-reward samples are intrinsically hard”；
- “redundant samples contain no useful information”。

## 13. 推荐实验与消融

### 13.1 组件消融

至少比较：

1. Random selection；
2. Reward-only filtering；
3. Full Token-TopK absolute cosine，不做 global feature selection；
4. Energy selected features + signed cosine；
5. Energy selected features + absolute cosine；
6. Energy selected features + absolute cosine + zero-reward safeguard；
7. AbsVar/PF-AbsVar selected features + absolute cosine，作为 feature criterion 消融。

该顺序分别验证 feature compression、absolute invariance 和 reward safeguard 的贡献。

### 13.2 应报告的 selection 指标

- feature coverage 的 p5/median/p95；
- source-silent 与 projected-zero 数量；
- active、selected、redundant 和 final dataset 数量；
- core-set retention ratio；
- redundant-to-representative cosine 分布；
- selected、redundant、silent 三组的 reward_mean 分布；
- zero-reward samples 的实际保留率，必须为 100% 才能支持 hard-example safeguard claim；
- 不同 $K$、$\tau$、random rollout seeds 下 selected IDs 的稳定性；
- 最终训练的 accuracy/reward、训练 tokens 和 wall-clock efficiency。

### 13.3 公平比较原则

不同方法应保持：

- 相同候选数据池；
- 相同最终数据量，或同时报告性能—数据量 Pareto curve；
- 相同训练 token budget、optimizer 和训练 steps；
- 相同 reward evaluation protocol；
- feature selection 和 threshold tuning 不访问最终测试集。

## 14. 论文写作 takeaway

### 14.1 推荐的主叙事

最合适的主线不是“我们删除低 reward 数据”，而是：

> 我们先在 Gradient-SAE decoder coordinate system 中构造一个保留平均相对 gradient
> energy 的共享子空间，再在该子空间内通过 absolute cosine 提取结构多样的 gradient core
> set。最后加入 zero-reward safeguard，使压缩过程不会删除当前策略完全失败的样本。

该叙事对应三个清晰的 inductive biases：

1. **Energy preservation**：共享低维空间应覆盖大多数 samples 的相对 gradient signal；
2. **Gradient-pattern diversity**：高 cosine 的 absolute feature patterns 可以由更强样本
   代表；
3. **Hard-example protection**：当前策略完全失败的 samples 不受 redundancy pruning
   约束。

### 14.2 推荐的 contribution 表述

如果消融实验支持，可以将 contribution 写为：

1. 提出一种基于 Token-TopK Gradient-SAE response representation 的数据 core-set
   selection framework；
2. 使用 normalized-energy global features，在固定维度预算下最大化平均相对 gradient
   energy retention；
3. 使用 strength-ordered absolute-cosine greedy selection 去除已被更强 gradient pattern
   覆盖的 samples；
4. 使用 reward-aware safeguard 保留所有 observed zero-reward examples，在数据压缩和
   hard-example coverage 之间取得平衡。

不要把 Energy Top-K 或 greedy cosine 单独包装成全新的优化算法；更可信的贡献是它们在
Gradient-SAE data selection pipeline 中的组合、实现和实证验证。

### 14.3 可直接改写到论文 Method 的英文段落

> We first construct a sparse response-level gradient representation by projecting token-wise
> gradients onto an SAE decoder dictionary, retaining the largest-magnitude coordinates per
> token, and averaging them across response tokens and rollouts. We then select a shared set of
> decoder coordinates using normalized energy, which maximizes the average fraction of
> sample-level gradient energy retained under a fixed feature budget. In the selected subspace,
> each example is represented by the element-wise magnitude of its response-mean gradient
> coordinates. We process examples in descending projected gradient norm and retain an example
> only when its cosine similarity to every previously retained example is below a threshold.
> Finally, we augment the resulting gradient core set with all observed zero-reward examples to
> prevent the compression procedure from discarding examples on which the current policy
> consistently fails.

### 14.4 可直接改写到论文 Discussion 的谨慎结论

> The method should be interpreted as representation-dependent core-set construction rather
> than semantic deduplication. Its reward safeguard reflects the current policy and finite
> rollout sample, while its redundancy decisions depend on the chosen SAE dictionary, feature
> budget, and cosine threshold. Accordingly, we evaluate both downstream performance and the
> stability of the selected set across feature budgets, thresholds, and rollout seeds.

### 14.5 最重要的实验 takeaway

如果结果显示 EAGCS 在更少训练样本下保持或提高性能，最稳妥的结论是：

> 训练数据中存在可由相似 absolute gradient patterns 表示的结构冗余；normalized-energy
> feature compression 能够保留足以识别这类冗余的共享子空间，而 zero-reward safeguard
> 避免数据压缩过度偏向当前模型已经能够求解的区域。

如果加入 zero-reward samples 后才出现主要性能增益，则应把贡献归因于
**gradient diversity selection 与 failure-case preservation 的互补性**，而不是只归因于
SAE feature selection。
