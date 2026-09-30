# 基于 Gradient-SAE Representation 的数据集选择执行方案

## 1. 项目目标

本方案希望将每个训练样本的多个 rollout 梯度投影到预训练 SAE 的 decoder directions 上，构造一个 sample-level 的一维向量，并在该表示空间中进行数据集去重和子集选择。

核心假设是：

> 如果两个样本在 SAE decoder directions 上产生相似的 signed gradient pattern，那么它们在当前模型、当前 checkpoint 和当前训练目标下，会产生相似的局部功能更新作用。因此，可以将其中一部分样本视为冗余并删除。

本方案只使用 gradient，不乘 SAE activation。对应的表示是：

[
q_{irtf}
========

\left\langle
g_{irt},\hat d_f
\right\rangle,
]

其中：

* (i)：原始训练样本；
* (r)：该样本的第 (r) 个 rollout；
* (t)：rollout 序列中的 token；
* (f)：SAE feature；
* (g_{irt}=\partial L_{ir}/\partial h_{irt})：指定 Transformer 层输出上的梯度；
* (\hat d_f=d_f/|d_f|_2)：归一化后的 SAE decoder direction。

这个量表示 loss 沿 SAE feature direction 的方向导数，即“当前样本希望 hidden state 沿哪些 SAE directions 发生变化”。它不是 SAE activation，也不是 activation attribution。

---

# 2. 输入和输出

## 2.1 输入

默认实验设置：

```text
Dataset size:         103,022 samples
Rollouts per sample:  8
Transformer layer:    layer 26 output
Hidden dimension:     2048
SAE feature count:    130,915
Gradient source:      GRPO actor loss
Token region:         response-mask positions
```

每个 rollout 至少需要保存或在线获得：

```text
sample_id
rollout_id
gradient:       [valid_token_count, 2048]
response_mask
advantage
reward
```

如果当前得到的 gradient 已经是实际 GRPO loss 对 hidden state 的梯度，就不需要再次乘 advantage。

## 2.2 输出

表征阶段输出：

```text
sample_id
zero_signal
raw_gradient_norm
rollout_consistency
representation
```

Selection 阶段输出：

```text
sample_id
selected
selection_method
selection_score
cluster_id
representative_id
selection_reason
```

最终返回被选中的原始 `sample_id`，后续可以继续使用现有 curriculum 方法排序。

---

# 3. 阶段一：固定实验条件

在提取任何梯度之前，固定以下配置：

```text
model checkpoint
SAE checkpoint
hook layer
GRPO loss definition
response mask definition
rollout generation parameters
rollout seed
num_rollouts
decoder normalization method
```

## 原因

Gradient representation 是 checkpoint-dependent 的。同一个样本在不同模型状态下可能产生完全不同的梯度方向。

如果不同样本使用不同 checkpoint、不同 rollout 参数或不同 loss normalization，那么样本间距离将混合模型变化和数据差异，无法解释为数据冗余。

因此，第一轮静态 selection 必须在同一个 checkpoint 上完成。

---

# 4. 阶段二：归一化 SAE decoder

设原始 decoder：

[
D\in\mathbb R^{F\times H},
\qquad
F=130915,\quad H=2048.
]

对每个 decoder feature 做 L2 normalization：

[
\hat d_f
========

\frac{d_f}
{|d_f|_2+\epsilon}.
]

代码逻辑：

```python
decoder = decoder.float()
decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-8)
```

## 原因

未归一化时：

[
g^\top d_f
==========

|g|_2|d_f|_2\cos\theta.
]

投影值会同时受到 decoder norm 和方向相似度影响。

归一化后：

[
g^\top\hat d_f
==============

|g|_2\cos\theta,
]

不同 SAE feature 的坐标才具有可比性。

---

# 5. 阶段三：逐 token 投影

对于每个 rollout，获得：

[
G_{ir}
\in
\mathbb R^{T_{ir}\times H}.
]

逐 token 投影：

[
Z_{ir}
======

G_{ir}\hat D^\top
\in
\mathbb R^{T_{ir}\times F}.
]

其中：

[
Z_{irtf}
========

g_{irt}^\top\hat d_f.
]

* (Z_{irtf}>0)：沿 feature (f) 正方向移动会增大 loss；
* (Z_{irtf}<0)：沿 feature (f) 正方向移动会减小 loss；
* 两个样本表示符号相反，说明它们产生相反或冲突的局部更新信号。

该表示是 directional derivative，而不是 gradient 在过完备 SAE 字典上的唯一分解。

## 5.1 分块计算

不能直接生成完整：

```text
[T, 130915]
```

张量并长期保存。

推荐按 feature block 计算：

```python
for start in range(0, num_features, feature_block_size):
    end = min(start + feature_block_size, num_features)
    decoder_block = decoder[start:end]
    projected_block = gradient @ decoder_block.T
    update_token_pooling(projected_block)
```

推荐参数：

```text
feature_block_size = 2048 / 4096 / 8192
```

具体大小根据显存选择。

## 原因

对每个 feature block 完成 token pooling 后即可释放：

```text
[T, feature_block_size]
```

中间张量，不需要保存完整 token-level dense projection。

---

# 6. 阶段四：Token-level pooling

主表示采用两个通道：

1. token mean；
2. signed peak。

## 6.1 Mean 通道

[
m_{irf}
=======

\frac{1}{T_{ir}}
\sum_{t=1}^{T_{ir}}
Z_{irtf}.
]

代码：

```python
mean_block = projected_block.mean(dim=0)
```

## 原因

Mean 表示该 SAE direction 在整个 response 中持续、稳定出现的梯度趋势。

它还可以降低 response 长度对表示尺度的直接影响，适合后续使用 cosine distance 比较 gradient pattern。

## 6.2 Signed peak 通道

首先找到绝对值最大的 token：

[
t^*
===

\arg\max_t |Z_{irtf}|.
]

然后保留该位置的原始有符号值：

[
p_{irf}
=======

Z_{ir,t^*,f}.
]

代码：

```python
peak_index = projected_block.abs().argmax(dim=0)
signed_peak_block = projected_block.gather(
    dim=0,
    index=peak_index.unsqueeze(0),
).squeeze(0)
```

## 原因

不能直接使用：

[
\max_t Z_{irtf}.
]

例如：

[
[-10,-0.1]
]

普通 max 得到 (-0.1)，会丢掉真正最强的负梯度。

Signed peak 使用绝对值定位峰值，但保留原始符号，因此可以同时表示强正梯度和强负梯度。

## 6.3 每个 rollout 的表示

[
x_{ir}
======

[m_{ir}\Vert p_{ir}]
\in
\mathbb R^{2F}.
]

即：

```text
mean block:         130,915 dimensions
signed-peak block:  130,915 dimensions
total:              261,830 dimensions
```

---

# 7. 阶段五：Rollout-level pooling

默认对 8 个 rollout 求均值：

[
m_i
===

\frac1R\sum_{r=1}^R m_{ir},
]

[
p_i
===

\frac1R\sum_{r=1}^R p_{ir},
]

最终：

[
x_i
===

[m_i\Vert p_i].
]

代码逻辑：

```python
sample_mean += rollout_mean
sample_peak += rollout_peak

sample_mean /= num_rollouts
sample_peak /= num_rollouts
```

## 原因

同一个 prompt 的不同 rollout 会产生不同 response 和不同 gradient。

Rollout mean 表示该 prompt 在当前模型策略分布下的平均 gradient pattern，可以降低单个随机 rollout 的噪声。

这里不建议默认对每个 rollout 单独 L2 normalize，因为那会令梯度很小和梯度很大的 rollout 拥有完全相同的权重。

第一版实验应保留原始 rollout 相对尺度，完成 rollout mean 后，再对最终 sample representation 做 normalization。

---

# 8. 零梯度样本的处理

定义：

[
|x_i|_2<\epsilon
]

或所有 rollout 都满足：

```text
nonzero_advantage_rollouts == 0
```

时，将该样本标记为：

```text
zero_signal = True
```

## 处理规则

不要直接将零梯度样本解释成低质量数据。

在 GRPO 中，同一个 prompt 的所有 rollout reward 相同时，group advantage 为零，可能出现：

* 所有 rollout 全部答错；
* 所有 rollout 全部答对；
* reward 无法区分 rollout；
* 当前 checkpoint 对该样本没有 on-policy preference signal。

因此，零梯度只能表示：

> 当前 checkpoint 下没有可用于 GRPO 更新的区分性梯度信号。

推荐将这些样本放入单独的 `silent bucket`。

如果目标保留比例为 50%，则 silent bucket 也随机保留约 50%，而不是全部删除。

---

# 9. 阶段六：高维表示压缩

完整表示为：

[
x_i\in\mathbb R^{261830}.
]

不能直接为 103,022 个样本保存完整 float32 矩阵。

推荐使用两个实现分支。

---

## 9.1 分支 A：CountSketch 工程基线

在投影和 pooling 过程中，直接将 mean 和 signed peak 累积到固定维度：

[
x_i
\rightarrow
s_i\in\mathbb R^{1024}.
]

推荐：

```text
sketch dimension = 1024
random seed = fixed
mean and peak use separate hash spaces
```

例如：

```text
mean channel:  512 dimensions
peak channel:  512 dimensions
```

## 原因

CountSketch 是线性映射，可以在 streaming 状态下更新，不需要保存完整的 261,830 维表示。

它适合：

* 快速验证 gradient representation 是否有聚类结构；
* cosine 去重 baseline；
* 粗粒度 KMeans；
* 为真实 SAE feature selection 提供 pseudo-cluster。

缺点是 sketch 后的坐标无法对应具体 SAE feature。

---

## 9.2 分支 B：选择全局共享 SAE features

不能让每个样本单独选择自己的 Top-K feature，因为这样不同样本使用不同坐标集合，表示无法直接比较。

应当选择一个全数据集共享的 feature set：

[
S\subseteq{1,\ldots,F}.
]

建议采用两遍提取。

### 第一遍：Pilot 数据

随机选择：

```text
10,000–20,000 samples
```

对 pilot 数据构造 CountSketch 表示并聚类。

然后根据不同 gradient clusters 计算每个 SAE feature 的 group Fisher score。

对 feature (f)，将 mean 和 peak 看作同一个 group：

[
v_{if}
======

(m_{if},p_{if}).
]

计算 cluster 间差异：

[
B_f
===

\sum_c n_c
|\mu_{cf}-\mu_f|_2^2.
]

计算 cluster 内差异：

[
W_f
===

\sum_i
|v_{if}-\mu_{c(i)f}|_2^2.
]

最终：

[
J_f
===

\frac{B_f}{W_f+\epsilon}.
]

选择：

```text
top 4096 candidate features
```

然后对高度相关的 decoder directions 或 gradient coordinates 去冗余，最终保留：

```text
K = 1024 / 2048 / 4096
```

第一版推荐：

```text
K = 2048
```

### 第二遍：全量数据

只投影到选定的 2048 个 decoder directions：

[
Z_{irt,S}
=========

G_{irt}\hat D_S^\top.
]

最终每个样本只有：

[
2K=4096
]

维。

全局选择真实 feature，而不是每个样本分别选择 Top-K，能够保证所有样本处于相同坐标系中。

---

# 10. 阶段七：表示标准化

对于最终选定的每个 channel 和 feature，计算：

[
s_f
===

Q_{0.75}(|x_{\cdot f}|)+\epsilon.
]

进行 robust scaling：

[
\tilde x_{if}
=============

\operatorname{clip}
\left(
\frac{x_{if}}{s_f},
-10,10
\right).
]

然后进行 PCA-whitening：

```text
input dimension:  4096
output dimension: 256
```

最后 row L2 normalization：

[
y_i
===

\frac{\operatorname{PCAWhiten}(\tilde x_i)}
{|\operatorname{PCAWhiten}(\tilde x_i)|_2+\epsilon}.
]

## 原因

### Robust scaling

不同 SAE feature 的 gradient 数值尺度可能差异很大。不进行 feature scaling 时，少数高方差 feature 会主导距离。

不建议直接使用普通标准差，因为 gradient 分布可能具有明显重尾和离群值。

### PCA-whitening

SAE decoder 是过完备且非正交的，不同 decoder directions 可能高度相关。

Whitening 可以降低相关 feature 被重复计算的问题，使 cosine distance 更接近不同 gradient behavior modes 的差异。

### Row L2 normalization

最终 selection 的目标是比较 gradient pattern，而不是将 gradient norm 直接当作质量分数。

因此使用：

[
d(i,j)=1-y_i^\top y_j.
]

原始 gradient norm 单独保存，用于分析，不参与主要相似度计算。

---

# 11. 方法一：Cosine Threshold Dedup

## 11.1 目标

直接删除 gradient-SAE representation 极其相似的样本。

适合验证最核心的假设：

> 距离非常近的样本可以视为梯度作用近重复。

## 11.2 执行步骤

### Step 1：构建 ANN index

使用 FAISS：

```text
index type:       IndexFlatIP / IVF-PQ
metric:           inner product
input vectors:    L2-normalized y_i
top-k neighbors:  50
```

由于 (y_i) 已经 L2 normalize，inner product 就是 cosine similarity。

### Step 2：搜索近邻

对每个样本找到 50 个最近邻。

保留满足：

[
\cos(y_i,y_j)\ge\tau
]

的边。

推荐消融：

```text
tau = 0.95
tau = 0.97
tau = 0.98
tau = 0.99
```

默认：

```text
tau = 0.98
```

### Step 3：执行 greedy radius dedup

不要直接只使用 connected components，因为相似关系可能出现 chaining：

```text
A 与 B 相似
B 与 C 相似
但 A 与 C 不相似
```

推荐按局部密度从高到低处理：

1. 计算每个样本阈值范围内的邻居数；
2. 选择邻居数最多的样本作为 representative；
3. 保留 representative；
4. 删除其所有 cosine ≥ (\tau) 的邻居；
5. 在剩余样本中继续。

### Step 4：记录映射

输出：

```text
selected_sample_id
removed_sample_id
representative_id
cosine_to_representative
```

## 11.3 为什么这样做

高局部密度样本位于某种 gradient pattern 的中心，适合作为该模式的代表。

选择一个 representative 并删除其近邻，可以最直接地实现 gradient pattern 去重。

## 11.4 优点

* 最符合原始 intuition；
* 实现简单；
* 可解释性强；
* 不需要指定 cluster 数量。

## 11.5 风险

* 保留比例由阈值间接决定；
* 对 (\tau) 较敏感；
* 稠密区域会被大量压缩，稀疏区域几乎不会压缩。

## 11.6 推荐定位

```text
基础 baseline
```

---

# 12. 方法二：Cluster Quota + Medoid + Farthest-First

## 12.1 目标

在删除相似样本的同时，确保不同 gradient behavior modes 都得到保留，并且可以精确控制最终 subset size。

## 12.2 执行步骤

### Step 1：MiniBatchKMeans 聚类

输入：

```text
256-dimensional whitened representations
```

推荐聚类数：

```text
K_cluster = 512 / 1024 / 2048
```

默认：

```text
K_cluster = 1024
```

### Step 2：为每个 cluster 分配 quota

设总共保留 (M) 个样本，cluster (c) 包含 (n_c) 个样本。

定义：

[
M_c
===

M
\frac{n_c^\alpha}
{\sum_k n_k^\alpha}.
]

默认：

[
\alpha=0.5.
]

设置：

```text
minimum quota per non-empty cluster = 1
```

## 原因

* (\alpha=1)：严格按 cluster 大小分配，小 cluster 容易被忽略；
* (\alpha=0)：所有 cluster 等额，大量放大小 cluster；
* (\alpha=0.5)：兼顾数据频率和稀有 gradient modes。

### Step 3：选择 cluster medoid

精确 medoid 计算成本较高，可以近似为：

[
i_c^{(1)}
=========

\arg\min_{i:c(i)=c}
|y_i-\mu_c|_2.
]

即选择离 cluster centroid 最近的真实样本。

### Step 4：Farthest-first 补充

初始化：

[
S_c={i_c^{(1)}}.
]

重复选择：

[
i^*
===

\arg\max_{i:c(i)=c}
\min_{j\in S_c}
(1-y_i^\top y_j).
]

直到：

[
|S_c|=M_c.
]

### Step 5：合并各 cluster 结果

[
S=\bigcup_c S_c.
]

如果 rounding 导致数量不足，在所有未选择样本中按当前 nearest-selected distance 最大者补齐。

## 12.3 为什么这样做

Cluster 保证不同大类 gradient patterns 都有 quota。

Medoid 保留每个 cluster 的典型样本。

Farthest-first 不断选择当前尚未覆盖的方向，因此避免只选择 cluster 中心附近的一批重复样本。

## 12.4 优点

* 可以精确控制保留比例；
* 同时保留典型模式和边缘模式；
* 工程复杂度适中；
* 适合保留 25%、50%、75% 等较大比例。

## 12.5 风险

* 对 cluster 数量有一定依赖；
* PCA 和 KMeans 会引入近似；
* cluster 边界附近的样本可能被人为分开。

## 12.6 推荐定位

```text
第一推荐主方法
```

---

# 13. 方法三：Hierarchical K-Center

## 13.1 目标

让 selected subset 尽可能覆盖整个 gradient-SAE representation space。

优化目标近似为：

[
\min_{|S|=M}
\max_i
\min_{j\in S}
d(y_i,y_j).
]

其中：

[
d(y_i,y_j)=1-y_i^\top y_j.
]

## 13.2 为什么不直接做全局 K-center

全局 farthest-first 的复杂度大致为：

[
O(NM).
]

对于：

```text
N = 103,022
M = 50,000
```

直接实现成本过高。

因此采用 hierarchical K-center。

## 13.3 执行步骤

### Step 1：粗粒度聚类

使用：

```text
K_cluster = 2048
```

将全数据划分成局部区域。

### Step 2：分配 cluster quota

使用：

[
M_c
===

M
\frac{n_c^{0.5}}
{\sum_k n_k^{0.5}}.
]

### Step 3：cluster 内 K-center

每个 cluster 内：

1. 选择离 centroid 最近的样本作为第一个 center；
2. 保存每个样本到当前 selected set 的最小距离；
3. 每次选择最小距离最大的样本；
4. 更新其余样本的最小距离；
5. 直到达到 quota。

伪代码：

```python
selected = [nearest_to_centroid]
min_dist = cosine_distance(all_points, selected[0])

while len(selected) < quota:
    next_idx = argmax(min_dist)
    selected.append(next_idx)

    new_dist = cosine_distance(all_points, next_idx)
    min_dist = minimum(min_dist, new_dist)
```

### Step 4：全局修正

合并所有 cluster selected samples 后，计算所有未选择样本到全局 selected set 的距离。

将覆盖最差的样本补入，或者用其替换 selected set 中贡献最低的样本。

## 13.4 为什么这样做

K-center 直接优化“最差情况下，每个原始样本离 selected subset 有多远”。

如果两个样本非常相似，其中一个被选择后，另一个的 nearest-selected distance 会很小，因此不会被优先再次选择。

## 13.5 优点

* 与“相似样本冗余，不同样本应保留”的 intuition 高度一致；
* 不需要设定 cosine threshold；
* 直接优化 gradient-space coverage。

## 13.6 风险

* 容易选择离群点；
* 少数噪声 gradient 可能被当作“新方向”；
* 大保留比例下与 cluster farthest-first 方法较接近。

## 13.7 防止离群点的方法

在 K-center 前过滤：

```text
representation norm 非有限
PCA Mahalanobis distance > 99.9 percentile
rollout consistency 极低且 norm 极高
```

这里只删除明显数值异常，不根据 gradient norm 判断样本质量。

## 13.8 推荐定位

```text
第二推荐主方法
```

---

# 14. 方法四：Cluster-wise Log-det Selection

## 14.1 目标

选择一组尽量线性独立、能够张成不同 gradient update directions 的样本。

优化：

[
S^*
===

\arg\max_{|S|=M}
\log\det
\left(
\lambda I+
\sum_{i\in S}y_i y_i^\top
\right).
]

## 14.2 执行步骤

### Step 1：降低维度

Log-det 不适合直接在 4096 维运行。

使用：

```text
PCA-whiten dimension = 128 or 256
```

默认：

```text
dimension = 128
```

### Step 2：粗聚类和 quota

使用 512 或 1024 个 gradient clusters，并按 (n_c^{0.5}) 分配 quota。

### Step 3：候选预筛选

如果一个 cluster 很大，可以先保留：

```text
candidate_count = min(cluster_size, 10 × cluster_quota)
```

候选由以下方式产生：

* 离 centroid 最近的典型样本；
* farthest-first 得到的多样样本；
* 随机补充。

### Step 4：Lazy greedy log-det

定义：

[
A_S
===

\lambda I+
\sum_{j\in S}y_jy_j^\top.
]

加入样本 (i) 的边际收益：

[
\Delta_i
========

\log
\left(
1+y_i^\top A_S^{-1}y_i
\right).
]

每次选择 (\Delta_i) 最大的样本，并用 Cholesky rank-one update 更新矩阵。

推荐：

```text
lambda = 0.1
```

消融：

```text
lambda = 0.01 / 0.1 / 1.0
```

## 14.3 为什么这样做

如果新样本的 gradient direction 已经由 selected samples 表示：

[
y_i^\top A_S^{-1}y_i
]

会较小。

如果新样本提供新的、接近线性独立的方向，其边际增益会较大。

因此 log-det 同时具有：

* 去重；
* 多样性；
* 子空间覆盖；

三种作用。

## 14.4 优点

* 数学目标完整；
* 比简单聚类更关注线性独立的更新方向；
* 不需要人为定义近重复阈值。

## 14.5 风险

* 计算复杂；
* 大 subset size 时边际收益会逐渐变小；
* 需要 PCA-whitening 和候选预筛选。

## 14.6 推荐定位

```text
增强方法或小比例 selection 主方法
```

更适合：

```text
10%–30% retention
```

对于 50%–75% retention，不建议对全量数据直接运行完整 log-det，可以将其作为 cluster 内 tie-break。

---

# 15. 方法五：Gradient Matching

## 15.1 目标

选择一个子集，使其平均 gradient-SAE representation 尽可能接近完整数据集。

全集目标：

[
g_{\mathrm{full}}
=================

\frac1N\sum_{i=1}^N x_i.
]

选择子集和权重：

[
\min_{S,w_i\ge0}
\left|
g_{\mathrm{full}}
-----------------

\sum_{i\in S}w_ix_i
\right|_2^2.
]

## 15.2 表征区别

前四个方法主要比较 gradient direction，因此使用 token mean 和 row L2 normalization。

Gradient matching 需要保留真实的总体梯度贡献，因此推荐使用：

[
u_{ir}
======

\sum_t Z_{irt},
]

而不是 token mean。

再对 rollout 求和或平均：

[
u_i
===

\frac1R\sum_r u_{ir}.
]

不进行 row L2 normalization。

可以使用线性 CountSketch 降到 512 或 1024 维，因为线性映射能够保留求和结构。

## 15.3 执行步骤

### Step 1：计算完整数据目标向量

[
g_{\mathrm{full}}
=================

\frac1N\sum_i u_i.
]

### Step 2：初始化 residual

[
r_0=g_{\mathrm{full}}.
]

### Step 3：贪心选择

每轮选择与 residual 最对齐的样本：

[
i^*
===

\arg\max_i
\frac{u_i^\top r}
{|u_i|_2+\epsilon}.
]

### Step 4：重新拟合权重

对当前 selected samples 解非负最小二乘：

[
w^*
===

\arg\min_{w\ge0}
\left|
g_{\mathrm{full}}-U_Sw
\right|_2^2.
]

### Step 5：更新 residual

[
r
=

g_{\mathrm{full}}-U_Sw^*.
]

循环直到达到目标 subset size。

## 15.4 大规模实现

直接在 103k 数据上反复全量扫描成本较高。

建议：

1. 先进行 512–1024 个 gradient clusters；
2. 每个 cluster 分配 quota；
3. 在 cluster 内做 gradient matching；
4. 最后对所有 selected candidates 做一次全局权重拟合。

## 15.5 为什么这样做

前四种方法强调覆盖不同 gradient patterns。

Gradient matching 强调：

> selected subset 的总体训练信号是否可以替代 full dataset。

它是检验“子集是否保持整体更新作用”的重要 baseline。

## 15.6 风险

全集平均梯度可能由于不同方向互相抵消而很小。

因此即使 reconstruction error 很低，也不一定说明少见 gradient modes 得到了保留。

## 15.7 推荐定位

```text
理论对照 baseline
```

不建议作为唯一主方法。

---

# 16. 推荐的第一轮实验矩阵

为了控制实验成本，第一轮只实现以下四组。

## Experiment A：基础去重

```text
Token pooling:       mean
Rollout pooling:     mean
Compression:         CountSketch 1024
Normalization:       row L2
Selection:           cosine threshold dedup
Threshold:           0.98
```

目的：

验证是否存在明显的 gradient-pattern 近重复样本。

---

## Experiment B：完整表示去重

```text
Token pooling:       mean + signed peak
Rollout pooling:     mean
Feature set:         global K=2048 SAE features
Scaling:             robust scaling
Reduction:           PCA-whiten 256
Normalization:       row L2
Selection:           cosine threshold dedup
Threshold:           0.98
```

目的：

验证 signed peak 是否比纯 mean 提供更多区分能力。

---

## Experiment C：推荐主方法

```text
Token pooling:       mean + signed peak
Rollout pooling:     mean
Feature set:         global K=2048
Reduction:           PCA-whiten 256
Cluster count:       1024
Quota exponent:      alpha=0.5
Selection:           cluster medoid + farthest-first
Retention:           25% / 50% / 75%
```

目的：

在固定 subset size 下同时保证 gradient mode 覆盖和局部去重。

---

## Experiment D：Gradient Matching baseline

```text
Token pooling:       sum
Rollout pooling:     mean
Compression:         linear CountSketch 1024
Normalization:       no row normalization
Selection:           cluster-stratified gradient matching
Retention:           25% / 50%
```

目的：

比较“覆盖不同 gradient patterns”和“逼近完整平均 gradient”哪一种更有利于实际训练。

---

# 17. Selection 结果的离线验收指标

## 17.1 Nearest-selected coverage

对每个原始样本：

[
c_i
===

\max_{j\in S}
\cos(y_i,y_j).
]

报告：

```text
mean coverage
median coverage
5th percentile coverage
minimum coverage
```

值越高，说明 selected subset 越能覆盖原数据。

## 17.2 Duplicate compression

统计：

```text
cosine ≥ 0.95 的样本对减少比例
cosine ≥ 0.98 的样本对减少比例
```

## 17.3 Cluster retention

对每个 gradient cluster 报告：

```text
original count
selected count
retention ratio
```

防止大 cluster 垄断 selection。

## 17.4 Signed feature coverage

对于每个选定 SAE feature，分别统计：

```text
strong positive event coverage
strong negative event coverage
```

需要区分正负，因为相反梯度方向不是重复，而是可能存在更新冲突。基于 signed gradient event 做 coverage 的逻辑也可以完全不依赖 activation。

## 17.5 Full-gradient reconstruction error

主要用于 Gradient Matching：

[
E
=

\frac{
\left|
g_{\mathrm{full}}
-----------------

g_{\mathrm{subset}}
\right|*2
}{
|g*{\mathrm{full}}|_2+\epsilon
}.
]

---

# 18. 最终训练实验

对于每个 selection 方法，保持以下条件完全一致：

```text
base model
rollout configuration
optimizer
learning rate
training batch size
total optimization steps
evaluation frequency
curriculum method
random seeds
```

比较：

```text
Full 103k
Random 25%
Random 50%
Cosine dedup
Cluster + farthest-first
Hierarchical K-center
Log-det
Gradient matching
```

## 主要评价指标

```text
average math accuracy
AIME pass@8
other math benchmarks mean@8
steps to reach fixed accuracy
wall-clock time
training stability
gradient norm
KL
reward
```

## 公平性要求

Selection 后不应简单减少训练 step。

建议比较两种训练预算：

### Equal-step

所有方法训练相同步数。

回答：

> 更小的数据子集是否能在相同优化预算下提高数据效率？

### Equal-epoch

每种方法对自己的数据遍历相同 epoch 数。

回答：

> 去除冗余后是否能以更低计算量达到相似性能？

---

# 19. 方法优先级

| 优先级 | 方法                                      | 主要作用                   | 建议定位        |
| --: | --------------------------------------- | ---------------------- | ----------- |
|   1 | Cluster quota + medoid + farthest-first | 去重并覆盖不同 gradient modes | 主方法         |
|   2 | Cosine threshold dedup                  | 直接验证近重复假设              | 基础 baseline |
|   3 | Hierarchical K-center                   | 最小化最差覆盖距离              | 第二主方法       |
|   4 | Gradient matching                       | 保持全集平均更新               | 理论 baseline |
|   5 | Cluster-wise log-det                    | 覆盖线性独立方向               | 增强实验        |

---

# 20. 推荐的最终执行流水线

```text
1. 固定 model checkpoint、SAE 和 rollout 配置
2. 提取每个 rollout 的指定层 token-level gradient
3. 对 SAE decoder directions 做 L2 normalization
4. 分 feature block 计算 gradient × decoder^T
5. 在 response tokens 上计算 mean + signed peak
6. 对 8 个 rollouts 求 mean
7. 将零梯度样本放入独立 silent bucket
8. 使用 pilot split 选择全局共享的 2048 个 SAE features
9. 重新提取全量 4096 维 sample representation
10. 按 feature 做 robust scaling
11. PCA-whiten 到 256 维
12. 对 sample vector 做 L2 normalization
13. MiniBatchKMeans 划分 1024 个 gradient modes
14. 按 cluster_size^0.5 分配 selection quota
15. 每个 cluster 先选 medoid，再做 farthest-first
16. 合并 selected sample IDs
17. silent bucket 按相同 retention rate 随机采样
18. 对 selected subset 运行原有 curriculum
19. 与 random、cosine dedup 和 gradient matching 比较
```

该主线与已有材料中总结的“token-wise signed decoder projection → token pooling → rollout pooling → feature reduction → cosine/coverage selection”方向一致，但这里严格限定为纯 gradient representation，不乘 activation。
