# 基于两种 Gradient-SAE Representation 的数据集选择执行方案 V2

## 1. 项目目标

本方案希望将每个训练 sample 的多个 rollout 梯度投影到预训练 SAE decoder directions 上，构造两个 sample-level representation，并分别在两个表示空间中进行数据去重和子集选择。

核心假设是：

> 如果两个 sample 在 SAE decoder directions 上产生相似的 signed gradient pattern，那么它们在当前模型、当前 checkpoint 和当前训练目标下会产生相近的局部功能更新作用，因此其中一部分 sample 可能是冗余的。

本方案只使用 gradient，不乘 SAE activation。设：

$$
q_{irtf}
=
\left\langle g_{irt},\hat d_f\right\rangle,
$$

其中：

- `i`：原始训练 sample；
- `r`：该 sample 的第 `r` 个 rollout；
- `t`：rollout 序列中的有效 token；
- `f`：SAE feature；
- `g_{irt}=\partial L_{ir}/\partial h_{irt}`：指定 Transformer 层输出上的梯度；
- `\hat d_f=d_f/\lVert d_f\rVert_2`：归一化后的 SAE decoder direction。

`q_{irtf}` 表示 loss 沿 SAE feature direction 的方向导数。它不是 SAE activation，也不是 activation attribution。

本方案同时构造两个版本：

1. **Dense gradient-SAE representation**：保留每个 token 的全部 `F` 个 signed decoder projections；
2. **Token-TopK gradient-SAE representation**：每个 token 按 projection 绝对值选 Top-K，保留原始符号，其他 feature 位置设为 0。

两个版本都必须：

- 将 prompt 和 response 分开处理；
- 在 token 维度分别计算 mean pooling 和 signed max pooling；
- 在 rollout 维度做等权算术平均；
- 最终为每个 sample 得到一个逻辑维度为 `4F` 的向量。

固定布局为：

```text
[prompt_mean(F)
 || prompt_signed_peak(F)
 || response_mean(F)
 || response_signed_peak(F)]
```

因此每个 sample 一共得到两个独立的 `4F` representations。两个版本默认分别用于下游 selection，不默认拼接为一个 `8F` 向量。

---

## 2. 输入和输出

### 2.1 输入

默认实验设置：

```text
Dataset size:              103,022 samples
Rollouts per sample:       8
Transformer layer:         layer 26 output
Hidden dimension H:        2,048
SAE feature count F:       130,915
Gradient source:           GRPO actor loss
Token regions:             prompt and response, handled separately
Gradient feature Top-K:    128
```

每个 rollout 至少需要在线获得或暂存：

```text
sample_id
rollout_id
gradient:             [prompt_len + response_len, H]
prompt_len
response_len
attention_mask
response_mask
advantage
reward
sequence_nll
```

如果捕获到的 gradient 已经是实际 GRPO actor loss 对目标层 hidden state 的梯度，就不能再次乘 advantage。

### 2.2 Representation 输出

每个 source sample 输出一行，至少包含：

```text
sample_id
num_rollouts
nonzero_advantage_rollouts
reward_mean / reward_std / pass_rate
prompt_gradient_token_count
response_gradient_token_count
raw_gradient_norm diagnostics
zero_signal_dense
zero_signal_token_topk
dense representation metadata and values
token_topk representation metadata and sparse values
```

Dense 版本的逻辑输出：

```text
dense_grad_feat_vals: length = 4F
```

Token-TopK 版本的逻辑输出也是 `4F`，但默认用 feature ID 并集稀疏保存，并提供无损恢复为 `4F` 的函数。

### 2.3 Selection 输出

```text
sample_id
representation_version       # dense / token_topk
selected
selection_method
selection_score
cluster_id
representative_id
selection_reason
```

最终返回被选中的原始 `sample_id`，后续可以继续运行现有 curriculum 排序。

---

## 3. 阶段一：固定实验条件

在提取梯度之前，固定并记录：

```text
model checkpoint
SAE checkpoint
hook layer
GRPO loss definition
loss aggregation mode
prompt/response mask definition
rollout generation parameters
rollout seed
num_rollouts
decoder normalization method
gradient feature Top-K K
representation schema version
```

Gradient representation 是 checkpoint-dependent 的。同一个 sample 在不同模型状态下可能产生完全不同的梯度方向。

如果不同 sample 使用不同 checkpoint、rollout 参数、loss normalization 或 Top-K 配置，sample 间距离会混合模型变化和数据差异，无法解释为数据冗余。

不同 provenance 的 shards 必须拒绝合并。

---

## 4. 阶段二：归一化 SAE decoder

设 SAE decoder：

$$
D\in\mathbb R^{F\times H},
\qquad
F=130915,\quad H=2048.
$$

默认对每个 decoder feature 做 L2 normalization：

$$
\hat d_f
=
\frac{d_f}{\lVert d_f\rVert_2+\epsilon}.
$$

代码逻辑：

```python
decoder = decoder.float()
decoder_norms = decoder.norm(dim=1, keepdim=True)
decoder = decoder / decoder_norms.clamp_min(1e-8)
```

未归一化时：

$$
g^\top d_f
=
\lVert g\rVert_2\lVert d_f\rVert_2\cos\theta,
$$

projection 同时受到 decoder norm 和方向相似度影响。归一化后：

$$
g^\top\hat d_f
=
\lVert g\rVert_2\cos\theta,
$$

不同 SAE feature 坐标才更具有可比性。

实现时必须记录归一化前后的 decoder row norm min/mean/max。如果为了复现旧产物而使用 checkpoint 原始 `W_dec`，必须通过显式配置选择；两个 representation 版本必须共用完全相同的 decoder。

---

## 5. 阶段三：切分 prompt/response 并逐 token 投影

### 5.1 Prompt/response token 边界

去掉 padding 后，每个 rollout 的目标层梯度应按下列顺序排列：

```text
[valid prompt tokens || valid generated response tokens]
```

设：

```text
P = prompt_len
Q = response_len
S = P + Q
```

则：

```python
prompt_gradient = compact_gradient[:P]
response_gradient = compact_gradient[P:P + Q]
```

规则：

- padding 不参与 pooling；
- prompt 包括 chat template 中属于 prompt 的有效 token；
- response 包括有效生成 token，EOS 若在 response mask 中则保留；
- 不把 `prompt_len - 1` 的 shifted prediction position 混入 response span；
- 双 span 表示使用完整 attention scope。旧的 `response_loss` shifted scope 不能同时生成完整 prompt 与 response 分支，V2 中应直接拒绝。

虽然 actor loss 定义在 response labels 上，prompt hidden states 仍会通过 causal attention 接收 response loss 的梯度，因此 prompt gradient representation 是有效信号。

### 5.2 逐 token 投影

对每个 rollout、每个 span `s∈{prompt,response}`：

$$
G_{irs}\in\mathbb R^{T_{irs}\times H}.
$$

逐 token 投影：

$$
Z_{irs}
=
G_{irs}\hat D^\top
\in\mathbb R^{T_{irs}\times F}.
$$

其中：

$$
Z_{irstf}=g_{irst}^\top\hat d_f.
$$

- `Z>0`：沿 feature 正方向移动会增大 loss；
- `Z<0`：沿 feature 正方向移动会减小 loss；
- 两个 sample 在相同 feature 上符号相反，代表相反或冲突的局部更新信号。

### 5.3 分块计算

不能长期保存完整 token-level：

```text
[T, 130915]
```

推荐按 token chunk 计算：

```python
for token_chunk in span_gradient.split(projection_chunk_size):
    projected = token_chunk @ decoder.T       # [chunk_tokens, F]
    update_dense_pooling(projected)
    update_token_topk_pooling(projected)
```

当前实现的 decoder 全量驻留 GPU，因此按 token rows 分块最自然。两个版本必须复用同一次 `projected`，不能为 Dense 和 Token-TopK 重复执行 decoder matmul。

---

## 6. 阶段四：构造两个 token-level gradient 版本

### 6.1 Dense 版本

Dense 版本不做 threshold、abs transform 或 feature selection：

$$
Z^{dense}_{irs}=Z_{irs}.
$$

绝对值只允许用于 signed peak 的索引选择，不能替换实际保存值。

### 6.2 Token-TopK 版本

对每个 token 独立选择绝对值最大的 `K` 个 SAE projections：

$$
I_{irst}
=
\operatorname{TopK}_f\left(|Z_{irstf}|,K\right).
$$

然后定义：

$$
Z^{topk}_{irstf}
=
\begin{cases}
Z_{irstf}, & f\in I_{irst},\\
0, & f\notin I_{irst}.
\end{cases}
$$

代码逻辑：

```python
_, topk_ids = torch.topk(projected.abs(), k=K, dim=-1)
topk_vals = projected.gather(dim=-1, index=topk_ids)
```

这里：

- Top-K ranking 使用 `abs(projected)`；
- 保存值来自原始 `projected`，保留正负号；
- 非 Top-K feature 的逻辑值为 0；
- Top-K 必须在 decoder projection 后的 `F` 维执行，不能在 hidden dimension `H` 上执行；
- Top-K 必须逐 token 执行，不能先 token pooling 再 Top-K。

配置使用独立名称：

```text
gradient_feature_topk = 128
```

它与 rollout generation 的 sampling `top_k` 完全不同。默认 128 与当前 SAE checkpoint 的 `config.json: k=128` 对齐。

当 `K>F` 时取 `min(K,F)`。绝对值不大于 `gradient_zero_eps` 的零值不能把任意 feature ID 加入有效 support。

---

## 7. 阶段五：Token-level pooling

两个版本、两个 spans 都采用相同的两个 pooling channels：

1. token arithmetic mean；
2. signed peak，也就是按绝对值选最大 token、保留原符号。

### 7.1 Mean 通道

对版本 `v∈{dense,topk}`：

$$
m^v_{irsf}
=
\frac{1}{T_{irs}}
\sum_{t=1}^{T_{irs}}Z^v_{irstf}.
$$

Top-K 版本中，feature 未被某个 token 选中等价于该位置为 0，因此分母必须是 span 的全部有效 token 数 `T_{irs}`，不能使用 feature occurrence count。

例如某 feature 在 10 个 token 中仅入选一次且值为 `5`，mean 是 `0.5`，不是 `5`。

### 7.2 Signed peak 通道

首先找到绝对值最大的 token：

$$
t^*=\arg\max_t|Z^v_{irstf}|.
$$

然后保留该位置的原始有符号值：

$$
p^v_{irsf}=Z^v_{irs,t^*,f}.
$$

代码语义：

```python
peak_index = projected.abs().argmax(dim=0)
signed_peak = projected.gather(
    dim=0,
    index=peak_index.unsqueeze(0),
).squeeze(0)
```

不能直接使用普通：

$$
\max_t Z_{irstf}.
$$

例如：

```text
[-10, -0.1]
```

普通 max 得到 `-0.1`，会丢掉真正最强的负梯度；signed peak 应得到 `-10`。

跨 projection chunks 时必须比较候选值的绝对值。绝对值相同但符号相反时固定优先正值，并在 manifest 和测试中记录该 tie-break。

Top-K 版本从该 feature 实际入选过的 signed values 中选择最大绝对值；从未入选的 feature 输出 0。

### 7.3 每个 rollout 的四分支表示

对每个版本、每个 rollout：

$$
x^v_{ir}
=
[m^v_{ir,prompt}
\Vert p^v_{ir,prompt}
\Vert m^v_{ir,response}
\Vert p^v_{ir,response}]
\in\mathbb R^{4F}.
$$

布局固定为：

```text
prompt mean:          F
prompt signed peak:   F
response mean:        F
response signed peak: F
total:                4F
```

---

## 8. 阶段六：Rollout-level pooling 与 feature 并集

### 8.1 Rollout 等权平均

对同一 sample 的 `R` 个 rollout，分别对版本、span 和 channel 求算术平均：

$$
\bar m^v_{isf}
=
\frac1R\sum_{r=1}^{R}m^v_{irsf},
$$

$$
\bar p^v_{isf}
=
\frac1R\sum_{r=1}^{R}p^v_{irsf}.
$$

最终：

$$
x_i^v
=
[\bar m^v_{i,prompt}
\Vert\bar p^v_{i,prompt}
\Vert\bar m^v_{i,response}
\Vert\bar p^v_{i,response}]
\in\mathbb R^{4F}.
$$

代码语义：

```python
for rollout in rollouts:
    sample_prompt_mean += rollout_prompt_mean
    sample_prompt_peak += rollout_prompt_peak
    sample_response_mean += rollout_response_mean
    sample_response_peak += rollout_response_peak

sample_prompt_mean /= num_rollouts
sample_prompt_peak /= num_rollouts
sample_response_mean /= num_rollouts
sample_response_peak /= num_rollouts
```

同一个 prompt 的不同 rollout 会产生不同 response、advantage 和 gradient。Rollout mean 表示该 prompt 在当前 policy 分布下的平均 gradient pattern。

规则：

- 不把所有 rollout tokens 拼起来再 pooling，否则 response 较长的 rollout 权重更大；
- 不在 rollout 维度取 max；
- 不对每个 rollout 单独 L2 normalize；
- feature 未出现在某 rollout 的 Top-K support 时按 0 参与平均，不能只除以出现过的 rollout 数。

### 8.2 Token-TopK feature 并集

对每个 rollout、每个 span：

$$
U_{irs}=\bigcup_{t=1}^{T_{irs}}I_{irst}.
$$

对 sample-level 的该 span：

$$
U_{is}=\bigcup_{r=1}^{R}U_{irs}.
$$

“features 取并集”的具体含义是：

1. 先对同一 rollout、同一 span 内所有 token 的 Top-K IDs 取并集；
2. 再对同一 sample、同一 span 的所有 rollout supports 取并集；
3. prompt 与 response supports 分开保存；
4. 不同 sample 之间不取并集；feature ID 已经是全局 SAE 坐标。

同一 span 的 mean 和 signed peak 共用一份排序后的 feature IDs，value arrays 与 IDs 严格对齐。

必须显式跟踪 support，不能在 rollout 平均后用 `value != 0` 推断并集，因为不同 rollout 的正负值可能恰好抵消为 0，但该 feature 仍属于真实 support。

### 8.3 最终两个 sample representations

固定 layout：

```text
offset [0, F):       prompt_mean
offset [F, 2F):      prompt_signed_peak
offset [2F, 3F):     response_mean
offset [3F, 4F):     response_signed_peak
```

当前 `F=130,915`：

```text
4F = 523,660 dimensions
```

每个 sample 分别得到：

```text
x_dense_i       ∈ R^(4F)
x_token_topk_i  ∈ R^(4F)
```

两个版本使用相同坐标布局，分别执行后续 preprocessing 和 selection。

---

## 9. 零梯度样本的处理

对版本 `v` 定义：

$$
\mathrm{zero\_signal}_i^v
=
\mathbb 1[\lVert x_i^v\rVert_2<\epsilon]
\lor
\mathbb 1[\mathrm{nonzero\_advantage\_rollouts}=0].
$$

零梯度不能直接解释为低质量数据。在 GRPO 中，同一 prompt 的所有 rollout reward 相同时，group advantage 为 0，可能出现：

- 所有 rollout 全部答错；
- 所有 rollout 全部答对；
- reward 无法区分 rollout；
- 当前 checkpoint 下没有 on-policy preference signal。

因此，zero-signal 只表示：

> 当前 checkpoint 下没有可用于 GRPO 更新的区分性梯度信号。

处理规则：

1. Dense 和 Token-TopK 分别计算 zero flag；
2. zero-signal samples 不进入 PCA/KMeans 主空间；
3. 放入独立 silent bucket；
4. silent bucket 默认按目标 retention rate 随机保留；
5. 单独报告 silent bucket 的 reward、长度、pass rate 和数据来源分布。

---

## 10. 阶段七：Representation 保存与 streaming 实现

### 10.1 Dense 版本保存

Dense `4F` 使用完整 float32 保存：

```text
dense_grad_feat_vals: list<float32>, length exactly 4F
```

当前 `F=130,915` 时，一个 float32 `4F` 向量约 2.0 MiB。103,022 samples 约需要 201 GiB，未计 Parquet metadata 和压缩效果。

### 10.2 Token-TopK 稀疏保存

Token-TopK 的数学表示是 `4F`，但不写出确定为 0 的位置。参考 activation extraction 的 prompt/solution sparse aggregation 形式，保存：

```text
topk_prompt_feat_ids: list<int64>
topk_prompt_feat_mean: list<float32>
topk_prompt_feat_signed_peak: list<float32>

topk_response_feat_ids: list<int64>
topk_response_feat_mean: list<float32>
topk_response_feat_signed_peak: list<float32>
```

Prompt IDs 是 `U_i,prompt`，response IDs 是 `U_i,response`，都必须严格递增、去重并落在 `[0,F)`。

恢复逻辑：

```python
def materialize_topk_4f(row, F):
    out = torch.zeros(4 * F, dtype=torch.float32)
    pids = torch.as_tensor(row["topk_prompt_feat_ids"], dtype=torch.long)
    rids = torch.as_tensor(row["topk_response_feat_ids"], dtype=torch.long)

    out[pids] = torch.as_tensor(row["topk_prompt_feat_mean"])
    out[F + pids] = torch.as_tensor(row["topk_prompt_feat_signed_peak"])
    out[2 * F + rids] = torch.as_tensor(row["topk_response_feat_mean"])
    out[3 * F + rids] = torch.as_tensor(row["topk_response_feat_signed_peak"])
    return out
```

### 10.3 建议 Parquet schema

```text
sample_id: int64
representation_schema_version: int32 (=2)
sae_feature_size: int64 (=F)
gradient_feature_topk: int32 (=K)
dense_gradient_representation_size: int64 (=4F)
topk_gradient_representation_size: int64 (=4F)

dense_grad_feat_vals: list<float32>

topk_prompt_feat_ids: list<int64>
topk_prompt_feat_mean: list<float32>
topk_prompt_feat_signed_peak: list<float32>
topk_response_feat_ids: list<int64>
topk_response_feat_mean: list<float32>
topk_response_feat_signed_peak: list<float32>

prompt_gradient_token_count: int64
response_gradient_token_count: int64
topk_prompt_feature_union_size: int64
topk_response_feature_union_size: int64
```

不要继续用旧 `grad_feat_vals` 同时表示 `2F` 和 `4F`，否则下游容易发生 silent schema mismatch。V2 使用新字段名和 schema version；兼容旧脚本时另写显式 conversion 工具。

### 10.4 Streaming accumulator

每个 span 的 Dense 分支维护：

```text
sum[F]
signed_peak[F]
token_count
```

Token-TopK 分支第一版维护：

```text
sum[F]           float32
signed_peak[F]   float32
support[F]       bool
token_count      int64
```

只将每个 token 的 K 个 `(id,signed_value)` scatter 到 accumulator：

- `sum.scatter_add_` 更新总和；
- signed peak 按绝对值更新并保留符号；
- `support[id]=True`；
- 最终 mean 除以全部有效 token 数。

这与显式构造 zero-filled `[T,F]` 再 pooling 完全等价，但不需要创建第二份 dense token matrix。

Sample-level accumulator 对两个版本、两个 spans、两个 channels 共维护八个 rollout sums；Top-K 额外维护 prompt/response 两个 support unions。所有数值 sum 最终除以同一个 rollout count。

### 10.5 Manifest

每个 run 必须记录：

```text
representation_schema_version
decoder checkpoint and normalization
decoder row norm stats
gradient_feature_topk
token_topk_ranking = absolute_value
token_topk_value = original_signed_projection
token_topk_missing_value = zero
token_mean_denominator = all_valid_tokens_in_span
token_peak = argmax_abs_keep_sign
signed_peak_tie_break = positive
prompt/response span definitions
rollout_pool = equal_weight_arithmetic_mean_per_branch
feature_union = token_union_then_rollout_union_per_span
4F layout
physical storage format
```

这些字段必须加入 shard merge provenance 校验。

---

## 11. 阶段八：高维表示压缩

两个 representation 版本分别处理、分别拟合预处理器、分别执行 selection。除专门消融外，不拼成 `8F`。

### 11.1 分支 A：CountSketch 工程基线

使用 streaming CountSketch 将 `4F` 映射到 1024 维：

```text
prompt mean:          256
prompt signed peak:   256
response mean:        256
response signed peak: 256
```

四个 channels 使用独立 hash/sign spaces，固定随机种子。

CountSketch 适合：

- 快速验证表示是否有聚类结构；
- cosine dedup baseline；
- pilot KMeans；
- 为全局 SAE feature selection 提供 pseudo-clusters。

缺点是 sketch 坐标不能映射回真实 SAE feature，因此不作为最终唯一表示。

### 11.2 分支 B：选择全局共享 SAE features

不能让每个 sample 在下游单独选择一套 feature coordinates。Token-TopK sample supports 可以不同，但 feature IDs 始终位于同一个全局 SAE 坐标系；最终用于 dense selection matrix 的 feature subset 必须全数据共享。

#### 第一遍：Pilot 数据

从非 silent samples 中固定随机选择：

```text
10,000–20,000 samples
```

使用 CountSketch representation 聚类。对每个 SAE feature `f`，将四个 channels 看作一个 group：

$$
v_{if}
=
(m^{prompt}_{if},p^{prompt}_{if},m^{response}_{if},p^{response}_{if}).
$$

根据 pilot cluster 标签计算 group Fisher score：

$$
B_f
=
\sum_c n_c\lVert\mu_{cf}-\mu_f\rVert_2^2,
$$

$$
W_f
=
\sum_i\lVert v_{if}-\mu_{c(i)f}\rVert_2^2,
$$

$$
J_f=\frac{B_f}{W_f+\epsilon}.
$$

同时保存 feature prevalence、正负 prevalence、各 channel robust magnitude 和 decoder cosine redundancy。

Token-TopK 版本先过滤 prevalence 极低的 features，避免极少数异常 sample 的大值占满候选。

先选择 Fisher score 前 4096 个 candidates，再对高度相关的 decoder directions 或 sample coordinates 去冗余，最终保留：

```text
K_global = 1024 / 2048 / 4096
default K_global = 2048
```

#### 第二遍：构造全量共享坐标表示

从四个 channels 中抽取同一组 `K_global` features：

$$
X^v_{global}
\in
\mathbb R^{N\times 4K_{global}}.
$$

默认：

```text
4 × 2048 = 8192 dimensions
```

Dense 从完整 `4F` 中抽取列；Token-TopK 直接从 sparse IDs/values 构造 CSR/COO matrix，再抽取共享 features，不先 materialize 全 `4F`。

---

## 12. 阶段九：表示标准化

### 12.1 四个 channels 分别 robust scaling

Prompt/response 和 mean/peak 的尺度可能明显不同。对每个版本、每个 channel、每个共享 feature 单独拟合：

$$
s_{bf}=Q_{0.75}(|x_{\cdot,b,f}|)+\epsilon.
$$

然后：

$$
\tilde x_{i,b,f}
=
\operatorname{clip}
\left(
\frac{x_{i,b,f}}{s_{bf}},-10,10
\right).
$$

Token-TopK 大量位置为 0。如果全体 `Q0.75=0`，使用非零样本上的 `Q0.75`，同时保留 prevalence 统计。

### 12.2 PCA-whitening

```text
input dimension:  4K_global = 8192
output dimension: 128 / 256 / 512
default:          256
whitening:        enabled
```

SAE decoder 过完备且非正交。Whitening 可以降低相关 feature 被重复计算的问题。

### 12.3 Row L2 normalization

主 selection 比较 gradient pattern：

$$
y_i^v
=
\frac{\operatorname{PCAWhiten}(\tilde x_i^v)}
{\lVert\operatorname{PCAWhiten}(\tilde x_i^v)\rVert_2+\epsilon}.
$$

使用 cosine distance：

$$
d(i,j)=1-(y_i^v)^\top y_j^v.
$$

原始 gradient norm、representation norm 和四个 channel norms 单独保存，不参与主要 cosine distance。

两个 representation 版本必须分别拟合 feature set、scales 和 PCA，不共享已经由另一版本数据拟合的预处理器。

---

## 13. 方法一：Cosine Threshold Dedup

### 13.1 目标

直接删除 gradient-SAE representation 极其相似的 samples，验证“距离很近的 sample 是局部更新近重复”的假设。

### 13.2 执行步骤

使用 FAISS：

```text
index:          IndexFlatIP / IVF-PQ
metric:         inner product
input:          L2-normalized y_i
top neighbors:  50
```

保留满足：

$$
\cos(y_i,y_j)\ge\tau
$$

的边。

推荐阈值：

```text
tau = 0.95 / 0.97 / 0.98 / 0.99
default = 0.98
```

不要直接使用 connected components，因为相似关系可能 chaining：

```text
A 与 B 相似
B 与 C 相似
但 A 与 C 不相似
```

采用 greedy radius dedup：

1. 计算每个 sample 在阈值内的邻居数；
2. 选择局部密度最高的 sample 作为 representative；
3. 保留 representative；
4. 删除其阈值内邻居；
5. 在剩余 samples 中重复。

输出：

```text
selected_sample_id
removed_sample_id
representative_id
cosine_to_representative
local_density
```

该方法解释性强，但 retention 对阈值敏感，定位为基础 baseline。

---

## 14. 方法二：Cluster Quota + Medoid + Farthest-First

### 14.1 目标

在删除相似 samples 的同时，确保不同 gradient behavior modes 都有保留，并精确控制 subset size。这是默认主方法。

### 14.2 MiniBatchKMeans

输入：

```text
256-dimensional whitened and L2-normalized representations
```

推荐：

```text
K_cluster = 512 / 1024 / 2048
default = 1024
```

### 14.3 Cluster quota

设总共保留 `M` 个非 silent samples，cluster `c` 有 `n_c` 个 samples：

$$
M_c
=
M\frac{n_c^\alpha}{\sum_k n_k^\alpha},
\qquad
\alpha=0.5.
$$

设置：

```text
minimum quota per non-empty cluster = 1
```

使用 largest-remainder rounding 保证总数准确；quota 不超过 cluster size，溢出部分按剩余容量重新分配。

- `alpha=1`：严格按 cluster size；
- `alpha=0`：所有 cluster 等额；
- `alpha=0.5`：兼顾频率和稀有 gradient modes。

### 14.4 Medoid 初始化

使用离 centroid 最近的真实 sample 近似 medoid：

$$
i_c^{(1)}
=
\arg\min_{i:c(i)=c}\lVert y_i-\mu_c\rVert_2.
$$

### 14.5 Farthest-first 补充

初始化：

$$
S_c=\{i_c^{(1)}\}.
$$

重复选择：

$$
i^*
=
\arg\max_{i:c(i)=c}
\min_{j\in S_c}(1-y_i^\top y_j),
$$

直到 `|S_c|=M_c`。

合并所有 clusters 后，如果数量不足，在全局未选 samples 中按 nearest-selected distance 最大者补齐。

该方法可以精确控制 25%、50%、75% retention，同时保留典型和边缘 modes，是第一推荐主方法。

---

## 15. 方法三：Hierarchical K-Center

### 15.1 目标

近似优化最差覆盖距离：

$$
\min_{|S|=M}
\max_i\min_{j\in S}(1-y_i^\top y_j).
$$

全局 farthest-first 复杂度约为 `O(NM)`，对 103K samples 和大 retention 过高，因此使用 hierarchical K-center。

### 15.2 执行步骤

1. 使用约 2048 个 coarse clusters；
2. 按 `n_c^0.5` 分配 quota；
3. 每个 cluster 从 nearest-to-centroid sample 开始；
4. 保存每个候选到当前 selected set 的最小距离；
5. 每次选择最小距离最大的 sample；
6. 合并各 clusters；
7. 根据全局 nearest-selected distance 做补充或替换。

伪代码：

```python
selected = [nearest_to_centroid]
min_dist = cosine_distance(points, selected[0])

while len(selected) < quota:
    next_idx = min_dist.argmax()
    selected.append(next_idx)
    min_dist = minimum(
        min_dist,
        cosine_distance(points, next_idx),
    )
```

### 15.3 离群点防护

只过滤明确数值异常：

```text
non-finite representation
non-finite gradient norm
PCA Mahalanobis distance > 99.9 percentile
extremely low rollout consistency together with extreme norm
```

不能仅因 gradient norm 大或 reward 低而删除 sample。

K-center 直接优化 coverage，不需要 cosine threshold，定位为第二主方法。

---

## 16. 方法四：Cluster-wise Log-det Selection

### 16.1 目标

选择尽量线性独立、能够张成不同 gradient update directions 的 samples：

$$
S^*
=
\arg\max_{|S|=M}
\log\det
\left(
\lambda I+\sum_{i\in S}y_i y_i^\top
\right).
$$

### 16.2 执行步骤

```text
PCA-whiten dimension: 128 or 256, default 128
clusters:             512 or 1024
quota:                proportional to n_c^0.5
candidate count:      min(cluster_size, 10 × cluster_quota)
lambda:               0.01 / 0.1 / 1.0, default 0.1
```

候选由 medoid、farthest-first samples 和固定种子随机 samples 组成。

设：

$$
A_S=\lambda I+\sum_{j\in S}y_jy_j^\top.
$$

加入 sample `i` 的边际收益：

$$
\Delta_i
=
\log(1+y_i^\top A_S^{-1}y_i).
$$

使用 lazy greedy 和 Cholesky rank-one update。

该方法适合 10%–30% retention。对 50%–75% retention，更适合作为 cluster 内 tie-break，而不是全量运行。

---

## 17. 方法五：Gradient Matching Baseline

### 17.1 目标

选择一个带非负权重的子集，使其总体 representation 接近全集：

$$
g_{full}=\frac1N\sum_{i=1}^{N}u_i,
$$

$$
\min_{S,w_i\ge0}
\left\lVert
g_{full}-\sum_{i\in S}w_i u_i
\right\rVert_2^2.
$$

### 17.2 Representation 定义

需要区分：

1. **Representation matching**：`u_i` 使用 robust-scaled 但未做 row L2 normalization 的 `4K_global` 表示；
2. **True gradient-sum matching**：提取时额外保存 prompt/response token projection sum 的线性 CountSketch。

不能从已经 rollout-mean 的 response mean 和平均 response length 精确恢复 true gradient sum，因为 rollout 长度与 gradient 相关。True gradient-sum sketch 是额外诊断产物，不改变两个 canonical `4F` representations。

### 17.3 贪心求解

初始化 residual：

$$
r_0=g_{full}.
$$

每轮选择：

$$
i^*
=
\arg\max_i
\frac{u_i^\top r}{\lVert u_i\rVert_2+\epsilon}.
$$

对当前 selected samples 解非负最小二乘，更新 residual，直到达到目标 subset size。

大规模实现先聚成 512–1024 clusters，在 cluster 内 matching，再对合并候选做全局权重拟合。

Gradient matching 用于检验总体信号保持，不建议作为唯一主方法，因为全集平均可能因正负方向互相抵消而很小。

---

## 18. 推荐的第一轮实验矩阵

### 18.1 Representation 对照

| 版本 | Token feature 处理 | Span | Token pooling | Rollout pooling | 逻辑维度 |
| --- | --- | --- | --- | --- | --- |
| Dense | 全部 F 个 signed projections | prompt/response 分开 | mean + signed peak | mean | 4F |
| Token-TopK | 每 token 按 abs 选 K 并保留符号 | prompt/response 分开 | mean + signed peak | mean | 4F sparse |

两个版本使用相同 model、SAE、rollouts、reward、gradient、preprocessing capacity 和 selection budget。

### 18.2 Selection 实验

| 实验 | Representation | Preprocess | Selection | Retention |
| --- | --- | --- | --- | --- |
| R | 无 | 无 | fixed-seed random | 25/50/75% |
| D1 | dense | CountSketch 1024 + row L2 | cosine dedup | threshold sweep |
| T1 | token_topk | CountSketch 1024 + row L2 | cosine dedup | threshold sweep |
| D2 | dense | global 2048 features + PCA 256 | cluster medoid + farthest-first | 25/50/75% |
| T2 | token_topk | global 2048 features + PCA 256 | cluster medoid + farthest-first | 25/50/75% |
| D3 | dense | global 2048 features + PCA 256 | hierarchical K-center | 25/50% |
| T3 | token_topk | global 2048 features + PCA 256 | hierarchical K-center | 25/50% |

资源允许后增加 log-det 和 gradient matching。

### 18.3 必要消融

```text
gradient_feature_topk: 64 / 128 / 256
token pooling: mean only vs mean + signed peak
span: response only vs prompt + response
decoder: checkpoint_as_is vs runtime_l2_normalized
PCA dimension: 128 / 256 / 512
K_global: 1024 / 2048 / 4096
quota alpha: 0 / 0.5 / 1
```

普通 `amax` 只作为明确标记的语义对照，不与 signed peak 主结果混写。

---

## 19. Selection 结果的离线验收指标

### 19.1 Nearest-selected coverage

对每个非 silent sample：

$$
c_i=\max_{j\in S}\cos(y_i,y_j).
$$

报告：

```text
mean
median
p05
p01
minimum
```

### 19.2 Duplicate compression

报告 selection 前后：

```text
cosine >= 0.95 pair count
cosine >= 0.98 pair count
mean local density
largest duplicate neighborhood
```

### 19.3 Cluster retention

每个 cluster 报告：

```text
original count
selected count
retention ratio
mean nearest-selected coverage
```

### 19.4 四通道 signed feature coverage

分别统计：

```text
prompt mean positive/negative coverage
prompt signed-peak positive/negative coverage
response mean positive/negative coverage
response signed-peak positive/negative coverage
```

相反符号代表潜在更新冲突，不能当作重复合并。

### 19.5 Token-TopK support coverage

```text
selected subset feature-union recall
weighted feature prevalence error
rare feature retention
prompt/response union-size distribution drift
```

### 19.6 数据分布保持

比较 full 与 selected 的：

```text
source
difficulty
prompt length
response length
reward
pass rate
zero-signal status
```

Representation selection 不能在不知情的情况下退化为 difficulty 或长度筛选。

### 19.7 Gradient matching error

若运行 matching，报告：

$$
E
=
\frac{\lVert g_{full}-g_{subset}\rVert_2}
{\lVert g_{full}\rVert_2+\epsilon}.
$$

同时仍需报告 coverage；低 reconstruction error 不代表稀有 gradient modes 得到保留。

---

## 20. 最终训练实验

### 20.1 对照组

```text
Full 103K
Random 25/50/75%
Dense + cosine dedup
Token-TopK + cosine dedup
Dense + cluster medoid/farthest-first
Token-TopK + cluster medoid/farthest-first
Dense + hierarchical K-center
Token-TopK + hierarchical K-center
```

### 20.2 公平性要求

所有方法固定：

```text
base model
rollout configuration
optimizer
learning rate
global/micro batch sizes
reward function
evaluation frequency
curriculum method
random seeds
```

分别比较：

- **Equal-step**：所有方法训练相同步数，衡量数据效率；
- **Equal-epoch**：每种方法对自己的数据遍历相同 epoch，衡量计算节省。

如果 selection 输出 sample weights，主结果先运行无权重训练，weighted 结果单独报告。

### 20.3 主要评价指标

```text
average math accuracy
AIME pass@8
other math benchmark mean@8
steps to fixed accuracy
wall-clock time
training stability
gradient norm
KL
reward
response length
```

至少运行 3 个 training seeds，报告 mean、standard deviation 和 paired differences。

---

## 21. 当前代码核对与实现修改

### 21.1 当前活跃实现

当前 recipe [`run_qwen3_1_7b_deepmath.sh`](./run_qwen3_1_7b_deepmath.sh) 通过普通 VERL PPO/GRPO 入口启用 `static_gradient_extraction`，活跃实现为：

```text
verl/workers/actor/static_gradient_sae.py
```

`SAERL-0801/verl_gradient_sae/` 和 `SAERL-0801/code/pipeline/selection/10_extract_static_rollout_gradient_sae.py` 是历史或替代实现，不是当前 recipe 的直接路径。

### 21.2 代码行为确认

| 项目 | 当前代码行为 | 本方案要求 |
| --- | --- | --- |
| token projection | Dense `gradient @ W_dec.T` | 同一次 projection 产生 Dense 与 Token-TopK |
| token mean | 每个 rollout 内先 mean | 保留，并按 prompt/response 分开 |
| token max | 普通 `projected.amax(dim=0)` | 改为 argmax-abs keep-sign |
| rollout pooling | 各分支求和后除以 rollout count | 已经是等权 rollout mean |
| prompt/response | attention tokens 合并 pooling | 必须分开 |
| feature union | Dense 全 F，不取并集 | Token-TopK 做 token union 和 rollout union |
| representation | 单个 `2F` dense vector | 两个版本各自逻辑 `4F` |
| decoder normalization | 直接使用 checkpoint `W_dec` | 显式配置、校验并记录 |

因此，对“当前代码是否 rollout 取平均、features 取并集”的结论是：

- **rollout 已经取等权平均**；
- **features 当前没有取并集**，因为当前 gradient representation 是 Dense 全 feature；
- **当前 max 是普通数值最大值，不是所需 signed peak**；
- **当前 prompt 和 response 没有分开 pooling**。

### 21.3 Activation 参考代码确认

[`02_extract_activations_parquet_dir.py`](../../SAERL-0801/code/pipeline/02_extract_activations_parquet_dir.py) 调用：

```text
SAERL-0801/code/SAE-RL/src/sae_rl/sae_analysis/extract.py
```

其中 `_aggregate_sparse_span`：

1. 展平一个 prompt 或 solution span 内所有 token 的 sparse feature IDs；
2. 使用 `torch.unique(...,sorted=True)` 获得 feature ID 并集；
3. 计算 sum、count、mean 和 max；
4. prompt 与 solution 分开保存。

可直接参考的是：

- prompt/response 分开；
- sparse feature IDs 取并集；
- mean 使用 `sum / token_count`，不是 `sum / occurrence_count`；
- IDs 与多个统计 values 对齐保存。

不能直接照搬的是：

- 该代码没有 rollout 维度，不做 rollout mean；
- activation 通常非负，使用普通 max；signed gradient 必须改为 signed peak。

### 21.4 主要代码修改

`verl/workers/actor/static_gradient_sae.py`：

1. 升级 `gradient_sae_schema`；
2. 在 `SaeDecoderProjector` 中加入 decoder normalization/validation；
3. 将普通 max 改为 signed peak；
4. 同一次 projected chunk 更新 Dense 和 Token-TopK；
5. 在 `consume_micro_batch` 中按 `prompt_len/response_len` 切两个 spans；
6. 对八个数值分支分别做 rollout sum/mean；
7. 对 Token-TopK prompt/response supports 分别做 rollout union；
8. 输出 Dense `4F` 和 sparse-stored Token-TopK `4F`。

`verl/trainer/config/ppo_trainer.yaml`：

```yaml
static_gradient_extraction:
  representation_schema_version: 2
  gradient_token_scope: attention
  gradient_feature_topk: 128
  decoder_normalization: runtime_l2_normalized
  signed_peak_tie_break: positive
  topk_storage: sparse_union
```

Recipe 运行脚本显式覆盖关键参数，不能依赖未记录的默认值。

下游 merge/reader/materialization：

- 按 schema version fail fast；
- 显式选择 `dense` 或 `token_topk`；
- Token-TopK 使用 sparse IDs/values；
- 不自动拼为 `8F`；
- 将 representation provenance 加入 shard merge 校验。

---

## 22. 测试与验收

### 22.1 单元测试

1. **Signed peak 强负值**：`[-10,-0.1] -> -10`；
2. **Signed peak tie**：`[-3,+3] -> +3`；
3. **Token Top-K**：按绝对值选择、强负值可入选、保存值保留负号；
4. **Top-K mean 分母**：10 tokens 中一次值 5，mean 必须为 0.5；
5. **Token feature union**：IDs 排序、去重、values 对齐；
6. **Rollout support union**：正负 rollout 抵消为 0 时 ID 仍在 support；
7. **Prompt/response split**：没有跨 boundary token；
8. **Rollout mean**：等于 `stack(per_rollout).mean(0)`，不按 response length 加权；
9. **K=F 等价性**：Token-TopK 恢复的 `4F` 与 Dense 完全一致；
10. **Sparse round-trip**：Parquet sparse fields 无损恢复逻辑 `4F`。

### 22.2 集成测试

使用 toy SAE、小 feature size、2 samples、每个 2 rollouts：

1. 跑完整 rollout → reward → GRPO backward → projection → Parquet；
2. Dense 长度严格为 `4F`；
3. Token-TopK 恢复后长度严格为 `4F`；
4. prompt/response token counts 正确；
5. rollout count 正确；
6. optimizer 不 step，模型参数前后完全一致；
7. 各 rank representation provenance 一致；
8. merge 后 sample IDs 唯一、行数正确。

### 22.3 小规模真实数据验收

先运行 8–32 个真实 samples，报告：

```text
finite ratio
prompt/response zero-vector ratio
Top-K prompt/response union-size distribution
positive/negative signed-peak ratio
rollout cancellation ratio
Dense vs Token-TopK cosine distribution
Parquet bytes per sample
projection tokens/s
```

确认语义、吞吐和存储规模后再启动 103K 全量提取。

---

## 23. 方法优先级与最终执行流水线

### 23.1 方法优先级

| 优先级 | 方法 | 建议定位 |
| ---: | --- | --- |
| 1 | Cluster quota + medoid + farthest-first | 默认主方法 |
| 2 | Cosine threshold dedup | 基础 baseline |
| 3 | Hierarchical K-center | 第二主方法 |
| 4 | Cluster-wise log-det | 小 retention 或 cluster 内增强 |
| 5 | Gradient matching | 理论 baseline |

两个 representation 版本必须使用相同超参数搜索预算。

### 23.2 最终执行流水线

```text
1. 固定 model checkpoint、SAE、GRPO 和 rollout 配置
2. 加载并归一化 SAE decoder
3. 提取每个 rollout 的目标层 token gradient
4. 去 padding，并切分 prompt/response
5. 每个 token chunk 执行一次 gradient × decoder.T
6. Dense 分支更新 mean + signed peak
7. Token-TopK 分支按 abs 选 K、保留符号并更新 sparse stats
8. 每个 rollout 分别完成 prompt/response pooling
9. 对 R 个 rollouts 的八个数值分支做等权 mean
10. 对 Token-TopK 做 token support union 和 rollout support union
11. 写 Dense 4F 与 sparse-stored Token-TopK 4F
12. 将 zero-signal samples 放入 silent bucket
13. 分版本构造 CountSketch pilot representation
14. 分版本选择全局共享的 2048 SAE features
15. 四个 channels 分别 robust scale
16. PCA-whiten 到 256 维并 row L2 normalize
17. 分版本运行 cosine dedup、cluster farthest-first 和 K-center
18. 验收 coverage、duplicate compression、signed feature coverage 和分布漂移
19. 生成 selected sample IDs，并继续现有 curriculum
20. 运行 equal-step、equal-epoch、多 seed RL 训练
21. 比较 Dense 与 Token-TopK 的最终效果和端到端成本
```

---

## 24. 与 V1 的简要区别

1. 从单一 representation 扩展为 Dense 和 Token-TopK 两个版本；
2. 两个版本都将 prompt/response 分开，每个 sample 从 `2F` 改为 `4F`；
3. 新增 per-token absolute Top-K，保留原符号，其余位置为 0；
4. 新增 token feature union 和 rollout feature union；
5. Token-TopK 使用 sparse Parquet 无损保存逻辑 `4F`；
6. 明确 max pooling 为 `argmax(abs(value))` 后保留符号；
7. 下游全局 feature 将四个 channels 作为一组，选择后维度为 `4K_global`；
8. 两个版本分别 preprocessing、selection 和训练评估，不默认拼成 `8F`。
