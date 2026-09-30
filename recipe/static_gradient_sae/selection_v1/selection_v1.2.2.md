# Gradient-SAE Selected-Feature Redundancy Selection (v1.2.2)

## 1. Selected-feature response mean 表示

输入一个 `.npy` 文件，其中保存互不重复的 SAE feature index：

$$
A=(a_0,a_1,\ldots,a_{K-1}),
\qquad 0\le a_k<F.
$$

对 sample $i$，从 sample-level Token-TopK `response_mean` $m_i\in\mathbb{R}^F$ 中投影出：

$$
x_i[k]=m_i[a_k],
\qquad x_i\in\mathbb{R}^{K}.
$$

第 $k$ 维始终对应 `.npy` 中的第 $k$ 个 feature index。若该 feature 未出现在 sample 的
Token-TopK response support 中，则其值为 0；命中的 `response_mean` 值保持不变。本方法不使用
`response_signed_peak`，也不进行 feature-wise scaling、centering、PCA、whitening 或 sample-level 二次 Top-K。

## 2. Signal strength 与 pattern

在 selected-feature 空间中定义：

$$
M_i=\lVert x_i\rVert_2,
\qquad
u_i=\frac{x_i}{M_i}\quad(M_i>0).
$$

- $M_i$ 用于确定保留优先级；
- $u_i$ 用于判断功能冗余。

若 `zero_signal_token_topk=True`，或投影后 $M_i\le\epsilon$，则 sample 进入 silent set，不参与余弦判重。
因此，一个在原始 $F$ 维空间中非零的 sample，也可能因所有信号均落在未选 feature 上而在本方法中变为 silent。

## 3. 冗余判定

两个 active samples 的相似度为 selected-feature 空间中的 signed cosine：

$$
s_{ij}=u_i^\top u_j=\cos(x_i,x_j).
$$

保留符号，不使用绝对值。按 $M_i$ 降序处理；若强度相同，则按 `sample_id` 升序处理。维护 selected set
$S$，对当前 sample $i$ 计算：

$$
r_i=\max_{j\in S}u_i^\top u_j.
$$

- 若 $S=\varnothing$ 或 $r_i<\tau$，保留 $i$；
- 若 $r_i\ge\tau$，则 $i$ 由最大相似度对应的 selected sample 表示，并标记为 redundant。

即：

$$
\boxed{
M_j\ge M_i
\;\land\;
\cos(x_i,x_j)\ge\tau
\Longrightarrow
i\text{ is redundant}
}
$$

## 4. 流程与输出

完整流程为：

$$
\text{Token-TopK response mean}
\rightarrow
\text{selected-feature projection}
\rightarrow
\text{L2 normalization}
\rightarrow
\text{signed-cosine redundancy pruning}.
$$

投影后的矩阵仅在内存中构建，不额外保存中间 representation 文件。输出包括：

- `selected_sample_ids.parquet`：selected core set；
- `silent_sample_ids.parquet`：原始 zero-signal 或投影后为零的 samples；
- `selection_audit.parquet`：每个 sample 的强度、状态及 redundant representative；
- `selection_manifest.json`：阈值、feature-index 文件、index hash、映射维度、计算设备和计数。

