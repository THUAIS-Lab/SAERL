# Gradient-SAE Absolute Selected-Feature Redundancy Selection (v1.2.3)

## 1. Absolute selected-feature 表示

输入 `.npy` feature indices：

$$
A=(a_0,a_1,\ldots,a_{K-1}),
\qquad 0\le a_k<F.
$$

对 sample $i$，先从 Token-TopK `response_mean` $m_i\in\mathbb{R}^F$ 投影，再对每一维取绝对值：

$$
x_i[k]=\left|m_i[a_k]\right|,
\qquad x_i\in\mathbb{R}_{\ge0}^{K}.
$$

第 $k$ 维对应 `.npy` 中的第 $k$ 个 feature index。未出现在 sample support 中的 feature 为 0。
本方法只使用 `response_mean`，不使用 prompt、Dense representation 或 `response_signed_peak`，也不进行
feature scaling、centering、PCA、whitening 或 sample-level 二次 Top-K。

## 2. Signal strength 与 pattern

定义：

$$
M_i=\lVert x_i\rVert_2,
\qquad
u_i=\frac{x_i}{M_i}\quad(M_i>0).
$$

由于 $\lVert|v|\rVert_2=\lVert v\rVert_2$，v1.2.3 与 v1.2.2 的 signal strength、silent set
和处理顺序完全相同。若 `zero_signal_token_topk=True`，或投影后 $M_i\le\epsilon$，sample 进入
silent set。

## 3. 冗余判定

两个 active samples 的相似度为 absolute selected-feature pattern 的 cosine：

$$
s_{ij}=u_i^\top u_j
=\cos\!\left(|m_i[A]|,|m_j[A]|\right),
\qquad 0\le s_{ij}\le1.
$$

这里先对 feature 值取绝对值，再计算普通 cosine；不是对原始 signed cosine 的结果取绝对值。

按 $M_i$ 降序、`sample_id` 升序扫描并维护 selected set $S$：

- 若 $S=\varnothing$ 或 $\max_{j\in S}s_{ij}<\tau$，保留 $i$；
- 否则将 $i$ 标记为 redundant，并记录最大相似度对应的 representative。

即：

$$
\boxed{
M_j\ge M_i
\;\land\;
\cos(|m_i[A]|,|m_j[A]|)\ge\tau
\Longrightarrow
i\text{ is redundant}
}
$$

## 4. 流程与输出

$$
\text{Token-TopK response mean}
\rightarrow
\text{selected-feature projection}
\rightarrow
\text{element-wise absolute value}
\rightarrow
\text{L2 normalization}
\rightarrow
\text{cosine redundancy pruning}.
$$

低维矩阵只在内存中构建。输出仍为 selected IDs、silent IDs、逐样本 audit 和 manifest。

```bash
python recipe/static_gradient_sae/selection_v1/select_topk_response_mean_abs_selected_features_v1_2_3.py \
  --run-dir /path/to/run \
  --feature-indices /path/to/features.npy \
  --cosine-threshold 0.8
```

