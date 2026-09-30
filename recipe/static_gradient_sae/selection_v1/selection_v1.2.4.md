# Gradient-SAE Absolute Response-Mean Redundancy Selection (v1.2.4)

## 1. Sample 表示

直接使用 sample-level Token-TopK `response_mean`：

$$
m_i\in\mathbb{R}^{F}.
$$

未出现在 sample support 中的 feature 为 0。对所有 feature 逐元素取绝对值，定义：

$$
x_i=|m_i|\in\mathbb{R}_{\ge0}^{F}.
$$

本方法不使用 prompt、Dense representation 或 `response_signed_peak`，也不进行全局 feature selection、
feature scaling、centering、PCA、whitening 或 sample-level 二次 Top-K。

## 2. Signal strength 与 pattern

定义：

$$
M_i=\lVert x_i\rVert_2,
\qquad
u_i=\frac{x_i}{M_i}\quad(M_i>0).
$$

由于 $\lVert|m_i|\rVert_2=\lVert m_i\rVert_2$，v1.2.4 与 v1.2 的 signal strength、silent set
和处理顺序完全相同。若 `zero_signal_token_topk=True` 或 $M_i\le\epsilon$，sample 进入 silent set。

## 3. 冗余判定

两个 active samples 的相似度为绝对值 response-mean pattern 的 cosine：

$$
s_{ij}=u_i^\top u_j
=\cos(|m_i|,|m_j|),
\qquad 0\le s_{ij}\le1.
$$

这里是先对每个 feature value 取绝对值，再计算普通 cosine；不是计算
$|\cos(m_i,m_j)|$。

按 $M_i$ 降序、`sample_id` 升序处理，并维护 selected set $S$：

- 若 $S=\varnothing$ 或 $\max_{j\in S}s_{ij}<\tau$，保留 $i$；
- 否则将 $i$ 标记为 redundant，并记录最大相似度对应的 representative。

即：

$$
\boxed{
M_j\ge M_i
\;\land\;
\cos(|m_i|,|m_j|)\ge\tau
\Longrightarrow
i\text{ is redundant}
}
$$

## 4. 流程与输出

$$
\text{Token-TopK response mean}
\rightarrow
\text{element-wise absolute value}
\rightarrow
\text{L2 normalization}
\rightarrow
\text{cosine redundancy pruning}.
$$

绝对值 sparse representation 只在内存中构建。输出仍为 selected IDs、silent IDs、逐样本 audit
和 selection manifest。

```bash
python recipe/static_gradient_sae/selection_v1/select_topk_response_mean_abs_v1_2_4.py \
  --run-dir /path/to/extraction-run \
  --cosine-threshold 0.8
```

