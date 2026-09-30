# Gradient-SAE Functional Redundancy Selection（v1.3）

## 1. Sample 表示

只读取 sample-level Dense representation 的 response mean：

$$
m_i=x_i^{\mathrm{dense}}[2F:3F]\in\mathbb{R}^{F}.
$$

对每个 sample 按 feature 绝对值取 Top-$K$（默认 $K=128$；边界同值时优先较小的
feature id）：

$$
I_i=\operatorname{TopK}_f(|m_{if}|,K),
\qquad
z_{if}=m_{if}\mathbf 1[f\in I_i].
$$

本版本不使用 prompt、`response_signed_peak` 或 extraction 的 Token-TopK representation。
只有 response mean 一个 branch，因此全局标量校准不会改变 cosine 或强度排序，直接省略。

## 2. Signal Strength 与 Pattern

所有后续计算都基于截断后的 $z_i$：

$$
M_i=\lVert z_i\rVert_2,
\qquad
u_i=\frac{z_i}{M_i}.
$$

`zero_signal_dense=true` 或 $M_i\le\epsilon$ 的 sample 进入 silent set。其余 sample 按
$M_i$ 降序、`sample_id` 升序处理。

## 3. Signed Cosine 去冗余

候选 sample $i$ 与已选 sample $j$ 的相似度为

$$
s_{ij}=u_i^\top u_j=\cos(z_i,z_j).
$$

必须保留符号，不使用绝对 cosine。令

$$
r_i=\max_{j\in S}s_{ij}.
$$

- 若 $r_i\ge\tau$，则 $i$ 冗余，并记录对应的 `representative_sample_id`；
- 否则保留 $i$，加入 $S$。

因此完整流程为

$$
\text{Dense response mean}
\rightarrow
\text{sample-level abs-TopK}
\rightarrow
\text{L2 normalization}
\rightarrow
\text{signed-cosine pruning}.
$$

这里的 Top-K 只限制单样本非零数，不保证所有 samples 的 feature union 不超过 $K$。

## 4. 运行

```bash
python recipe/static_gradient_sae/selection_v1/select_v1_3.py \
  --run-dir /path/to/extraction-run \
  --top-k 128 \
  --cosine-threshold 0.98
```

结果默认写入 `RUN_DIR/selection_v1_3/`。
