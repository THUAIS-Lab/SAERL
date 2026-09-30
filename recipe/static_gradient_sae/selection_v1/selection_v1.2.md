# Gradient-SAE Functional Redundancy Selection (v1.2)

## 1. Sample 表示

直接使用 `plan_v2.md` 生成的 sample-level **Token-TopK response mean representation**。对 sample $i$：

$$
m_i=\texttt{response\_mean}\in\mathbb{R}^{F},
\qquad F=130{,}915.
$$

未出现在 sample-level support 中的 feature 视为 0。Selection representation 定义为：

$$
x_i=m_i.
$$

本版本不使用 `response_signed_peak`。由于只有一个分支，全局标量校准不会改变 cosine similarity 或 signal strength 的排序，因此不再进行 branch scaling。

## 2. Signal Strength 与 Pattern

对非 zero-signal sample，定义：

$$
M_i=\lVert m_i\rVert_2,
\qquad
u_i=\frac{m_i}{M_i}.
$$

- $M_i$：Gradient-SAE signal strength，用于确定保留优先级；
- $u_i$：normalized Gradient-SAE pattern，用于判断功能冗余。

若 $M_i\approx0$，或 `plan_v2.md` 已将其标为 zero-signal，则将该 sample 放入独立的 silent set，不参与余弦冗余判断。Zero-signal 只表示当前 checkpoint 下没有有效的非零训练信号，不代表数据低质或无价值。

## 3. 冗余判定与选择

两个 samples 的 pattern similarity 定义为 signed cosine：

$$
s_{ij}=u_i^\top u_j=\cos(m_i,m_j),
\qquad -1\le s_{ij}\le1.
$$

仅当 $s_{ij}\ge\tau$ 时认为二者冗余。必须保留符号，不能使用 $|u_i^\top u_j|$；符号相反的 gradient pattern 不应视为重复。

对所有非 zero-signal samples 按 $M_i$ 降序排列，并维护 selected set $S$。依次处理 sample $i$：

$$
r_i=\max_{j\in S}u_i^\top u_j,
\qquad
j_i^*=\arg\max_{j\in S}u_i^\top u_j.
$$

- 若 $S=\varnothing$ 或 $r_i<\tau$，保留 $i$，令 $S\leftarrow S\cup\{i\}$；
- 若 $r_i\ge\tau$，则 $i$ 被 $j_i^*$ 表示，判为冗余并删除。

由于按 signal strength 降序处理，冗余判据可概括为：

$$
\boxed{
M_j\ge M_i
\;\land\;
u_i^\top u_j\ge\tau
\quad\Longrightarrow\quad
i\text{ is redundant}
}
$$

即：若一个 sample 的 response-mean gradient pattern 已被信号更强且高度相似的 selected sample 覆盖，则不再保留。

## 4. 阈值与输出

$\tau$ 是唯一的冗余阈值：$\tau$ 越高，判定越严格、保留越多；$\tau$ 越低，压缩越强。若需固定 retention ratio（如 25%、50%、75%），通过调整 $\tau$ 获得接近目标比例的 subset。

最终流程为：

$$
\text{response mean}
\rightarrow
\text{L2-normalized pattern}
\rightarrow
\text{signed-cosine redundancy}
\rightarrow
\text{magnitude-prioritized pruning}.
$$

输出 selected core set $S$，并将其送入后续 SAERL curriculum；silent set 单独保留，不参与本轮 selection。
