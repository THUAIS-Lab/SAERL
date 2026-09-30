# Raw-gradient response-mean selection baseline

本目录直接使用 extraction Parquet 中 `raw_grad_vals` 的 **response mean channel**，不经过 SAE
decoder representation。`raw_grad_vals` 的存储大小为 `4H`，顺序为：

```text
[prompt_mean, prompt_signed_peak, response_mean, response_signed_peak]
```

因此三个脚本都只读取：

```python
raw_grad_vals[2 * H : 3 * H]
```

所得 raw response-mean 表示为 `H` 维。feature indices 使用该 `H` 维向量的局部坐标 `[0, H)`，
不包含 Parquet 存储中的 `2H` channel offset。

## 1. 选择全局 raw response-mean 维度

`select_energy_features.py` 对每个非零 sample 的 raw response mean 做 L2 normalization，累加各维度
的平方值，再选择 normalized energy 最大的 Top-K 个共享维度。这与
`final_energy_abs_selection.md` 中的 SAE response-mean energy selection 对齐。

```bash
python recipe/static_gradient_sae/grad_baseline/select_energy_features.py \
  --run-dir /path/to/extraction-run \
  --top-k 256
```

默认输出：

```text
RUN_DIR/raw_gradient_feature_selection/response_mean_energy_feature_indices.npy
```

## 2. 在所选维度上筛选 samples

```bash
python recipe/static_gradient_sae/grad_baseline/select_raw_grad_abs_selected_features.py \
  --run-dir /path/to/extraction-run \
  --feature-indices /path/to/response_mean_energy_feature_indices.npy \
  --cosine-threshold 0.7
```

默认输出目录会包含 threshold，例如：

```text
RUN_DIR/raw_gradient_response_mean_selection_abs_selected_response_mean_energy_feature_indices_threshold_0.7
```

对 sample $i$ 的 raw response mean $m_i\in\mathbb R^H$ 和全局维度集合 $G$，该脚本使用：

```text
abs(m_i[G]) -> L2 normalization -> exact cosine greedy selection
```

样本按投影后的 response-mean L2 norm 降序处理。

## 3. 不选择维度，直接使用完整 raw response mean

```bash
python recipe/static_gradient_sae/grad_baseline/select_raw_grad_abs.py \
  --run-dir /path/to/extraction-run \
  --cosine-threshold 0.7
```

默认输出目录：

```text
RUN_DIR/raw_gradient_response_mean_selection_abs_full_threshold_0.7
```

该脚本直接使用 `abs(raw_grad_vals[2H:3H])` 的完整 `H` 维 pattern，不使用 prompt mean 或任一
signed-peak channel。

两个 sample-selection 脚本都会输出 `selected_sample_ids.parquet`、
`silent_sample_ids.parquet`、`selection_audit.parquet` 和 `selection_manifest.json`，格式与现有
gradient-SAE selection 脚本一致。只有未显式传入 `--output-dir` 时才自动追加 threshold；显式指定
输出目录时完全使用传入路径。

若 `RUN_DIR/representation_manifest.json` 存在，脚本会校验 `4H` 大小、四通道 layout，并读取
`gradient_zero_eps`；若不存在（如本目录中的单 shard 示例），则从 Parquet 推断 `4H`，要求其能被
4 整除，并默认使用 `zero_eps=1e-12`。

旧版本由完整 `4H` 生成的 `energy_feature_indices.npy` 与当前实现不兼容，必须重新运行阶段一生成
`response_mean_energy_feature_indices.npy`。
