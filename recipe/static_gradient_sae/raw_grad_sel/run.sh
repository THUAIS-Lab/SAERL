#!/usr/bin/env bash
set -euo pipefail

# # select response-mean dims (old full-4H indices must not be reused)
# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/grad_baseline/select_energy_features.py \
#   --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813 \
#   --top-k 256

# # select samples on selected dims
# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/grad_baseline/select_raw_grad_abs_selected_features.py \
#   --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813 \
#   --feature-indices /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813/raw_gradient_feature_selection/response_mean_energy_feature_indices.npy \
#   --cosine-threshold 0.727

# select samples on raw gradients
python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/raw_grad_sel/select_raw_grad_abs.py \
  --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813 \
  --cosine-threshold 0.67
