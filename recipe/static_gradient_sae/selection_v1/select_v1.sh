# topk response mean
# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/select_v1_2.py \
#     --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461 \
#     --cosine-threshold 0.5


# dense response mean
# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/selection_v1/select_dense_response_mean_v1_2.py \
#         --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461 \
#         --cosine-threshold 0.8


# # topk response mean + peak
# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/selection_v1/select_topk_response_mean_peak_v1.py \
#   --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461 \
#   --cosine-threshold 0.8

# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/selection_v1/select_v1_3.py \
#   --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461 \
#   --top-k 128 \
#   --cosine-threshold 0.9

# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/selection_v1/select_v1_4.py \
#   --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461 \
#   --feature-threshold 1e-9 \
#   --cosine-threshold 0.9

# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/selection_v1/select_topk_response_mean_abs_selected_features_v1_2_3.py \
#   --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461 \
#   --feature-indices /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260815-031029-164461/feature_selection/energy_feature_indices.npy \
#   --cosine-threshold 0.7

# python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/selection_v1/select_topk_response_mean_abs_v1_2_4.py \
#   --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813 \
#   --cosine-threshold 0.7

# final select v1
python /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/selection_v1/select_topk_response_mean_abs_selected_features_v1_2_3.py \
  --run-dir /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813 \
  --feature-indices /home/fit/lijuanzi1/WORK/dz/SAERL/SAERL-0801/runs/selection/qwen2_5/20260817-022813-076813/feature_selection/variance_feature_indices.npy \
  --cosine-threshold 0.7 \
  --output-dir /home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae/selection_v1/test

