# # gsm8k
# python evaluation/eval_single_ckpt.py \
#     --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/gsm8k/test.parquet"  \
#     --dataset "gsm8k" \
#     --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/" \
#     --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/"


# aime
python evaluation/eval_single_ckpt.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/AIME_2024/train.parquet" \
    --dataset "aime" \
    --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/"

# # amc eval
# python evaluation/eval_single_ckpt.py \
#     --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/amc23/train.parquet" \
#     --dataset "amc" \
#     --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/" \
#     --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/"


# # math
# python evaluation/eval_single_ckpt.py \
#     --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/math500/test.parquet" \
#     --dataset "math" \
#     --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/" \
#     --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/"



# # olympiad
# python evaluation/eval_single_ckpt.py \
#     --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/olympiad_bench/train_1024.parquet" \
#     --dataset "olympiad" \
#     --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/" \
#     --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/"

# # minerva
# python evaluation/eval_single_ckpt.py \
#     --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/minerva/test.parquet"  \
#     --dataset "minerva" \
#     --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/" \
#     --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/emb_sel_v6r/"

