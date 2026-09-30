# gsm8k
python evaluation/eval.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/gsm8k/test.parquet"  \
    --dataset "gsm8k" \
    --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/"


# aime
python evaluation/eval.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/AIME_2024/train.parquet" \
    --dataset "aime" \
    --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/"

# amc eval
python evaluation/eval.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/amc23/train.parquet" \
    --dataset "amc" \
    --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/"


# math
python evaluation/eval.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/math500/test.parquet" \
    --dataset "math" \
    --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/"



# olympiad
python evaluation/eval.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/olympiad_bench/train_1024.parquet" \
    --dataset "olympiad" \
    --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/"

# minerva
python evaluation/eval.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/minerva/test.parquet"  \
    --dataset "minerva" \
    --p "/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-1.5B/sorted_deepmath128/irds/"

