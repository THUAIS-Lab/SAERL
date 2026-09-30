# gsm8k
python evaluation/eval_single_ckpt.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/gsm8k/test.parquet"  \
    --dataset "gsm8k" \
    --p "/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-7B/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-7B/step0/run1/"


# aime
python evaluation/eval_single_ckpt.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/AIME_2024/train.parquet" \
    --dataset "aime" \
    --p "/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-7B/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-7B/step0/run1/"

# amc eval
python evaluation/eval_single_ckpt.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/amc23/train.parquet" \
    --dataset "amc" \
    --p "/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-7B/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-7B/step0/run1/"


# math
python evaluation/eval_single_ckpt.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/math500/test.parquet" \
    --dataset "math" \
    --p "/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-7B/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-7B/step0/run1/"



# olympiad
python evaluation/eval_single_ckpt.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/olympiad_bench/train_1024.parquet" \
    --dataset "olympiad" \
    --p "/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-7B/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-7B/step0/run1/"

# minerva
python evaluation/eval_single_ckpt.py \
    --eval_dataset_path "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/minerva/test.parquet"  \
    --dataset "minerva" \
    --p "/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-7B/" \
    --result_dir "/WORK/PUBLIC/lijuanzi1_work/dz/result/Qwen2.5-Math-7B/step0/run1/"

