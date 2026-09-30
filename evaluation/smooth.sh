python evaluation/smooth_window.py \
    --series "GRPO:evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_grpo.json" \
    --series "DIFFICULTY:evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_difficulty_baseline.json" \
    --series "SAERL:evaluation/final_results/Qwen2.5-Math-1.5B/avg128/avg_v4mix8.json" \
    --no-markers \
    --smooth ema --ema-alpha 0.3 \
    --max-step 1000 \
    --x-tick-step 100 --y-tick-step 1 \
    --figsize 10 5 \
    --out evaluation/final_results/Qwen2.5-Math-1.5B/avg128/fig 
    # --show-raw

# python evaluation/smooth_window.py \
#     --series "GRPO:evaluation/final_results/Qwen2.5-Math-7B/avg128/avg_grpo.json" \
#     --series "DIFFICULTY:evaluation/final_results/Qwen2.5-Math-7B/avg128/avg_difficulty_baseline.json" \
#     --series "GAINRL:evaluation/final_results/Qwen2.5-Math-7B/avg128/avg_gainrl.json" \
#     --series "AdaRFT:evaluation/final_results/Qwen2.5-Math-7B/avg128/avg_adarft_grpo_100.json" \
#     --series "SAERL(w/GRPO):evaluation/final_results/Qwen2.5-Math-7B/avg128/avg_v4mix8.json" \
#     --no-markers \
#     --smooth ema --ema-alpha 0.3 \
#     --max-step 900 \
#     --x-tick-step 100 --y-tick-step 1 \
#     --figsize 10 5 \
#     --out evaluation/final_results/Qwen2.5-Math-7B/avg128/fig