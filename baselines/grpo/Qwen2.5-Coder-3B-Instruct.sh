bash /home/dz/SAERL/grpo_baseline/grpo.sh \
    max_prompt_length=2048 \
    max_response_length=16384 \
    train_batch_size=512 \
    ppo_mini_batch_size=64 $@