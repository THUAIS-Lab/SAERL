export CUDA_VISIBLE_DEVICES=4,5,6,7
# export VLLM_USE_V1=1

bash /home/daiz/SAERL/grpo_baseline/grpo.sh \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=3000 \
    data.train_files="/data/daiz/data/SAERL/my_data/deepscaler/train.parquet" \
    data.val_files="/data/daiz/data/SAERL/my_data/AIME_2024/train.parquet" \
    actor_rollout_ref.model.path="/data/MODEL/Qwen3-1.7B" \
    trainer.project_name='saerl' \
    trainer.experiment_name='Qwen3-1.7B_sae_global_metric_BasicTest' \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="/data/daiz/ckpt/saerl/BasicTest/Qwen3-1.7B_sae_global_metric_BasicTest" \
    data.truncation='error' \
    data.shuffle=False \
    data.saerl.enable=enable \
    data.saerl.processed_file=/home/daiz/SAERL/angle_sorted_data/rank_deepscaler.pt