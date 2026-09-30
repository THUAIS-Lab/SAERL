export CUDA_VISIBLE_DEVICES=6,7
# export VLLM_USE_V1=1

bash /home/dz/SAERL/grpo_baseline/grpo.sh \
    data.max_response_length=3000 \
    data.train_files="/home/dz/SAERL/data/gsm8k/train.parquet" \
    data.val_files="/home/dz/SAERL/data/gsm8k/test.parquet" \
    actor_rollout_ref.model.path="/data0/dz/models/Qwen2.5-Math-1.5B-Instruct" \
    trainer.project_name='gainrl' \
    trainer.experiment_name='Qwen2.5-Math-1.5B-Instruct_gsm8k_GRPO' \
    trainer.n_gpus_per_node=2 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="/data0/dz/ckpt/gainrl/Qwen2.5-Math-1.5B-Instruct_gsm8k_GRPO"