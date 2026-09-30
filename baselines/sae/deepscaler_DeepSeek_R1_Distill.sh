#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/data/daiz/data/SAERL/my_data/deepscaler/train.parquet" \
    'data.val_files=["/data/daiz/data/SAERL/my_data/AIME_2024/train.parquet", "/data/daiz/data/SAERL/my_data/math500/test.parquet"]' \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path="/data/MODEL/DeepSeek-R1-Distill-Qwen-1.5B"  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepscaler_difficulty_DeepSeek-R1-Distill-Qwen-1.5B' \
    trainer.experiment_name='difficulty_cluster_batch_v4d_clean' \
    trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}" \
    trainer.n_gpus_per_node=4 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="/data/daiz/ckpt/saerl/DeepSeek-R1-Distill-Qwen-1.5B/deepscaler/grpo/" \
    trainer.validation_data_dir="/data/daiz/val_rollout/DeepSeek-R1-Distill-Qwen-1.5B/deepscaler/grpo/" \
    trainer.rollout_data_dir="/data/daiz/train_rollout/DeepSeek-R1-Distill-Qwen-1.5B/deepscaler/grpo/" \
    data.truncation='error' \
    data.shuffle=False \
    data.saerl.enable=enable \
    data.saerl.processed_file="/home/daiz/SAERL/angle_sorted_data/DeepSeek-R1-Distill-Qwen-1.5B/deepmath_difficulty_cluster_batch_v4d_clean_2k.pt"