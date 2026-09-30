#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# export VLLM_ATTENTION_BACKEND=XFORMERS
# export CUDA_VISIBLE_DEVICES=0,1,2,3

export WANDB_MODE=offline
export WANDB_DIR=/home/fit/lijuanzi1/WORK/dz/SAERL/wandb

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/sorted_deepmath/train.parquet" \
    'data.val_files=["/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/AIME_2024/train.parquet", "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/math500/test.parquet"]' \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path="/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-7B"  \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepmath_difficulty_Qwen2.5-Math-7B' \
    trainer.experiment_name='sel_v1_plus_silent_v4mix8' \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}" \
    trainer.n_gpus_per_node=4 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="/WORK/PUBLIC/lijuanzi1_work/dz/ckpt/Qwen2.5-Math-7B/sorted_deepmath128/sel_v1_plus_silent_v4mix8/" \
    data.truncation='error' \
    data.shuffle=False \
    data.saerl.enable=enable \
    data.saerl.processed_file="/home/fit/lijuanzi1/WORK/dz/SAERL/rank_sorted/Qwen2.5-Math-1.5B/sel/sorted_deepmath_sel_v1_thres07_plus_silent_v4mix8.pt"