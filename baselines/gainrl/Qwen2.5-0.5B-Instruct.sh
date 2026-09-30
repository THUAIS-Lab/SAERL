set -x

export CUDA_VISIBLE_DEVICES=4,5
# export VLLM_USE_V1=1
# export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/home/dz/SAERL/data/gsm8k/train.parquet" \
    data.val_files="/home/dz/SAERL/data/gsm8k/test.parquet" \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.gainrl.enable=enable \
    data.gainrl.processed_file="/home/dz/SAERL/angle_sorted_data/Qwen2.5-0.5B-Instruct/processed_gsm8k.pt" \
    actor_rollout_ref.model.path=/data0/dz/models/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='gainrl' \
    trainer.experiment_name='Qwen2.5-0.5B-Instruct_gsm8k_GAINRL_new' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=200 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="/data0/dz/ckpt/gainrl/Qwen2.5-0.5B-Instruct_gsm8k_GAINRL_new/" $@


    # data.train_batch_size=1024 
    # trainer.n_gpus_per_node=2 
    # actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8000