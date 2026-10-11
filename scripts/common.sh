#!/usr/bin/env bash

SAERL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SAERL_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$SAERL_ROOT/wandb}"
mkdir -p "$WANDB_DIR"

SAERL_TRAIN_DATA="${SAERL_TRAIN_DATA:-path/to/train.parquet}"
SAERL_VAL_DATA="${SAERL_VAL_DATA:-path/to/val.parquet}"
SAERL_MODEL_PATH="${SAERL_MODEL_PATH:-path/to/model}"
SAERL_WORK_DIR="${SAERL_WORK_DIR:-$SAERL_ROOT/runs}"

COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    "data.train_files=$SAERL_TRAIN_DATA"
    "data.val_files=$SAERL_VAL_DATA"
    "data.train_batch_size=${SAERL_TRAIN_BATCH_SIZE:-128}"
    data.val_batch_size=512
    data.max_prompt_length=1024
    data.max_response_length=3072
    data.filter_overlong_prompts=True
    data.truncation=error
    "actor_rollout_ref.model.path=$SAERL_MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    "actor_rollout_ref.actor.ppo_mini_batch_size=${SAERL_PPO_MINI_BATCH_SIZE:-64}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85
    "actor_rollout_ref.rollout.n=${SAERL_ROLLOUT_N:-8}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.n=8
    trainer.critic_warmup=0
    'trainer.logger=[console,wandb]'
    trainer.project_name=saerl
    trainer.nnodes=1
    trainer.save_freq=20
    trainer.test_freq=20
    trainer.default_hdfs_dir=null
    trainer.resume_mode=auto
)
