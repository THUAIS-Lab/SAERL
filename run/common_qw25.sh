#!/usr/bin/env bash
# Shared paths and Qwen2.5-Math-1.5B settings; sourced by the three launchers.

SAERL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SAERL_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$SAERL_ROOT/wandb}"
mkdir -p "$WANDB_DIR"

SAERL_TRAIN_DATA="${SAERL_TRAIN_DATA:-/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/sorted_deepmath/train.parquet}"
SAERL_VAL_DATA="${SAERL_VAL_DATA:-[\"/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/AIME_2024/train.parquet\",\"/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/math500/test.parquet\"]}"
SAERL_MODEL_PATH="${SAERL_MODEL_PATH:-/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-1.5B}"
SAERL_SAE_PATH="${SAERL_SAE_PATH:-/WORK/PUBLIC/lijuanzi1_work/dz/SAERL/SAERL-0801/OpenSAE-Qwen2.5-Math-1.5B-layer26}"
SAERL_PROCESSED_FILE="${SAERL_PROCESSED_FILE:-/WORK/PUBLIC/lijuanzi1_work/dz/SAERL/rank_sorted/combine_curr/deepmath_sel_hidden_diff_sae_batch_v4mix8.pt}"
SAERL_WORK_DIR="${SAERL_WORK_DIR:-$SAERL_ROOT/runs/qw25}"
SAERL_RUN_NAME="${SAERL_RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"

DATA_ARGS=(
    "data.train_files=$SAERL_TRAIN_DATA"
    "data.val_files=$SAERL_VAL_DATA"
    "data.train_batch_size=${SAERL_TRAIN_BATCH_SIZE:-128}"
    data.val_batch_size=512
    data.max_prompt_length=1024
    data.max_response_length=3072
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL_ARGS=(
    "actor_rollout_ref.model.path=$SAERL_MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp
    "actor_rollout_ref.actor.ppo_mini_batch_size=${SAERL_PPO_MINI_BATCH_SIZE:-64}"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.temperature=0.6
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85
    "actor_rollout_ref.rollout.n=${SAERL_ROLLOUT_N:-8}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
)

GRPO_ARGS=(
    "${DATA_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    algorithm.adv_estimator=grpo
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.n=8
    algorithm.kl_ctrl.kl_coef=0.001
    trainer.critic_warmup=0
    'trainer.logger=[console,wandb]'
    trainer.project_name=deepmath_difficulty
    trainer.experiment_name=grpo
    trainer.val_before_train=True
    trainer.nnodes=1
    "trainer.n_gpus_per_node=${SAERL_TRAIN_GPUS:-4}"
    "trainer.total_epochs=${SAERL_TOTAL_EPOCHS:-30}"
    trainer.save_freq=20
    trainer.test_freq=20
    trainer.default_hdfs_dir=null
    trainer.resume_mode=auto
    "trainer.default_local_dir=$SAERL_WORK_DIR/checkpoints/grpo"
)
