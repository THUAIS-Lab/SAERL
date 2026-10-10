#!/bin/bash
set -euo pipefail
set -x

VERL_ROOT=/home/fit/lijuanzi1/WORK/dz/SAERL
PYTHON_BIN="${PYTHON_BIN:-/WORK/PUBLIC/lijuanzi1_work/jy/miniconda3/envs/verl_dz/bin/python}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export WANDB_MODE=offline
export WANDB_DIR=/home/fit/lijuanzi1/WORK/dz/SAERL/recipe/static_gradient_sae

cd "${VERL_ROOT}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/sorted_deepmath/train.parquet" \
    'data.val_files=["/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/AIME_2024/train.parquet", "/WORK/PUBLIC/lijuanzi1_work/dz/data/SAERL/my_data/math500/test.parquet"]' \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=/WORK/PUBLIC/lijuanzi1_work/MODELS/Qwen2.5-Math-1.5B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='deepmath_difficulty' \
    trainer.experiment_name='static_grad' \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=2 \
    trainer.use_legacy_worker_impl=enable \
    static_gradient_extraction.enable=True \
    static_gradient_extraction.sae_path="${VERL_ROOT}/SAERL-0801/OpenSAE-Qwen2.5-Math-1.5B-layer26" \
    static_gradient_extraction.output_dir="${VERL_ROOT}/SAERL-0801/runs/selection/qwen2_5" \
    static_gradient_extraction.sample_id_key=index \
    static_gradient_extraction.gradient_token_scope=attention \
    static_gradient_extraction.projection_chunk_size=4096 \
    # static_gradient_extraction.run_name="20260815-031029-164461" \
    # static_gradient_extraction.resume_from_step=743 \
    "$@"
