#!/bin/bash
set -euo pipefail
set -x

VERL_ROOT=/home/fit/lijuanzi1/WORK/dz/SAERL
PYTHON_BIN="${PYTHON_BIN:-/WORK/PUBLIC/lijuanzi1_work/jy/miniconda3/envs/verl_dz/bin/python}"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "${VERL_ROOT}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    trainer.experiment_name='static_grad' \
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
