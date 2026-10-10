#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common_qw25.sh"

exec "$PYTHON_BIN" -m saerl.selection.extraction \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=deepmath_difficulty \
    trainer.experiment_name=static_grad \
    trainer.nnodes=1 \
    "trainer.n_gpus_per_node=${SAERL_EXTRACT_GPUS:-2}" \
    "static_gradient_extraction.sae_path=$SAERL_SAE_PATH" \
    "static_gradient_extraction.output_dir=$SAERL_WORK_DIR/extraction" \
    "static_gradient_extraction.run_name=$SAERL_RUN_NAME" \
    static_gradient_extraction.sample_id_key=index \
    "static_gradient_extraction.gradient_feature_topk=${SAERL_TOKEN_TOP_K:-128}" \
    "static_gradient_extraction.projection_chunk_size=${SAERL_PROJECTION_CHUNK_SIZE:-32}" \
    "$@"
