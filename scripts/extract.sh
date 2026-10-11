#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

SAERL_SAE_PATH="${SAERL_SAE_PATH:-path/to/sae}"
SAERL_RUN_NAME="${SAERL_RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"

exec "$PYTHON_BIN" -m saerl.selection.extraction \
    "${COMMON_ARGS[@]}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    trainer.experiment_name=static_grad \
    trainer.resume_mode=disable \
    "trainer.n_gpus_per_node=${SAERL_EXTRACT_GPUS:-4}" \
    "static_gradient_extraction.sae_path=$SAERL_SAE_PATH" \
    "static_gradient_extraction.output_dir=$SAERL_WORK_DIR/extraction" \
    "static_gradient_extraction.run_name=$SAERL_RUN_NAME" \
    static_gradient_extraction.sample_id_key=index \
    "static_gradient_extraction.gradient_feature_topk=${SAERL_TOKEN_TOP_K:-128}" \
    "static_gradient_extraction.projection_chunk_size=${SAERL_PROJECTION_CHUNK_SIZE:-4096}" \
    "$@"
