#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

SAERL_RUN_DIR="${SAERL_RUN_DIR:-path/to/gradient_run}"
SAERL_SELECTION_DIR="${SAERL_SELECTION_DIR:-path/to/selection_output}"
SAERL_FEATURE_TOP_K="${SAERL_FEATURE_TOP_K:-256}"
SAERL_COSINE_THRESHOLD="${SAERL_COSINE_THRESHOLD:-0.7}"
FEATURE_INDICES="$SAERL_SELECTION_DIR/feature_selection/feature_indices.npy"

RUN_DIRS=("$@")
if (( ${#RUN_DIRS[@]} == 0 )); then
    RUN_DIRS=("$SAERL_RUN_DIR")
fi

"$PYTHON_BIN" -m saerl.selection.feature_selection \
    --run-dir "${RUN_DIRS[@]}" \
    --top-k "$SAERL_FEATURE_TOP_K" \
    --output "$FEATURE_INDICES"

"$PYTHON_BIN" -m saerl.selection.sample_selection \
    --run-dir "${RUN_DIRS[@]}" \
    --feature-indices "$FEATURE_INDICES" \
    --device cuda \
    --cosine-threshold "$SAERL_COSINE_THRESHOLD" \
    --read-batch-size "${SAERL_SELECTION_READ_BATCH_SIZE:-4}" \
    --similarity-block-size "${SAERL_SIMILARITY_BLOCK_SIZE:-64}" \
    --output-dir "$SAERL_SELECTION_DIR"

exec "$PYTHON_BIN" -m saerl.selection.export_subset \
    --run-dir "${RUN_DIRS[@]}" \
    --selection-dir "$SAERL_SELECTION_DIR" \
    --dataset "$SAERL_TRAIN_DATA"
