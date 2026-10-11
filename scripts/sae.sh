#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

SAERL_PROCESSED_FILE="${SAERL_PROCESSED_FILE:-path/to/processed.pt}"

exec "$PYTHON_BIN" -m saerl.verl.train --config-name saerl \
    "${COMMON_ARGS[@]}" \
    "data.saerl.processed_file=$SAERL_PROCESSED_FILE" \
    trainer.experiment_name=saerl \
    trainer.val_before_train=True \
    "trainer.n_gpus_per_node=${SAERL_TRAIN_GPUS:-4}" \
    "trainer.total_epochs=${SAERL_TOTAL_EPOCHS:-30}" \
    "trainer.default_local_dir=$SAERL_WORK_DIR/checkpoints/saerl" \
    "$@"
