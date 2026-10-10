#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common_qw25.sh"

# processed_file already contains selected full-dataset row indices in curriculum order.
exec "$PYTHON_BIN" -m saerl.verl.train --config-name saerl \
    "${GRPO_ARGS[@]}" \
    data.filter_overlong_prompts=False \
    data.train_max_samples=-1 \
    "data.saerl.processed_file=$SAERL_PROCESSED_FILE" \
    trainer.experiment_name=saerl \
    trainer.val_before_train=False \
    "trainer.default_local_dir=$SAERL_WORK_DIR/checkpoints/saerl" \
    "$@"
