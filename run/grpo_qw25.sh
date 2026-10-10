#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common_qw25.sh"

exec "$PYTHON_BIN" -m saerl.verl.train "${GRPO_ARGS[@]}" "$@"
