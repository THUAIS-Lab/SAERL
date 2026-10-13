#!/usr/bin/env bash
set -euo pipefail

SAERL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SAERL_VERL_COMMIT="875788ef6e218d0b98cb64fff919d9fef014f986"
SAERL_INSTALL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/saerl-install.XXXXXX")"
trap 'rm -rf -- "$SAERL_INSTALL_DIR"' EXIT

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Use the dependency installer from the same upstream commit as requirements.txt.
curl --fail --location \
    "https://raw.githubusercontent.com/verl-project/verl/$SAERL_VERL_COMMIT/scripts/install_vllm_sglang_mcore.sh" \
    --output "$SAERL_INSTALL_DIR/upstream.sh"

# The upstream script hardcodes MAX_JOBS=32 and the FALSE FlashAttention ABI.
SAERL_FLASH_ATTN_ABI="$(python -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')"
sed \
    -e 's/export MAX_JOBS=32/export MAX_JOBS=${MAX_JOBS:-2}/' \
    -e "s/cxx11abiFALSE/cxx11abi$SAERL_FLASH_ATTN_ABI/g" \
    "$SAERL_INSTALL_DIR/upstream.sh" > "$SAERL_INSTALL_DIR/install.sh"

# Keep the upstream dependency installation within this project's version pins.
export PIP_CONSTRAINT="$SAERL_ROOT/requirements.txt"
(
    cd "$SAERL_INSTALL_DIR"
    MAX_JOBS="${MAX_JOBS:-2}" USE_MEGATRON=0 USE_SGLANG=0 bash -e install.sh
)

# The train extra installs the pinned official verl and this project's adapters.
python -m pip install -e "$SAERL_ROOT[train]" --no-build-isolation
python -m pip check
