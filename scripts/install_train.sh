#!/usr/bin/env bash
set -euo pipefail

SAERL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SAERL_VERL_COMMIT="875788ef6e218d0b98cb64fff919d9fef014f986"
SAERL_INSTALL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/saerl-install.XXXXXX")"
trap 'rm -rf -- "$SAERL_INSTALL_DIR"' EXIT

export PIP_CONSTRAINT="$SAERL_ROOT/requirements.txt"

printf "[1/5] Installing PyTorch CUDA 12.8 and prerequisite packages\n"
python -m pip install --upgrade pip setuptools 'packaging>=24.2' wheel
python -m pip install numpy==1.26.4 filelock==3.25.2
python -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

printf "[2/5] Downloading the pinned verl dependency installer\n"
# Use the dependency installer from the same upstream commit as requirements.txt.
curl --fail --location \
    "https://raw.githubusercontent.com/verl-project/verl/$SAERL_VERL_COMMIT/scripts/install_vllm_sglang_mcore.sh" \
    --output "$SAERL_INSTALL_DIR/upstream.sh"

# The upstream script hardcodes MAX_JOBS=32 and the FALSE FlashAttention ABI.
printf "[3/5] Importing torch to read the FlashAttention ABI\n"
SAERL_FLASH_ATTN_ABI="$(python -u -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')"
sed \
    -e 's/export MAX_JOBS=32/export MAX_JOBS=${MAX_JOBS:-2}/' \
    -e "s/cxx11abiFALSE/cxx11abi$SAERL_FLASH_ATTN_ABI/g" \
    "$SAERL_INSTALL_DIR/upstream.sh" > "$SAERL_INSTALL_DIR/install.sh"

printf "[4/5] Running the official FSDP + vLLM dependency installer\n"
(
    cd "$SAERL_INSTALL_DIR"
    MAX_JOBS="${MAX_JOBS:-2}" USE_MEGATRON=0 USE_SGLANG=0 bash -e install.sh
)

printf "[5/5] Installing pinned verl and SAERL\n"
# The train extra installs the pinned official verl and this project's adapters.
python -m pip install -e "$SAERL_ROOT[train]" --no-build-isolation
python -m pip check
