#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ui_only=0
if [[ "${1:-}" == "--ui-only" ]]; then ui_only=1; fi
if command -v uv >/dev/null 2>&1; then
  spl_uv="$(command -v uv)"
else
  python3 -m venv .bootstrap
  .bootstrap/bin/pip install 'uv>=0.11'
  spl_uv="$PWD/.bootstrap/bin/uv"
fi
if [[ ! -x .venv/bin/python ]]; then
  "$spl_uv" venv --python 3.12 .venv
fi
if [[ "$ui_only" == 1 ]]; then
  "$spl_uv" pip install --python .venv/bin/python -e '.[test]'
else
  if ! .venv/bin/python -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    "$spl_uv" pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu130
  fi
  "$spl_uv" pip install --python .venv/bin/python -e '.[train,bench]'
  .venv/bin/python -c 'import torch; print("PyTorch:",torch.__version__,"CUDA:",torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA not available. See README troubleshooting."'
  if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
    bash scripts/enable-amd64.sh
  fi
fi
printf '\nSPL-FT installed. Start with: bash scripts/start.sh\n'
