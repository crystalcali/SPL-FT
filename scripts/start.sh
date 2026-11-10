#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/spl-ft ]]; then bash scripts/setup.sh; fi
export SPL_FT_HOME="${SPL_FT_HOME:-$PWD/.spl-ft}"
exec .venv/bin/spl-ft web --host "${SPL_FT_HOST:-0.0.0.0}" --port "${SPL_FT_PORT:-7860}"
