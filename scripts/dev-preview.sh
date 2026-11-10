#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
spl_preview_port="${PORT:-7860}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) spl_preview_port="$2"; shift 2 ;;
    --host) shift 2 ;;
    *) shift ;;
  esac
done
export SPL_FT_HOME="${SPL_FT_HOME:-$PWD/.spl-ft-preview}"
export PORT="$spl_preview_port"
exec .venv/bin/python scripts/preview_app.py
