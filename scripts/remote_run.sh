#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$(basename "$CODE_ROOT")" = "codes" ] || [ "$(basename "$CODE_ROOT")" = "published_codes" ]; then
  ROOT="$(cd "$CODE_ROOT/.." && pwd)"
else
  ROOT="$CODE_ROOT"
fi
cd "$ROOT"

export PYTHONPATH="$CODE_ROOT/src"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$ROOT/.cache/hf/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export TMPDIR="$ROOT/.cache/tmp"
export TMP="$ROOT/.cache/tmp"
export TEMP="$ROOT/.cache/tmp"
export PIP_PROGRESS_BAR=off
export PYTHONIOENCODING=utf-8
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"

mkdir -p "$ROOT/.deps"
export PYTHONPATH="$ROOT/.deps:$CODE_ROOT/src"
if [ ! -d "$ROOT/.deps/pandas" ] || [ ! -d "$ROOT/.deps/h5py" ]; then
  python3 -m pip install --target "$ROOT/.deps" --cache-dir "$PIP_CACHE_DIR" -r "$CODE_ROOT/requirements-remote-minimal.txt"
fi

while [ ! -f "$ROOT/.assets_ready" ]; do
  echo "Waiting for .assets_ready marker before starting full offline run..."
  sleep 60
done

bash "$CODE_ROOT/scripts/run_all_local.sh"
