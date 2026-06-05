#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$(basename "$CODE_ROOT")" = "codes" ] || [ "$(basename "$CODE_ROOT")" = "published_codes" ]; then
  ROOT="$(cd "$CODE_ROOT/.." && pwd)"
else
  ROOT="$CODE_ROOT"
fi
export PYTHONPATH="$ROOT/.deps:$CODE_ROOT/src:${PYTHONPATH:-}"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$ROOT/.cache/hf/hub"
export TORCH_HOME="$ROOT/.cache/torch"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export TMPDIR="$ROOT/.cache/tmp"

mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"

python "$CODE_ROOT/scripts/01_preprocess_bulk.py"
python "$CODE_ROOT/scripts/02_run_lr_and_structure.py"
python "$CODE_ROOT/scripts/03_run_pde_bulk_proxy.py"
python "$CODE_ROOT/scripts/04_eval_bulk.py"
python "$CODE_ROOT/scripts/05_preprocess_scpca_singlecell.py"
python "$CODE_ROOT/scripts/06_run_spatial_pde.py"
python "$CODE_ROOT/scripts/07_fetal_reference_mapping.py"
python "$CODE_ROOT/scripts/09_eval_full_ablation.py"
python "$CODE_ROOT/scripts/10_compile_paper_case_studies.py"
