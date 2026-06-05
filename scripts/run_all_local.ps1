$ErrorActionPreference = "Stop"
$CodeRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ((Split-Path $CodeRoot -Leaf) -in @("codes", "published_codes")) {
  $Root = Resolve-Path (Join-Path $CodeRoot "..")
} else {
  $Root = $CodeRoot
}
$env:PYTHONPATH = "$(Join-Path $Root ".deps");$(Join-Path $CodeRoot "src");$env:PYTHONPATH"
$env:PIP_CACHE_DIR = Join-Path $Root ".cache\pip"
$env:HF_HOME = Join-Path $Root ".cache\hf"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $Root ".cache\hf\hub"
$env:TORCH_HOME = Join-Path $Root ".cache\torch"
$env:MPLCONFIGDIR = Join-Path $Root ".cache\matplotlib"
$env:XDG_CACHE_HOME = Join-Path $Root ".cache\xdg"
$env:TMPDIR = Join-Path $Root ".cache\tmp"
$env:TEMP = Join-Path $Root ".cache\tmp"
$env:TMP = Join-Path $Root ".cache\tmp"

@($env:PIP_CACHE_DIR, $env:HF_HOME, $env:TORCH_HOME, $env:MPLCONFIGDIR, $env:XDG_CACHE_HOME, $env:TMPDIR) |
  ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

python (Join-Path $CodeRoot "scripts\01_preprocess_bulk.py")
python (Join-Path $CodeRoot "scripts\02_run_lr_and_structure.py")
python (Join-Path $CodeRoot "scripts\03_run_pde_bulk_proxy.py")
python (Join-Path $CodeRoot "scripts\04_eval_bulk.py")
python (Join-Path $CodeRoot "scripts\05_preprocess_scpca_singlecell.py")
python (Join-Path $CodeRoot "scripts\06_run_spatial_pde.py")
python (Join-Path $CodeRoot "scripts\07_fetal_reference_mapping.py")
python (Join-Path $CodeRoot "scripts\09_eval_full_ablation.py")
python (Join-Path $CodeRoot "scripts\10_compile_paper_case_studies.py")
