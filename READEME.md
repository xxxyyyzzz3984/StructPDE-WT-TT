# Implementation of StructPDE-WT-TT

This directory contains the publication-ready implementation of **StructPDE-WT-TT**, a structure-constrained graph reaction-diffusion framework for Wilms tumor spatial communication analysis.

The code package intentionally does **not** include raw data, processed data, model weights, generated results, local credentials, remote transfer archives, or cache files. All downloads, caches, intermediate files, model weights, and outputs should stay inside this project directory after the reader runs the commands below.

## 1. Project Structure

```text
StructPDE-WT-TT/
  codes/                              # source code and reproducibility scripts
    configs/
      data.yaml                       # public resource definitions
      model.yaml                      # model and analysis parameters
    scripts/
      00_download_data.py             # public data, LR resources, fetal kidney reference
      01_preprocess_bulk.py           # TARGET/GEO bulk preprocessing
      02_run_lr_and_structure.py       # LR expression and prior structure scoring
      03_run_pde_bulk_proxy.py         # bulk graph PDE proxy
      04_eval_bulk.py                 # TARGET/GEO validation summaries
      05_preprocess_scpca_singlecell.py # ScPCA snRNA/scRNA scoring
      06_run_spatial_pde.py           # ScPCA Visium graph reaction-diffusion
      07_fetal_reference_mapping.py   # fetal kidney reference scoring
      08_download_structure_weights.py # Boltz/Protenix/AlphaFold-related weight downloader
      09_eval_full_ablation.py        # ablation and LOO spatial classifier
      10_compile_paper_case_studies.py # case-study tables and notes
      11_run_structure_complex_confidence.py
      12_structure_evidence_rescoring.py
      13_run_refined_key_axis_predictions.py
      generate_paper_figures.py       # optional manuscript figure generator
      run_all_local.ps1               # Windows orchestration
      run_all_local.sh                # Linux/macOS orchestration
      remote_run.sh                   # Linux/3090 server orchestration
      remote_full_pipeline.py         # optional private-server helper; requires user-created ssh-account.txt
      upload_remote_assets.py         # optional private-server helper; requires user-created ssh-account.txt
    src/wtai/
      communication/                  # ligand-receptor database integration
      data/                           # download, h5ad, spatial zip, signature helpers
      eval/                           # bulk validation
      pde/                            # graph Laplacian and reaction-diffusion kernels
      structure/                      # structure evidence scoring
      paths.py                        # resolves data/models/results from the outer project root
    pyproject.toml
    requirements-remote-minimal.txt
  data/                               # downloaded raw data and processed matrices; not included in the code package
    raw/                              # ScPCA, TARGET-WT, GEO, fetal kidney reference
    external/                         # ligand-receptor resources
    processed/                        # intermediate processed matrices
  models/                             # downloaded structure-predictor weights; not included in the code package
    structure_predictors/
      boltz/
      protenix/
  results/                            # generated tables, figures, structure predictions, and paper materials
  paper/                              # optional generated manuscript figures
```

## 2. Environment

Recommended:

- Python 3.10 or 3.11
- Linux GPU server with an NVIDIA RTX 3090 or equivalent for Boltz-2 structure prediction
- Windows or Linux CPU machine for bulk, ScPCA, spatial PDE, ablation, and paper-table generation
- At least 150 GB free space for the complete ScPCA + structure workflow

Create a clean environment:

```bash
cd StructPDE-WT-TT/codes
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell:

```powershell
cd E:\path\to\StructPDE-WT-TT\codes
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

All scripts call `wtai.paths.configure_local_cache()` so the following caches stay inside the outer project directory, parallel to `codes/`:

```text
.cache/pip
.cache/hf
.cache/torch
.cache/matplotlib
.cache/xdg
.cache/tmp
```

## 3. Data Resources and Placement

The project expects all raw data under `data/raw/`, external annotation under `data/external/`, processed matrices under `data/processed/`, model weights under `models/`, and outputs under `results/`.

### 3.1 ScPCA Wilms Tumor snRNA/scRNA and Visium

Official project:

- ScPCA SCPCP000006: https://scpca.alexslemonade.org/projects/SCPCP000006
- ScPCA API: https://api.scpca.alexslemonade.org/v1

This data requires acceptance of ScPCA terms and either an API token or token creation through email.

Automatic download:

```bash
export SCPCA_EMAIL="your_email@example.org"
python scripts/00_download_data.py --accept-scpca-terms
```

or:

```bash
export SCPCA_TOKEN="your_scpca_token"
python scripts/00_download_data.py
```

Manual placement if automatic download is unavailable:

```text
data/raw/scpca/SCPCP000006/computed_files.tsv
data/raw/scpca/SCPCP000006/project_metadata.json
data/raw/scpca/SCPCP000006/SCPCP000006_SINGLE-CELL_ANN-DATA.zip
data/raw/scpca/SCPCP000006/SCPCP000006_SPATIAL_SINGLE-CELL-EXPERIMENT.zip
```

The preprocessing scripts also accept the filenames used in the original run:

```text
data/raw/scpca/SCPCP000006/SCPCP000006_ann-data_2026-05-24.zip
data/raw/scpca/SCPCP000006/SCPCP000006_spaceranger_2026-05-24.zip
```

### 3.2 TARGET-WT Open-Access Bulk Expression and Clinical Metadata

Official resources:

- GDC project portal: https://portal.gdc.cancer.gov/projects/TARGET-WT
- GDC API: https://api.gdc.cancer.gov
- TARGET-WT publication page: https://gdc.cancer.gov/about-data/publications/TARGET-WT-2017

Automatic download:

```bash
python scripts/00_download_data.py --skip-scpca --skip-fetal-kidney --download-target-expression
```

Expected paths:

```text
data/raw/target_wt/clinical.tsv
data/raw/target_wt/gdc_open_expression_files.tsv
data/raw/target_wt/expression/*.tsv
```

TARGET controlled-access genomics are not required for reproducing this paper version. Those files require dbGaP/GDC approval and were treated as future validation material.

### 3.3 GEO External Bulk Cohorts

Official resources:

- GSE31403: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE31403
- GSE10320: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10320
- GEO FTP root: https://ftp.ncbi.nlm.nih.gov/geo/series/

Automatic download:

```bash
python scripts/00_download_data.py --skip-scpca --skip-fetal-kidney --geo GSE31403 GSE10320
```

Expected paths:

```text
data/raw/geo/GSE31403/GSE31403_series_matrix.txt.gz
data/raw/geo/GSE31403/GSE31403_family.soft.gz
data/raw/geo/GSE31403/GSE31403_miniml.tgz
data/raw/geo/GSE10320/GSE10320_series_matrix.txt.gz
data/raw/geo/GSE10320/GSE10320_family.soft.gz
data/raw/geo/GSE10320/GSE10320_miniml.tgz
```

### 3.4 Fetal Kidney Reference

Official resources:

- HCA project: https://explore.data.humancellatlas.org/projects/d8ae869c-39c2-4cdd-b3fc-2d0d8f60e7b8
- HCA Azul file API: https://service.azul.data.humancellatlas.org/index/files
- Reference article: https://pmc.ncbi.nlm.nih.gov/articles/PMC6104812/

Automatic download:

```bash
python scripts/00_download_data.py --skip-scpca
```

Expected paths:

```text
data/raw/fetal_kidney_hca/d8ae869c-39c2-4cdd-b3fc-2d0d8f60e7b8/tableOfCounts.mtx
data/raw/fetal_kidney_hca/d8ae869c-39c2-4cdd-b3fc-2d0d8f60e7b8/tableOfCounts_colLabels.tsv
data/raw/fetal_kidney_hca/d8ae869c-39c2-4cdd-b3fc-2d0d8f60e7b8/tableOfCounts_rowLabels.tsv
data/raw/fetal_kidney_hca/d8ae869c-39c2-4cdd-b3fc-2d0d8f60e7b8/Haniffa-Human-10x3pv2_metadata_04-07-2023.xlsx
```

### 3.5 Ligand-Receptor Resource

Official resource:

- OmniPath LR interactions: https://omnipathdb.org/interactions?datasets=ligrecextra&organisms=9606&genesymbols=1&fields=sources,references,curation_effort,type

Expected path:

```text
data/external/ligand_receptor/omnipath_ligrecextra.tsv
```

## 4. Structure Model Weights and Placement

Structure prediction is optional for the CPU-only pipeline, but required to reproduce the Boltz-2 complex confidence and refined key-axis experiments.

### 4.1 Boltz-2

Official resources:

- Boltz GitHub: https://github.com/jwohlwend/boltz
- Boltz-2 Hugging Face repository: https://huggingface.co/boltz-community/boltz-2

Download with:

```bash
python scripts/08_download_structure_weights.py --families boltz
```

Important expected paths:

```text
models/structure_predictors/boltz/boltz2_conf.ckpt
models/structure_predictors/boltz/boltz2_aff.ckpt
models/structure_predictors/boltz/mols.tar
models/structure_predictors/boltz/mols/
models/structure_predictors/boltz/weights_manifest.json
```

For GPU prediction, install Boltz in a repository-local dependency directory:

```bash
python -m pip install --target .deps_structure boltz
```

The structure scripts add `.deps_structure` and `.deps_structure/bin` to `PYTHONPATH` and `PATH` when launching `boltz predict`.

### 4.2 Protenix

Official resources:

- Protenix GitHub: https://github.com/bytedance/Protenix
- Checkpoints used by the downloader are listed in `scripts/08_download_structure_weights.py`.

Download with:

```bash
python scripts/08_download_structure_weights.py --families protenix
```

Expected paths:

```text
models/structure_predictors/protenix/checkpoint/protenix-v2.pt
models/structure_predictors/protenix/checkpoint/protenix_base_default_v1.0.0.pt
models/structure_predictors/protenix/checkpoint/protenix_mini_default_v0.5.0.pt
models/structure_predictors/protenix/checkpoint/protenix_tiny_default_v0.5.0.pt
models/structure_predictors/protenix/common/components.cif
models/structure_predictors/protenix/common/components.cif.rdkit_mol.pkl
```

Protenix is documented as an optional structure predictor. The reported full experiment primarily used Boltz-2.

### 4.3 AlphaFold DB and UniProt/PDB

The code does not bulk-download AlphaFold DB or PDB. It uses identifiers and public endpoints as structure evidence sources:

```text
AlphaFold DB file base: https://alphafold.ebi.ac.uk/files
UniProt REST API:       https://rest.uniprot.org/uniprotkb
RCSB PDB Search API:    https://search.rcsb.org/rcsbsearch/v2/query
```

If readers manually add AlphaFold monomer files, place them under:

```text
models/structure_predictors/alphafold/
```

## 5. Reproducing the Main Results

Run the following commands from `StructPDE-WT-TT/codes`. Set repository-local cache variables first. On Linux/macOS:

```bash
export PROJECT_ROOT="$(cd .. && pwd)"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export HF_HOME="$PROJECT_ROOT/.cache/hf"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/.cache/hf/hub"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"
```

On Windows PowerShell:

```powershell
$ProjectRoot = Resolve-Path ..
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
$env:PIP_CACHE_DIR = "$ProjectRoot\.cache\pip"
$env:HF_HOME = "$ProjectRoot\.cache\hf"
$env:HUGGINGFACE_HUB_CACHE = "$ProjectRoot\.cache\hf\hub"
$env:TORCH_HOME = "$ProjectRoot\.cache\torch"
$env:MPLCONFIGDIR = "$ProjectRoot\.cache\matplotlib"
$env:XDG_CACHE_HOME = "$ProjectRoot\.cache\xdg"
$env:TMPDIR = "$ProjectRoot\.cache\tmp"
$env:TEMP = "$ProjectRoot\.cache\tmp"
$env:TMP = "$ProjectRoot\.cache\tmp"
```

### 5.1 Download Public Resources

Full download attempt:

```bash
python scripts/00_download_data.py --accept-scpca-terms --download-target-expression
```

For users who cannot download ScPCA automatically, place the ScPCA files manually as described in section 3.1 and run:

```bash
python scripts/00_download_data.py --skip-scpca --download-target-expression
```

### 5.2 CPU Pipeline

Run each step explicitly:

```bash
python scripts/01_preprocess_bulk.py
python scripts/02_run_lr_and_structure.py
python scripts/03_run_pde_bulk_proxy.py
python scripts/04_eval_bulk.py
python scripts/05_preprocess_scpca_singlecell.py
python scripts/06_run_spatial_pde.py
python scripts/07_fetal_reference_mapping.py
python scripts/09_eval_full_ablation.py
python scripts/10_compile_paper_case_studies.py
```

Or run the platform wrapper:

```bash
bash scripts/run_all_local.sh
```

Windows:

```powershell
.\scripts\run_all_local.ps1
```

Expected major outputs:

```text
results/singlecell/scpca_singlecell_qc_summary.tsv
results/singlecell/scpca_sample_signature_summary.tsv
results/spatial/scpca_spatial_library_summary.tsv
results/spatial/scpca_spatial_spot_scores.tsv
results/spatial/scpca_spatial_lr_axis_summary.tsv
results/structure/lr_structure_scores.tsv
results/tables/target_wt_bulk_validation_summary.tsv
results/tables/full_spatial_ablation_metrics.tsv
results/models/spatial_library_classifier_predictions.tsv
results/case_studies/integrated_paper_case_studies.tsv
results/full_experiment_summary.json
```

### 5.3 Structure Prediction Pipeline on GPU

Install Boltz and download weights:

```bash
python -m pip install --target .deps_structure boltz
python scripts/08_download_structure_weights.py --families boltz
```

Run first-pass top LR complex confidence:

```bash
PYTHONPATH=src:.deps_structure python scripts/11_run_structure_complex_confidence.py --override
```

Run posterior structure evidence rescoring:

```bash
PYTHONPATH=src:.deps_structure python scripts/12_structure_evidence_rescoring.py
```

Run refined key-axis predictions:

```bash
PYTHONPATH=src:.deps_structure python scripts/13_run_refined_key_axis_predictions.py --override
```

Expected structure outputs:

```text
results/structure/structure_complex_confidence.tsv
results/structure/structure_complex_confidence_report.md
results/structure/structure_complex_confidence_manifest.tsv
results/structure/boltz_complex_predictions/
results/structure/structure_evidence_posterior.tsv
results/structure/structure_adjusted_spatial_axis_summary.tsv
results/structure/structure_adjusted_spatial_library_summary.tsv
results/structure/structure_evidence_rescoring_report.md
results/structure/refined_key_axis_manifest.tsv
results/structure/refined_key_axis_boltz2.tsv
results/structure/refined_key_axis_summary.tsv
results/structure/refined_key_axis_report.md
results/structure/refined_boltz_predictions/
```

### 5.4 Optional Paper Figures

After all results exist:

```bash
python scripts/generate_paper_figures.py
```

This generates manuscript figures only. Manuscript document assembly is outside this code package.

Expected outputs:

```text
paper/figures/fig1_method_overview.png
paper/figures/fig2_data_landscape.png
paper/figures/fig3_structure_axes.png
paper/figures/fig4_spatial_fields.png
paper/figures/fig5_model_performance.png
paper/figures/fig6_case_and_validation.png
```

Figure sources:

```text
Fig. 1  Method overview: generated from the workflow design encoded in the script.
Fig. 2  Data landscape: single-cell, spatial, and fetal-reference summary tables.
Fig. 3  Structure axes: LR structure scores, posterior evidence, and refined Boltz summaries.
Fig. 4  Spatial fields: spot-level StructPDE scores and spatial program scores.
Fig. 5  Model performance: LOO classifier predictions, coefficients, and ablation metrics.
Fig. 6  Case and validation: integrated case studies, TARGET bulk proxy, and fetal-reference context.
```

## 6. Smoke Tests

For a quick check before running the full dataset:

```bash
python scripts/01_preprocess_bulk.py
python scripts/02_run_lr_and_structure.py
python scripts/05_preprocess_scpca_singlecell.py --max-libraries 2
python scripts/06_run_spatial_pde.py --max-libraries 2
```

For structure input generation without GPU prediction:

```bash
python scripts/11_run_structure_complex_confidence.py --prepare-only
python scripts/13_run_refined_key_axis_predictions.py --prepare-only
```

## 7. Main Model Summary

The core spatial model builds a graph over Visium spots and diffuses ligand fields:

```text
f_l^(t+1) = f_l^t - dt * L f_l^t - dt * lambda * f_l^t + dt * s_l
```

The LR activation at spot `i` is:

```text
a_lr,i = r_i * sigmoid(beta * S_lr * minmax(f_l,i))
```

where:

- `s_l` is ligand expression;
- `f_l` is the diffused ligand field;
- `r_i` is receptor expression at the target spot;
- `S_lr` is the structure evidence score;
- `L` is the normalized graph Laplacian.

The posterior structure evidence module combines prior structure evidence, topology/domain accessibility, Boltz interface confidence, and label evidence:

```text
S_post = 0.35 * S_prior
       + 0.25 * domain_topology_score
       + 0.25 * interface_confidence
       + 0.15 * boltz_label_score
```

for LR axes with Boltz evidence, and:

```text
S_post = 0.70 * S_prior + 0.30 * domain_topology_score
```

otherwise.

## 8. Reproducibility Notes

- Do not place raw data or model weights outside this project directory if exact path reproducibility is desired.
- Do not commit `data/`, `models/`, `results/`, `.cache/`, `.deps/`, or `.deps_structure/`.
- `ssh-account.txt` is not distributed. The two optional private-server helper scripts expect the reader to create their own credential file if they want to reproduce the exact remote-copy workflow.
- ScPCA downloads may require manual browser/token workflow depending on institutional network policy.
- TARGET controlled-access genomics were not required for this implementation and should not be mixed into this result set unless a separate approved validation analysis is added.
- Boltz-2 predictions are stochastic across seeds and GPU/software versions; reported tables should be regenerated from the same scripts and recorded seeds.
