from __future__ import annotations

import gzip
import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from wtai.data.h5ad_minimal import build_gene_index, read_csc_gene_columns, read_gene_symbols, read_obs_frame
from wtai.data.signatures import HIGH_RISK_PROGRAMS, SIGNATURES
from wtai.paths import PROCESSED, RAW, RESULTS, CACHE, configure_local_cache


OBS_COLUMNS = [
    "sample_id",
    "library_id",
    "subdiagnosis",
    "relapse_status",
    "vital_status",
    "consensus_celltype_annotation",
    "singler_celltype_annotation",
    "cellassign_celltype_annotation",
    "cluster",
]


def signature_score(expr: dict[str, np.ndarray], genes: list[str], n_cells: int) -> np.ndarray:
    present = [expr[g.upper()] for g in genes if g.upper() in expr]
    if not present:
        return np.zeros(n_cells, dtype=np.float32)
    return np.vstack(present).mean(axis=0)


def load_lr_genes() -> set[str]:
    path = RESULTS / "tables" / "target_wt_lr_expression_scores.tsv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t").head(100)
    genes = set(df["ligand"].astype(str).str.upper()) | set(df["receptor"].astype(str).str.upper())
    return genes


def first_existing(base: Path, names: list[str]) -> Path:
    for name in names:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing ScPCA snRNA/scRNA AnnData zip. Expected one of: " + ", ".join(str(base / name) for name in names)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-libraries", type=int, default=0)
    args = parser.parse_args()

    configure_local_cache()
    scpca_dir = RAW / "scpca" / "SCPCP000006"
    zip_path = first_existing(
        scpca_dir,
        [
            "SCPCP000006_ann-data_2026-05-24.zip",
            "SCPCP000006_SINGLE-CELL_ANN-DATA.zip",
        ],
    )
    out_dir = PROCESSED / "scpca_singlecell"
    res_dir = RESULTS / "singlecell"
    tmp_dir = CACHE / "tmp" / "scpca_singlecell"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    lr_genes = load_lr_genes()
    wanted_genes = set(lr_genes)
    for genes in SIGNATURES.values():
        wanted_genes.update(g.upper() for g in genes)

    celltype_rows: list[pd.DataFrame] = []
    sample_rows: list[dict] = []
    gene_mean_rows: list[dict] = []
    slim_path = res_dir / "scpca_cell_scores_slim.tsv.gz"

    with zipfile.ZipFile(zip_path) as zf, gzip.open(slim_path, "wt", encoding="utf-8") as slim:
        h5ads = [n for n in zf.namelist() if n.endswith("_processed_rna.h5ad")]
        if args.max_libraries:
            h5ads = h5ads[: args.max_libraries]
        slim.write(
            "cell_id\tsample_id\tlibrary_id\tsubdiagnosis\tcelltype\tumap_1\tumap_2\t"
            + "\t".join(HIGH_RISK_PROGRAMS)
            + "\tstructpde_wt_tt_cell_score\n"
        )
        for member in tqdm(h5ads, desc="ScPCA snRNA h5ad"):
            tmp = tmp_dir / Path(member).name
            with zf.open(member) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024 * 8)
            try:
                obs = read_obs_frame(tmp, OBS_COLUMNS)
                genes = read_gene_symbols(tmp, raw=True)
                gene_index = build_gene_index(genes, wanted_genes)
                expr = read_csc_gene_columns(tmp, gene_index)
                n_cells = len(obs)
                for name, sig_genes in SIGNATURES.items():
                    obs[name] = signature_score(expr, sig_genes, n_cells)
                obs["structpde_wt_tt_cell_score"] = obs[HIGH_RISK_PROGRAMS].mean(axis=1)
                if "consensus_celltype_annotation" in obs:
                    obs["celltype"] = obs["consensus_celltype_annotation"].replace("", "unassigned")
                else:
                    obs["celltype"] = "unassigned"

                # UMAP is optional; avoid pulling heavy matrices here and keep slim columns stable.
                obs["umap_1"] = np.nan
                obs["umap_2"] = np.nan

                group_cols = ["sample_id", "library_id", "subdiagnosis", "celltype"]
                agg = obs.groupby(group_cols, dropna=False)[HIGH_RISK_PROGRAMS + ["structpde_wt_tt_cell_score"]].agg(
                    ["mean", "median", "std", "count"]
                )
                agg.columns = ["_".join(col).strip("_") for col in agg.columns]
                celltype_rows.append(agg.reset_index())

                sample_summary = obs.groupby(["sample_id", "library_id", "subdiagnosis"], dropna=False)[
                    HIGH_RISK_PROGRAMS + ["structpde_wt_tt_cell_score"]
                ].mean()
                for idx, row in sample_summary.iterrows():
                    record = dict(zip(["sample_id", "library_id", "subdiagnosis"], idx))
                    record.update(row.to_dict())
                    record["n_cells"] = n_cells
                    record["n_celltypes"] = int(obs["celltype"].nunique())
                    sample_rows.append(record)

                for (sample_id, library_id, subdiagnosis, celltype), idx in obs.groupby(group_cols, dropna=False).groups.items():
                    record = {
                        "sample_id": sample_id,
                        "library_id": library_id,
                        "subdiagnosis": subdiagnosis,
                        "celltype": celltype,
                        "n_cells": len(idx),
                    }
                    for gene, values in expr.items():
                        record[gene] = float(np.mean(values[list(idx)]))
                    gene_mean_rows.append(record)

                slim_cols = [
                    "cell_id",
                    "sample_id",
                    "library_id",
                    "subdiagnosis",
                    "celltype",
                    "umap_1",
                    "umap_2",
                    *HIGH_RISK_PROGRAMS,
                    "structpde_wt_tt_cell_score",
                ]
                obs[slim_cols].to_csv(slim, sep="\t", index=False, header=False)
            finally:
                tmp.unlink(missing_ok=True)

    celltype_df = pd.concat(celltype_rows, ignore_index=True) if celltype_rows else pd.DataFrame()
    sample_df = pd.DataFrame(sample_rows)
    gene_mean_df = pd.DataFrame(gene_mean_rows)
    celltype_df.to_csv(res_dir / "scpca_signature_by_celltype.tsv", sep="\t", index=False)
    sample_df.to_csv(res_dir / "scpca_sample_signature_summary.tsv", sep="\t", index=False)
    gene_mean_df.to_csv(out_dir / "celltype_gene_means.tsv", sep="\t", index=False)
    summary = {
        "n_libraries": int(sample_df["library_id"].nunique()) if not sample_df.empty else 0,
        "n_samples": int(sample_df["sample_id"].nunique()) if not sample_df.empty else 0,
        "n_cells": int(sample_df["n_cells"].sum()) if not sample_df.empty else 0,
        "n_gene_features_used": len(wanted_genes),
    }
    pd.Series(summary).to_csv(res_dir / "scpca_singlecell_qc_summary.tsv", sep="\t", header=False)
    print(res_dir / "scpca_sample_signature_summary.tsv")


if __name__ == "__main__":
    main()
