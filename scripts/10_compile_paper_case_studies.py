from __future__ import annotations

import json

import numpy as np
import pandas as pd

from wtai.data.signatures import HIGH_RISK_PROGRAMS
from wtai.paths import PROCESSED, RESULTS, configure_local_cache


CASE_DIR = RESULTS / "case_studies"


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    mean = float(values.mean())
    return (values - mean) / std


def top_label_from_columns(row: pd.Series, columns: dict[str, str]) -> str:
    best_label = "unknown"
    best_value = float("-inf")
    for label, column in columns.items():
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.notna(value) and float(value) > best_value:
            best_value = float(value)
            best_label = label
    return best_label


def load_bulk_table() -> pd.DataFrame:
    sig = pd.read_csv(PROCESSED / "target_wt" / "target_wt_signature_scores.tsv", sep="\t", index_col=0)
    pde = pd.read_csv(RESULTS / "pde" / "target_wt_structpde_activation_score.tsv", sep="\t", index_col=0)
    clinical = pd.read_csv(PROCESSED / "target_wt" / "target_wt_clinical.tsv", sep="\t")
    bulk = sig.join(pde, how="left")
    bulk.index.name = "sample_id"
    bulk = bulk.reset_index().merge(
        clinical.rename(columns={"submitter_id": "sample_id"}), on="sample_id", how="left"
    )
    bulk["followup_days"] = pd.to_numeric(bulk["days_to_last_follow_up"], errors="coerce")
    bulk["bulk_case_score"] = (
        zscore(bulk["structpde_wt_tt_proxy"])
        + zscore(bulk["structpde_activation_proxy"])
        + zscore(bulk["anaplasia_proliferation"])
        + zscore(bulk["blastemal_progenitor"])
    ) / 4.0
    bulk["top_high_risk_program"] = bulk.apply(
        lambda row: top_label_from_columns(row, {name: name for name in HIGH_RISK_PROGRAMS}), axis=1
    )
    bulk["structpde_activation_rank"] = bulk["structpde_activation_proxy"].rank(ascending=False, method="dense")
    bulk["bulk_case_label"] = "background"
    if bulk["followup_days"].notna().sum() >= 8:
        hi_score = bulk["bulk_case_score"].quantile(0.85)
        lo_score = bulk["bulk_case_score"].quantile(0.15)
        short_follow = bulk["followup_days"].quantile(0.25)
        long_follow = bulk["followup_days"].quantile(0.75)
        bulk.loc[(bulk["bulk_case_score"] >= hi_score) & (bulk["followup_days"] <= short_follow), "bulk_case_label"] = (
            "high_risk_short_followup"
        )
        bulk.loc[(bulk["bulk_case_score"] >= hi_score) & (bulk["followup_days"] >= long_follow), "bulk_case_label"] = (
            "discordant_high_risk_long_followup"
        )
        bulk.loc[(bulk["bulk_case_score"] <= lo_score) & (bulk["followup_days"] >= long_follow), "bulk_case_label"] = (
            "low_risk_long_followup"
        )
    return bulk.sort_values(["bulk_case_score", "structpde_activation_proxy"], ascending=[False, False]).reset_index(drop=True)


def compile_singlecell_case_studies() -> pd.DataFrame:
    single = pd.read_csv(RESULTS / "singlecell" / "scpca_sample_signature_summary.tsv", sep="\t")
    celltypes = pd.read_csv(RESULTS / "singlecell" / "scpca_signature_by_celltype.tsv", sep="\t")
    celltypes["n_cells"] = pd.to_numeric(celltypes["structpde_wt_tt_cell_score_count"], errors="coerce").fillna(0).astype(int)
    total_cells = celltypes.groupby(["sample_id", "library_id"], dropna=False)["n_cells"].transform("sum").replace(0, np.nan)
    composition = celltypes[["sample_id", "library_id", "subdiagnosis", "celltype", "n_cells"]].copy()
    composition["cell_fraction"] = composition["n_cells"] / total_cells
    composition.to_csv(CASE_DIR / "scpca_celltype_composition.tsv", sep="\t", index=False)

    dominant = (
        composition.sort_values(["sample_id", "library_id", "n_cells"], ascending=[True, True, False])
        .drop_duplicates(["sample_id", "library_id"])
        .rename(columns={"celltype": "dominant_celltype", "n_cells": "dominant_celltype_n_cells", "cell_fraction": "dominant_celltype_fraction"})
    )
    high_risk_ct = (
        celltypes[celltypes["n_cells"] >= 20]
        .sort_values(["sample_id", "library_id", "structpde_wt_tt_cell_score_mean"], ascending=[True, True, False])
        .drop_duplicates(["sample_id", "library_id"])
        .rename(
            columns={
                "celltype": "top_risk_celltype",
                "structpde_wt_tt_cell_score_mean": "top_risk_celltype_score",
                "n_cells": "top_risk_celltype_n_cells",
            }
        )
    )
    program_cols = {name: name for name in HIGH_RISK_PROGRAMS}
    single["top_high_risk_program"] = single.apply(lambda row: top_label_from_columns(row, program_cols), axis=1)
    single["singlecell_case_score"] = (
        zscore(single["structpde_wt_tt_cell_score"])
        + zscore(single["anaplasia_proliferation"])
        + zscore(single["blastemal_progenitor"])
        + zscore(single["igf"])
    ) / 4.0
    single["singlecell_case_label"] = np.where(
        single["subdiagnosis"].astype(str).str.contains("anaplastic", case=False, na=False),
        "anaplastic_reference_case",
        "favorable_histology_background",
    )
    if len(single) >= 6:
        hi_score = single["singlecell_case_score"].quantile(0.85)
        single.loc[
            ~single["subdiagnosis"].astype(str).str.contains("anaplastic", case=False, na=False)
            & (single["singlecell_case_score"] >= hi_score),
            "singlecell_case_label",
        ] = "favorable_high_risk_outlier"

    case_df = (
        single.merge(
            dominant[
                [
                    "sample_id",
                    "library_id",
                    "dominant_celltype",
                    "dominant_celltype_n_cells",
                    "dominant_celltype_fraction",
                ]
            ],
            on=["sample_id", "library_id"],
            how="left",
        )
        .merge(
            high_risk_ct[
                [
                    "sample_id",
                    "library_id",
                    "top_risk_celltype",
                    "top_risk_celltype_score",
                    "top_risk_celltype_n_cells",
                ]
            ],
            on=["sample_id", "library_id"],
            how="left",
        )
        .sort_values(["singlecell_case_score", "structpde_wt_tt_cell_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    case_df.to_csv(CASE_DIR / "singlecell_case_study_candidates.tsv", sep="\t", index=False)
    return case_df


def compile_spatial_case_studies() -> pd.DataFrame:
    spatial = pd.read_csv(RESULTS / "spatial" / "scpca_spatial_library_summary.tsv", sep="\t")
    program_cols = {name: f"mean_{name}" for name in HIGH_RISK_PROGRAMS}
    spatial["top_high_risk_program"] = spatial.apply(lambda row: top_label_from_columns(row, program_cols), axis=1)
    spatial["graph_delta_vs_shuffled"] = spatial["mean_full_structpde_lr_score"] - spatial["mean_shuffled_graph_score"]
    spatial["spatial_case_score"] = (
        zscore(spatial["mean_structpde_wt_tt_spatial_score"])
        + zscore(spatial["moran_structpde_wt_tt_spatial_score"])
        + zscore(spatial["graph_delta_vs_shuffled"])
        + zscore(spatial["mean_anaplasia_proliferation"])
    ) / 4.0
    spatial["spatial_case_label"] = np.where(
        spatial["subdiagnosis"].astype(str).str.contains("anaplastic", case=False, na=False),
        "anaplastic_spatial_reference",
        "favorable_spatial_background",
    )
    if len(spatial) >= 6:
        hi_score = spatial["spatial_case_score"].quantile(0.85)
        spatial.loc[
            ~spatial["subdiagnosis"].astype(str).str.contains("anaplastic", case=False, na=False)
            & (spatial["spatial_case_score"] >= hi_score),
            "spatial_case_label",
        ] = "favorable_spatial_outlier"

    if (RESULTS / "spatial" / "scpca_spatial_lr_axis_summary.tsv").exists():
        lr_axes = pd.read_csv(RESULTS / "spatial" / "scpca_spatial_lr_axis_summary.tsv", sep="\t")
        top_lr = (
            lr_axes.sort_values(["library_id", "mean_full_activation"], ascending=[True, False])
            .drop_duplicates(["library_id"])
            .rename(
                columns={
                    "ligand": "top_spatial_ligand",
                    "receptor": "top_spatial_receptor",
                    "mean_full_activation": "top_spatial_lr_activation",
                    "mean_expression_axis": "top_spatial_expression_axis",
                }
            )
        )
        spatial = spatial.merge(
            top_lr[
                [
                    "sample_id",
                    "library_id",
                    "top_spatial_ligand",
                    "top_spatial_receptor",
                    "top_spatial_lr_activation",
                    "top_spatial_expression_axis",
                ]
            ],
            on=["sample_id", "library_id"],
            how="left",
        )
        lr_axes.sort_values(["mean_full_activation", "structure_score"], ascending=[False, False]).to_csv(
            CASE_DIR / "spatial_lr_axis_case_studies.tsv", sep="\t", index=False
        )

    hotspot_path = RESULTS / "spatial" / "scpca_spatial_hotspots.tsv.gz"
    if hotspot_path.exists():
        hotspots = pd.read_csv(hotspot_path, sep="\t")
        hotspot_summary = hotspots.groupby(["sample_id", "library_id"], dropna=False).agg(
            hotspot_max_structpde_score=("structpde_wt_tt_spatial_score", "max"),
            hotspot_mean_structpde_score=("structpde_wt_tt_spatial_score", "mean"),
            hotspot_mean_full_lr_score=("full_structpde_lr_score", "mean"),
            hotspot_top_program_score=("anaplasia_proliferation", "max"),
        ).reset_index()
        spatial = spatial.merge(hotspot_summary, on=["sample_id", "library_id"], how="left")

    spatial = spatial.sort_values(
        ["spatial_case_score", "mean_structpde_wt_tt_spatial_score"], ascending=[False, False]
    ).reset_index(drop=True)
    spatial.to_csv(CASE_DIR / "spatial_case_study_candidates.tsv", sep="\t", index=False)
    return spatial


def compile_structure_case_studies() -> pd.DataFrame:
    structure = pd.read_csv(RESULTS / "structure" / "lr_structure_scores.tsv", sep="\t")
    structure["structure_case_label"] = np.where(
        structure["structure_evidence"].eq("curated_known_axis"),
        "curated_reference_axis",
        "candidate_axis",
    )
    structure.to_csv(CASE_DIR / "structure_axis_case_studies.tsv", sep="\t", index=False)
    return structure


def compile_reference_context() -> pd.DataFrame | None:
    path = RESULTS / "fetal_reference" / "hca_fetal_reference_signature_summary.tsv"
    if not path.exists():
        return None
    ref = pd.read_csv(path, sep="\t").sort_values("mean_score", ascending=False).reset_index(drop=True)
    ref.to_csv(CASE_DIR / "fetal_reference_program_context.tsv", sep="\t", index=False)
    return ref


def compile_integrated_case_studies(single: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    spatial_by_sample = (
        spatial.sort_values(["sample_id", "spatial_case_score"], ascending=[True, False])
        .drop_duplicates(["sample_id"])
        .rename(
            columns={
                "library_id": "spatial_library_id",
                "mean_structpde_wt_tt_spatial_score": "sample_spatial_structpde_score",
                "moran_structpde_wt_tt_spatial_score": "sample_spatial_moran_i",
                "top_high_risk_program": "spatial_top_program",
                "top_spatial_ligand": "sample_top_spatial_ligand",
                "top_spatial_receptor": "sample_top_spatial_receptor",
            }
        )
    )
    single_by_sample = (
        single.sort_values(["sample_id", "singlecell_case_score"], ascending=[True, False])
        .drop_duplicates(["sample_id"])
        .rename(
            columns={
                "library_id": "singlecell_library_id",
                "structpde_wt_tt_cell_score": "sample_singlecell_structpde_score",
                "top_high_risk_program": "singlecell_top_program",
            }
        )
    )
    merged = single_by_sample.merge(
        spatial_by_sample[
            [
                "sample_id",
                "spatial_library_id",
                "sample_spatial_structpde_score",
                "sample_spatial_moran_i",
                "spatial_top_program",
                "sample_top_spatial_ligand",
                "sample_top_spatial_receptor",
                "spatial_case_score",
                "spatial_case_label",
            ]
        ],
        on="sample_id",
        how="outer",
    )
    merged["integrated_case_score"] = (
        zscore(merged["sample_singlecell_structpde_score"].fillna(merged["sample_singlecell_structpde_score"].median()))
        + zscore(merged["sample_spatial_structpde_score"].fillna(merged["sample_spatial_structpde_score"].median()))
        + zscore(merged["sample_spatial_moran_i"].fillna(merged["sample_spatial_moran_i"].median()))
    ) / 3.0
    merged["paper_case_tier"] = "supporting_case"
    if len(merged) >= 4:
        hi_score = merged["integrated_case_score"].quantile(0.75)
        merged.loc[merged["integrated_case_score"] >= hi_score, "paper_case_tier"] = "primary_case"
    merged.to_csv(CASE_DIR / "integrated_paper_case_studies.tsv", sep="\t", index=False)
    return merged.sort_values("integrated_case_score", ascending=False).reset_index(drop=True)


def write_case_study_notes(
    bulk: pd.DataFrame,
    single: pd.DataFrame,
    spatial: pd.DataFrame,
    integrated: pd.DataFrame,
    reference: pd.DataFrame | None,
) -> None:
    lines = ["# Paper-ready case study notes", ""]

    if not integrated.empty:
        lines.append("## Integrated ScPCA cases")
        for row in integrated.head(5).itertuples(index=False):
            lines.append(
                "- "
                f"{row.sample_id}: tier={row.paper_case_tier}, "
                f"singlecell_score={getattr(row, 'sample_singlecell_structpde_score', np.nan):.3f}, "
                f"spatial_score={getattr(row, 'sample_spatial_structpde_score', np.nan):.3f}, "
                f"spatial_lr={getattr(row, 'sample_top_spatial_ligand', 'NA')}-{getattr(row, 'sample_top_spatial_receptor', 'NA')}"
            )
        lines.append("")

    if not spatial.empty:
        lines.append("## Spatial highlights")
        for row in spatial.head(5).itertuples(index=False):
            lines.append(
                "- "
                f"{row.sample_id}/{row.library_id}: {row.spatial_case_label}, "
                f"mean_spatial_score={row.mean_structpde_wt_tt_spatial_score:.3f}, "
                f"moran_i={row.moran_structpde_wt_tt_spatial_score:.3f}, "
                f"top_program={row.top_high_risk_program}"
            )
        lines.append("")

    if not single.empty:
        lines.append("## Single-cell highlights")
        for row in single.head(5).itertuples(index=False):
            lines.append(
                "- "
                f"{row.sample_id}/{row.library_id}: {row.singlecell_case_label}, "
                f"cell_score={row.structpde_wt_tt_cell_score:.3f}, "
                f"risk_celltype={getattr(row, 'top_risk_celltype', 'NA')}, "
                f"top_program={row.top_high_risk_program}"
            )
        lines.append("")

    if not bulk.empty:
        lines.append("## Bulk TARGET highlights")
        for row in bulk.head(5).itertuples(index=False):
            followup = getattr(row, "followup_days", np.nan)
            lines.append(
                "- "
                f"{row.sample_id}: {row.bulk_case_label}, "
                f"bulk_case_score={row.bulk_case_score:.3f}, "
                f"structpde_activation_proxy={row.structpde_activation_proxy:.3f}, "
                f"followup_days={followup if pd.notna(followup) else 'NA'}"
            )
        lines.append("")

    if reference is not None and not reference.empty:
        lines.append("## Fetal reference context")
        for row in reference.head(3).itertuples(index=False):
            lines.append(
                "- "
                f"{row.signature}: mean_score={row.mean_score:.3f}, detected_genes={row.detected_genes}"
            )
        lines.append("")

    (CASE_DIR / "paper_case_study_notes.md").write_text("\n".join(lines), encoding="utf-8")


def update_experiment_registry(summary_counts: dict[str, int]) -> None:
    matrix_path = RESULTS / "tables" / "experiment_completion_matrix.tsv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path, sep="\t")
    else:
        matrix = pd.DataFrame(columns=["experiment", "status", "primary_output"])
    row = pd.DataFrame(
        [
            {
                "experiment": "10_case_study_and_data_export",
                "status": "completed",
                "primary_output": "results/case_studies/integrated_paper_case_studies.tsv; results/case_studies/paper_case_study_notes.md",
            }
        ]
    )
    matrix = pd.concat([matrix[matrix["experiment"] != "10_case_study_and_data_export"], row], ignore_index=True)
    matrix.to_csv(matrix_path, sep="\t", index=False)

    summary_path = RESULTS / "full_experiment_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload["case_studies"] = summary_counts
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    configure_local_cache()
    CASE_DIR.mkdir(parents=True, exist_ok=True)

    bulk = load_bulk_table()
    bulk.to_csv(RESULTS / "tables" / "target_wt_bulk_scores_by_sample.tsv", sep="\t", index=False)
    bulk.to_csv(CASE_DIR / "bulk_case_study_candidates.tsv", sep="\t", index=False)

    single = compile_singlecell_case_studies()
    spatial = compile_spatial_case_studies()
    structure = compile_structure_case_studies()
    reference = compile_reference_context()
    integrated = compile_integrated_case_studies(single, spatial)
    write_case_study_notes(bulk, single, spatial, integrated, reference)

    update_experiment_registry(
        {
            "bulk_case_candidates": int(len(bulk)),
            "singlecell_case_candidates": int(len(single)),
            "spatial_case_candidates": int(len(spatial)),
            "integrated_case_candidates": int(len(integrated)),
            "structure_axis_records": int(len(structure)),
        }
    )
    print(CASE_DIR / "integrated_paper_case_studies.tsv")


if __name__ == "__main__":
    main()
