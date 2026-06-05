from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from wtai.paths import RESULTS, configure_local_cache


DEVELOPMENTAL_FAMILIES = {
    "WNT",
    "BMP_TGF",
    "FGF",
    "IGF",
    "NOTCH",
    "VEGF",
    "ECM_INTEGRIN",
    "CHEMOKINE",
}

CANONICAL_SURFACE_RECEPTOR_TOKENS = (
    "CXCR",
    "ACKR",
    "IGF1R",
    "IGF2R",
    "INSR",
    "NOTCH",
    "BMPR",
    "TGFBR",
    "FGFR",
    "KDR",
    "FLT",
    "ITGA",
    "ITGB",
    "DDR",
    "RPSA",
)


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min()
    hi = values.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=series.index)
    return (values - lo) / (hi - lo)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (values - values.mean()) / std


def binary_labels(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().str.contains("anaplastic").astype(int).to_numpy()


def safe_auc(y: np.ndarray, scores: pd.Series) -> float:
    values = pd.to_numeric(scores, errors="coerce").fillna(0).to_numpy(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, values))


def surface_receptor_score(receptor: str) -> float:
    receptor = str(receptor).upper()
    if any(token in receptor for token in CANONICAL_SURFACE_RECEPTOR_TOKENS):
        return 0.90
    return 0.55


def domain_topology_score(row: pd.Series) -> float:
    family = str(row.get("family", "OTHER"))
    evidence = str(row.get("structure_evidence", ""))
    receptor = str(row.get("receptor", ""))
    if evidence == "curated_known_axis":
        return 0.95
    if family in DEVELOPMENTAL_FAMILIES:
        return 0.70 + 0.20 * surface_receptor_score(receptor)
    return 0.45 + 0.15 * surface_receptor_score(receptor)


def boltz_label_score(label: str) -> float:
    return {
        "high_confidence_complex": 0.95,
        "moderate_interface_support": 0.72,
        "fold_confident_interface_uncertain": 0.58,
        "screening_level_support": 0.46,
        "low_confidence_or_uncertain_interface": 0.18,
        "prediction_failed_or_missing": 0.0,
    }.get(str(label), 0.0)


def posterior_label(row: pd.Series) -> str:
    score = float(row["posterior_structure_score"])
    interface = float(row["interface_confidence"])
    if score >= 0.82 and interface >= 0.45:
        return "posterior_high_with_interface_support"
    if score >= 0.72:
        return "posterior_high_prior_or_domain_supported"
    if score >= 0.58:
        return "posterior_moderate_screening_supported"
    if score >= 0.42:
        return "posterior_low_uncertain"
    return "posterior_deprioritized"


def load_boltz() -> pd.DataFrame:
    path = RESULTS / "structure" / "structure_complex_confidence.tsv"
    if not path.exists():
        return pd.DataFrame(columns=["ligand", "receptor"])
    cols = [
        "ligand",
        "receptor",
        "confidence_score",
        "protein_iptm",
        "complex_iplddt",
        "complex_ipde",
        "pair_chains_iptm_offdiag_mean",
        "confidence_interpretation",
        "ligand_region",
        "receptor_region",
        "ligand_region_label",
        "receptor_region_label",
    ]
    df = pd.read_csv(path, sep="\t")
    present = [c for c in cols if c in df.columns]
    return df[present].drop_duplicates(["ligand", "receptor"]).copy()


def build_posterior_table() -> pd.DataFrame:
    lr = pd.read_csv(RESULTS / "structure" / "lr_structure_scores.tsv", sep="\t")
    boltz = load_boltz()
    df = lr.merge(boltz, on=["ligand", "receptor"], how="left")
    df["domain_topology_score"] = df.apply(domain_topology_score, axis=1)
    protein_iptm = pd.to_numeric(
        df.get("protein_iptm", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).fillna(0.0)
    pair_iptm = pd.to_numeric(
        df.get("pair_chains_iptm_offdiag_mean", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).fillna(0.0)
    df["interface_confidence"] = np.maximum(
        protein_iptm.to_numpy(float), pair_iptm.to_numpy(float)
    ).clip(0, 1)
    df["fold_confidence"] = (
        0.5 * pd.to_numeric(df.get("confidence_score"), errors="coerce").fillna(0.0)
        + 0.5 * pd.to_numeric(df.get("complex_iplddt"), errors="coerce").fillna(0.0)
    ).clip(0, 1)
    df["boltz_label_score"] = df.get("confidence_interpretation", pd.Series("", index=df.index)).map(boltz_label_score)
    has_boltz = df.get("confidence_score", pd.Series(np.nan, index=df.index)).notna()
    prior = pd.to_numeric(df["structure_score"], errors="coerce").fillna(0.5)
    posterior = pd.Series(index=df.index, dtype=float)
    posterior.loc[has_boltz] = (
        0.35 * prior.loc[has_boltz]
        + 0.25 * df.loc[has_boltz, "domain_topology_score"]
        + 0.25 * df.loc[has_boltz, "interface_confidence"]
        + 0.15 * df.loc[has_boltz, "boltz_label_score"]
    )
    posterior.loc[~has_boltz] = 0.70 * prior.loc[~has_boltz] + 0.30 * df.loc[~has_boltz, "domain_topology_score"]
    df["posterior_structure_score"] = posterior.clip(0, 1)
    df["posterior_struct_expression_score"] = df["expression_lr_score"] * df["posterior_structure_score"]
    df["posterior_delta_vs_prior"] = df["posterior_structure_score"] - df["structure_score"]
    df["posterior_evidence_label"] = df.apply(posterior_label, axis=1)
    df = df.sort_values("posterior_struct_expression_score", ascending=False).reset_index(drop=True)
    df.insert(0, "posterior_rank", np.arange(1, len(df) + 1))
    return df


def build_adjusted_spatial(posterior: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spatial_path = RESULTS / "spatial" / "scpca_spatial_lr_axis_summary.tsv"
    if not spatial_path.exists():
        return pd.DataFrame(), pd.DataFrame(), {}
    axes = pd.read_csv(spatial_path, sep="\t")
    post_cols = [
        "ligand",
        "receptor",
        "posterior_structure_score",
        "posterior_struct_expression_score",
        "posterior_evidence_label",
        "confidence_score",
        "protein_iptm",
        "complex_iplddt",
        "pair_chains_iptm_offdiag_mean",
    ]
    axes = axes.merge(posterior[[c for c in post_cols if c in posterior.columns]], on=["ligand", "receptor"], how="left")
    axes["posterior_structure_score"] = axes["posterior_structure_score"].fillna(axes["structure_score"])
    prior = pd.to_numeric(axes["structure_score"], errors="coerce").replace(0, np.nan).fillna(0.5)
    ratio = (axes["posterior_structure_score"] / prior).clip(0.25, 1.5)
    axes["posterior_adjusted_activation"] = axes["mean_full_activation"] * ratio
    axes["posterior_adjustment_ratio"] = ratio

    library = (
        axes.groupby(["sample_id", "library_id", "subdiagnosis"], dropna=False)
        .agg(
            n_lr_axes=("ligand", "count"),
            mean_prior_lr_activation=("mean_full_activation", "mean"),
            mean_posterior_lr_activation=("posterior_adjusted_activation", "mean"),
            mean_posterior_structure_score=("posterior_structure_score", "mean"),
            max_posterior_axis_activation=("posterior_adjusted_activation", "max"),
        )
        .reset_index()
    )
    top_axis = (
        axes.sort_values(["sample_id", "library_id", "posterior_adjusted_activation"], ascending=[True, True, False])
        .drop_duplicates(["sample_id", "library_id"])
        .rename(
            columns={
                "ligand": "top_posterior_ligand",
                "receptor": "top_posterior_receptor",
                "posterior_adjusted_activation": "top_posterior_activation",
                "posterior_evidence_label": "top_posterior_evidence_label",
            }
        )
    )
    library = library.merge(
        top_axis[
            [
                "sample_id",
                "library_id",
                "top_posterior_ligand",
                "top_posterior_receptor",
                "top_posterior_activation",
                "top_posterior_evidence_label",
            ]
        ],
        on=["sample_id", "library_id"],
        how="left",
    )
    y = binary_labels(library["subdiagnosis"])
    metrics = {
        "n_libraries": int(len(library)),
        "mean_prior_lr_activation": float(library["mean_prior_lr_activation"].mean()),
        "mean_posterior_lr_activation": float(library["mean_posterior_lr_activation"].mean()),
        "auc_prior_lr_activation": safe_auc(y, library["mean_prior_lr_activation"]),
        "auc_posterior_lr_activation": safe_auc(y, library["mean_posterior_lr_activation"]),
        "ap_prior_lr_activation": float(average_precision_score(y, library["mean_prior_lr_activation"]))
        if len(np.unique(y)) > 1
        else float("nan"),
        "ap_posterior_lr_activation": float(average_precision_score(y, library["mean_posterior_lr_activation"]))
        if len(np.unique(y)) > 1
        else float("nan"),
        "top_posterior_axes": (
            library.assign(axis=library["top_posterior_ligand"] + "-" + library["top_posterior_receptor"])
            .groupby("axis")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        ),
    }
    return axes, library, metrics


def write_report(posterior: pd.DataFrame, metrics: dict, out_path: Path) -> None:
    counts = posterior["posterior_evidence_label"].value_counts().to_dict()
    lines = [
        "# Structure evidence posterior rescoring",
        "",
        "This report combines the original curated/domain prior with Boltz-2 complex confidence and topology/domain accessibility.",
        "",
        "## Posterior evidence counts",
        "",
    ]
    for label, n in counts.items():
        lines.append(f"- {label}: {n}")
    lines.extend(
        [
            "",
            "## Spatial reweighting metrics",
            "",
            f"- n_libraries: {metrics.get('n_libraries', 'NA')}",
            f"- mean_prior_lr_activation: {metrics.get('mean_prior_lr_activation', float('nan')):.4f}",
            f"- mean_posterior_lr_activation: {metrics.get('mean_posterior_lr_activation', float('nan')):.4f}",
            f"- auc_prior_lr_activation: {metrics.get('auc_prior_lr_activation', float('nan')):.4f}",
            f"- auc_posterior_lr_activation: {metrics.get('auc_posterior_lr_activation', float('nan')):.4f}",
            "",
            "## Top posterior LR axes",
            "",
            "| Rank | Axis | Family | Prior | Posterior | Interface | Boltz label | Posterior structural expression |",
            "|---:|---|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in posterior.head(20).itertuples(index=False):
        lines.append(
            f"| {row.posterior_rank} | {row.ligand}-{row.receptor} | {row.family} | "
            f"{row.structure_score:.3f} | {row.posterior_structure_score:.3f} | "
            f"{row.interface_confidence:.3f} | {getattr(row, 'confidence_interpretation', '')} | "
            f"{row.posterior_struct_expression_score:.3f} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def update_summary(posterior: pd.DataFrame, metrics: dict) -> None:
    summary_path = RESULTS / "full_experiment_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload["structure_evidence_rescoring"] = {
        "experiment": "12_structure_evidence_rescoring",
        "n_lr_axes": int(len(posterior)),
        "posterior_label_counts": posterior["posterior_evidence_label"].value_counts().to_dict(),
        "mean_prior_structure_score": float(posterior["structure_score"].mean()),
        "mean_posterior_structure_score": float(posterior["posterior_structure_score"].mean()),
        "spatial_reweighting": metrics,
        "top_posterior_axes": posterior[
            [
                "ligand",
                "receptor",
                "family",
                "posterior_structure_score",
                "posterior_struct_expression_score",
                "posterior_evidence_label",
            ]
        ]
        .head(10)
        .to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    configure_local_cache()
    out_dir = RESULTS / "structure"
    posterior = build_posterior_table()
    posterior.to_csv(out_dir / "structure_evidence_posterior.tsv", sep="\t", index=False)
    axes, library, metrics = build_adjusted_spatial(posterior)
    if not axes.empty:
        axes.to_csv(out_dir / "structure_adjusted_spatial_axis_summary.tsv", sep="\t", index=False)
        library.to_csv(out_dir / "structure_adjusted_spatial_library_summary.tsv", sep="\t", index=False)
    (out_dir / "structure_evidence_rescoring_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(posterior, metrics, out_dir / "structure_evidence_rescoring_report.md")
    update_summary(posterior, metrics)
    print(out_dir / "structure_evidence_posterior.tsv")


if __name__ == "__main__":
    main()
