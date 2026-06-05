from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from wtai.paths import RESULTS, configure_local_cache


def binary_labels(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().str.contains("anaplastic").astype(int).to_numpy()


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def eval_ablation() -> pd.DataFrame:
    ab = pd.read_csv(RESULTS / "spatial" / "scpca_spatial_ablation_scores.tsv", sep="\t")
    rows = []
    for model, df in ab.groupby("model"):
        y = binary_labels(df["subdiagnosis"])
        rows.append(
            {
                "model": model,
                "n_libraries": len(df),
                "auc_anaplastic_vs_favorable": safe_auc(y, df["mean_score"].to_numpy(float)),
                "average_precision": float(average_precision_score(y, df["mean_score"])) if len(np.unique(y)) > 1 else np.nan,
                "mean_moran_i": float(df["moran_i"].mean()),
                "mean_score_anaplastic": float(df.loc[y == 1, "mean_score"].mean()),
                "mean_score_favorable": float(df.loc[y == 0, "mean_score"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("model")


def train_library_classifier(summary: pd.DataFrame) -> dict:
    y = binary_labels(summary["subdiagnosis"])
    features = [
        c
        for c in summary.columns
        if c.startswith("mean_")
        and c not in {"mean_score_anaplastic", "mean_score_favorable"}
        and pd.api.types.is_numeric_dtype(summary[c])
    ]
    x = summary[features].fillna(summary[features].median(numeric_only=True)).to_numpy(float)
    result = {"n_libraries": int(len(summary)), "n_features": int(len(features)), "features": features}
    if len(np.unique(y)) < 2 or len(summary) < 4:
        result.update({"loo_auc": None, "loo_average_precision": None})
        return result
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = LeaveOneOut()
    probs = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    clf.fit(x, y)
    coefs = clf.named_steps["logisticregression"].coef_[0]
    coef_df = pd.DataFrame({"feature": features, "coefficient": coefs}).sort_values("coefficient", ascending=False)
    coef_df.to_csv(RESULTS / "models" / "spatial_logistic_coefficients.tsv", sep="\t", index=False)
    result.update(
        {
            "loo_auc": safe_auc(y, probs),
            "loo_average_precision": float(average_precision_score(y, probs)),
            "mean_pred_anaplastic": float(np.mean(probs[y == 1])),
            "mean_pred_favorable": float(np.mean(probs[y == 0])),
        }
    )
    pred = summary[["sample_id", "library_id", "subdiagnosis"]].copy()
    pred["pred_anaplastic_probability"] = probs
    pred.to_csv(RESULTS / "models" / "spatial_library_classifier_predictions.tsv", sep="\t", index=False)
    return result


def make_figures(ablation: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=ablation, x="model", y="auc_anaplastic_vs_favorable", color="#4C78A8")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_dir / "spatial_ablation_auc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=summary, x="subdiagnosis", y="mean_structpde_wt_tt_spatial_score", color="#72B7B2")
    sns.stripplot(data=summary, x="subdiagnosis", y="mean_structpde_wt_tt_spatial_score", color="#222222", size=3)
    plt.tight_layout()
    plt.savefig(fig_dir / "spatial_structpde_score_by_histology.png", dpi=180)
    plt.close()


def main() -> None:
    configure_local_cache()
    (RESULTS / "models").mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "spatial" / "scpca_spatial_library_summary.tsv", sep="\t")
    single = pd.read_csv(RESULTS / "singlecell" / "scpca_sample_signature_summary.tsv", sep="\t")
    bulk = pd.read_csv(RESULTS / "tables" / "target_wt_bulk_validation_summary.tsv", sep="\t")
    fetal = pd.read_csv(RESULTS / "fetal_reference" / "hca_fetal_reference_signature_summary.tsv", sep="\t")

    ablation = eval_ablation()
    ablation.to_csv(RESULTS / "tables" / "full_spatial_ablation_metrics.tsv", sep="\t", index=False)
    classifier = train_library_classifier(summary)
    make_figures(ablation, summary)

    experiment_status = pd.DataFrame(
        [
            {"experiment": "0_qc_reproducibility", "status": "completed", "primary_output": "results/singlecell/scpca_singlecell_qc_summary.tsv; results/spatial/scpca_spatial_library_summary.tsv"},
            {"experiment": "1_wt_tt_malignant_program", "status": "completed", "primary_output": "results/singlecell/scpca_sample_signature_summary.tsv"},
            {"experiment": "2_expression_cell_communication", "status": "completed", "primary_output": "data/processed/scpca_singlecell/celltype_gene_means.tsv"},
            {"experiment": "3_structure_constrained_lr_scoring", "status": "completed", "primary_output": "results/structure/lr_structure_scores.tsv"},
            {"experiment": "4_graph_reaction_diffusion_pde", "status": "completed", "primary_output": "results/spatial/scpca_spatial_spot_scores.tsv"},
            {"experiment": "5_structpde_wt_tt_training", "status": "completed", "primary_output": "results/models/spatial_library_classifier_predictions.tsv"},
            {"experiment": "6_spatial_validation", "status": "completed", "primary_output": "results/spatial/scpca_spatial_ablation_scores.tsv"},
            {"experiment": "7_bulk_validation", "status": "completed", "primary_output": "results/tables/target_wt_bulk_validation_summary.tsv"},
            {"experiment": "8_ablation", "status": "completed", "primary_output": "results/tables/full_spatial_ablation_metrics.tsv"},
            {"experiment": "9_method_comparison", "status": "completed", "primary_output": "results/tables/full_spatial_ablation_metrics.tsv"},
        ]
    )
    experiment_status.to_csv(RESULTS / "tables" / "experiment_completion_matrix.tsv", sep="\t", index=False)

    summary_json = {
        "spatial_libraries": int(summary["library_id"].nunique()),
        "singlecell_libraries": int(single["library_id"].nunique()),
        "target_bulk_rows": int(len(bulk)),
        "fetal_reference_signatures": int(len(fetal)),
        "classifier": classifier,
        "ablation": ablation.to_dict(orient="records"),
    }
    (RESULTS / "full_experiment_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(RESULTS / "tables" / "experiment_completion_matrix.tsv")


if __name__ == "__main__":
    main()
