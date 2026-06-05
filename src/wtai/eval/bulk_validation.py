from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def target_bulk_summary(signature_scores: pd.DataFrame, pde_score: pd.Series, clinical: pd.DataFrame) -> pd.DataFrame:
    merged = signature_scores.join(pde_score, how="left")
    merged["sample"] = merged.index
    clin = clinical.rename(columns={"submitter_id": "sample"})
    merged = merged.merge(clin, on="sample", how="left")
    rows = []
    score_cols = list(signature_scores.columns) + [pde_score.name]
    numeric_outcome = pd.to_numeric(merged.get("days_to_last_follow_up"), errors="coerce")
    for col in score_cols:
        vals = pd.to_numeric(merged[col], errors="coerce")
        ok = vals.notna() & numeric_outcome.notna()
        if ok.sum() >= 5:
            rho, p = stats.spearmanr(vals[ok], numeric_outcome[ok])
        else:
            rho, p = np.nan, np.nan
        rows.append({"score": col, "n": int(ok.sum()), "spearman_days_followup": rho, "pvalue": p})
    return pd.DataFrame(rows).sort_values("pvalue", na_position="last")


def geo_group_discovery(score_path: Path, meta_path: Path) -> pd.DataFrame:
    scores = pd.read_csv(score_path, sep="\t", index_col=0)
    meta = pd.read_csv(meta_path, sep="\t", index_col=0)
    text = meta.astype(str).agg(" | ".join, axis=1).str.lower()
    labels = pd.Series(np.nan, index=meta.index, dtype=float)
    labels[text.str.contains("wilms|tumou?r|tumor|neoplasm", regex=True)] = 1.0
    labels[text.str.contains("normal|control|adjacent", regex=True)] = 0.0
    rows = []
    for col in scores.columns:
        common = scores.index.intersection(labels.dropna().index)
        if labels.loc[common].nunique() == 2:
            try:
                auc = roc_auc_score(labels.loc[common], scores.loc[common, col])
            except ValueError:
                auc = np.nan
            rows.append({"score": col, "n": len(common), "auc_text_label_tumor_vs_normal": auc})
    return pd.DataFrame(rows)


def save_validation(target_summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_summary.to_csv(out_dir / "target_wt_bulk_validation_summary.tsv", sep="\t", index=False)
