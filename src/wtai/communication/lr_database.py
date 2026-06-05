from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


DEVELOPMENTAL_FAMILIES = {
    "WNT": ["WNT", "FZD", "LRP5", "LRP6"],
    "BMP_TGF": ["BMP", "BMPR", "TGFB", "TGFBR"],
    "FGF": ["FGF", "FGFR"],
    "IGF": ["IGF", "IGF1R", "IGF2R", "INSR"],
    "NOTCH": ["JAG", "DLL", "NOTCH"],
    "VEGF": ["VEGF", "KDR", "FLT"],
    "ECM_INTEGRIN": ["COL", "FN1", "LAM", "ITGA", "ITGB", "SPP1"],
    "CHEMOKINE": ["CXCL", "CXCR", "CCL", "CCR"],
}


def load_omnipath_lr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "source_genesymbol" not in df.columns:
        df["source_genesymbol"] = df["source"]
        df["target_genesymbol"] = df["target"]
    out = df.rename(columns={"source_genesymbol": "ligand", "target_genesymbol": "receptor"}).copy()
    out["ligand"] = out["ligand"].astype(str).str.upper()
    out["receptor"] = out["receptor"].astype(str).str.upper()
    out = out[out["ligand"].ne(out["receptor"])]
    out = out.drop_duplicates(["ligand", "receptor"])
    out["family"] = out.apply(lambda row: family_label(row["ligand"], row["receptor"]), axis=1)
    return out


def family_label(ligand: str, receptor: str) -> str:
    pair = f"{ligand}|{receptor}"
    for label, tokens in DEVELOPMENTAL_FAMILIES.items():
        if any(token in pair for token in tokens):
            return label
    return "OTHER"


def expressed_lr_scores(expr: pd.DataFrame, lr: pd.DataFrame, top_n: int = 250) -> pd.DataFrame:
    genes = {g.upper(): g for g in expr.index.astype(str)}
    rows = []
    expr_log = np.log2(expr.astype(float) + 1.0)
    mean_expr = expr_log.mean(axis=1)
    for row in lr.itertuples(index=False):
        ligand = getattr(row, "ligand")
        receptor = getattr(row, "receptor")
        if ligand not in genes or receptor not in genes:
            continue
        lval = float(mean_expr.loc[genes[ligand]])
        rval = float(mean_expr.loc[genes[receptor]])
        expression_score = (lval * rval) ** 0.5 if lval > 0 and rval > 0 else 0.0
        family_bonus = 1.25 if getattr(row, "family") != "OTHER" else 1.0
        evidence = float(getattr(row, "curation_effort", 1) or 1)
        score = expression_score * family_bonus * (1.0 + min(evidence, 25.0) / 25.0)
        rows.append(
            {
                "ligand": ligand,
                "receptor": receptor,
                "family": getattr(row, "family"),
                "mean_ligand_log2_tpm": lval,
                "mean_receptor_log2_tpm": rval,
                "curation_effort": evidence,
                "expression_lr_score": score,
                "sources": getattr(row, "sources", ""),
                "references": getattr(row, "references", ""),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("expression_lr_score", ascending=False).head(top_n).reset_index(drop=True)
