from __future__ import annotations

from pathlib import Path

import pandas as pd


HIGH_CONFIDENCE_KNOWN_AXES = {
    ("IGF2", "IGF1R"),
    ("IGF1", "IGF1R"),
    ("VEGFA", "KDR"),
    ("VEGFA", "FLT1"),
    ("JAG1", "NOTCH1"),
    ("JAG1", "NOTCH2"),
    ("DLL4", "NOTCH1"),
    ("TGFB1", "TGFBR1"),
    ("BMP4", "BMPR1A"),
    ("BMP7", "BMPR1A"),
    ("SPP1", "ITGB1"),
    ("CXCL12", "CXCR4"),
}


def score_structure(lr_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in lr_scores.itertuples(index=False):
        ligand = row.ligand
        receptor = row.receptor
        known = (ligand, receptor) in HIGH_CONFIDENCE_KNOWN_AXES
        family = row.family
        if known:
            structure_score = 0.95
            evidence = "curated_known_axis"
        elif family != "OTHER":
            structure_score = 0.72
            evidence = "developmental_family_afdb_candidate"
        else:
            structure_score = 0.50
            evidence = "expression_only_candidate"
        rows.append(
            {
                **row._asdict(),
                "structure_score": structure_score,
                "structure_evidence": evidence,
                "alphafold_ligand_url": f"https://alphafold.ebi.ac.uk/search/text/{ligand}",
                "alphafold_receptor_url": f"https://alphafold.ebi.ac.uk/search/text/{receptor}",
                "struct_expression_score": row.expression_lr_score * structure_score,
            }
        )
    return pd.DataFrame(rows).sort_values("struct_expression_score", ascending=False).reset_index(drop=True)


def save_structure_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "lr_structure_scores.tsv", sep="\t", index=False)
    failed = df[df["structure_evidence"].eq("expression_only_candidate")]
    failed.to_csv(out_dir / "failed_or_low_confidence_pairs.tsv", sep="\t", index=False)

