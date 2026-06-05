from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from wtai.data.signatures import HIGH_RISK_PROGRAMS, SIGNATURES
from wtai.paths import RAW, RESULTS, configure_local_cache


def load_hca_rows(root: Path) -> pd.DataFrame:
    rows = pd.read_csv(root / "tableOfCounts_rowLabels.tsv", sep="\t")
    rows["Symbol"] = rows["Symbol"].astype(str).str.upper()
    return rows


def stream_signature_counts(root: Path, rows: pd.DataFrame) -> pd.DataFrame:
    row_to_signature: dict[int, list[str]] = {}
    for name, genes in SIGNATURES.items():
        wanted = {g.upper() for g in genes}
        for row_number in rows.loc[rows["Symbol"].isin(wanted), "RowNumber"]:
            row_to_signature.setdefault(int(row_number), []).append(name)

    col_meta = pd.read_csv(root / "tableOfCounts_colLabels.tsv", sep="\t", usecols=["ColNumber", "SangerID"])
    n_cols = int(col_meta["ColNumber"].max())
    sums = {name: np.zeros(n_cols, dtype=np.float64) for name in SIGNATURES}
    nnz = {name: np.zeros(n_cols, dtype=np.int32) for name in SIGNATURES}
    gene_hits = {name: set() for name in SIGNATURES}
    symbol_by_row = dict(zip(rows["RowNumber"].astype(int), rows["Symbol"]))

    mtx = root / "tableOfCounts.mtx"
    with mtx.open("r", encoding="utf-8", errors="ignore") as handle:
        dimensions_seen = False
        for line in handle:
            if line.startswith("%"):
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                if not dimensions_seen:
                    dimensions_seen = True
                    continue
                # MatrixMarket data line after the dimensions record.
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    value = float(parts[2])
                except ValueError:
                    continue
                if i not in row_to_signature:
                    continue
                col = j - 1
                if col < 0 or col >= n_cols:
                    continue
                val = math.log1p(value)
                for sig in row_to_signature[i]:
                    sums[sig][col] += val
                    nnz[sig][col] += 1
                    gene_hits[sig].add(symbol_by_row.get(i, str(i)))

    score_df = pd.DataFrame({"ColNumber": np.arange(1, n_cols + 1)})
    for name in SIGNATURES:
        denom = max(1, len({g.upper() for g in SIGNATURES[name]} & set(rows["Symbol"])))
        score_df[name] = sums[name] / denom
        score_df[f"{name}_nnz"] = nnz[name]
    score_df = score_df.merge(col_meta, on="ColNumber", how="left")
    return score_df, gene_hits


def main() -> None:
    configure_local_cache()
    root = RAW / "fetal_kidney_hca" / "d8ae869c-39c2-4cdd-b3fc-2d0d8f60e7b8"
    res_dir = RESULTS / "fetal_reference"
    res_dir.mkdir(parents=True, exist_ok=True)
    rows = load_hca_rows(root)
    score_df, gene_hits = stream_signature_counts(root, rows)
    score_df["fetal_reference_structpde_score"] = score_df[HIGH_RISK_PROGRAMS].mean(axis=1)

    summary = []
    for name in SIGNATURES:
        summary.append(
            {
                "signature": name,
                "genes_requested": len(SIGNATURES[name]),
                "genes_detected": len(gene_hits[name]),
                "mean_score": float(score_df[name].mean()),
                "median_score": float(score_df[name].median()),
                "p95_score": float(score_df[name].quantile(0.95)),
                "detected_genes": ";".join(sorted(gene_hits[name])),
            }
        )
    score_df.groupby("SangerID", dropna=False)[HIGH_RISK_PROGRAMS + ["fetal_reference_structpde_score"]].mean().reset_index().to_csv(
        res_dir / "hca_fetal_reference_signature_by_sangerid.tsv", sep="\t", index=False
    )
    pd.DataFrame(summary).to_csv(res_dir / "hca_fetal_reference_signature_summary.tsv", sep="\t", index=False)
    print(res_dir / "hca_fetal_reference_signature_summary.tsv")


if __name__ == "__main__":
    main()
