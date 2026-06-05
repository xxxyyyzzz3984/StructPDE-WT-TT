from __future__ import annotations

import pandas as pd

from wtai.eval.bulk_validation import geo_group_discovery, save_validation, target_bulk_summary
from wtai.paths import PROCESSED, RESULTS, configure_local_cache


def main() -> None:
    configure_local_cache()
    sig = pd.read_csv(PROCESSED / "target_wt" / "target_wt_signature_scores.tsv", sep="\t", index_col=0)
    pde = pd.read_csv(RESULTS / "pde" / "target_wt_structpde_activation_score.tsv", sep="\t", index_col=0).iloc[:, 0]
    clinical = pd.read_csv(PROCESSED / "target_wt" / "target_wt_clinical.tsv", sep="\t")
    summary = target_bulk_summary(sig, pde, clinical)
    save_validation(summary, RESULTS / "tables")
    geo_rows = []
    for score_path in (PROCESSED / "geo").glob("GSE*_signature_scores.tsv"):
        acc = score_path.name.split("_")[0]
        meta_path = PROCESSED / "geo" / f"{acc}_metadata.tsv"
        geo = geo_group_discovery(score_path, meta_path)
        if not geo.empty:
            geo.insert(0, "cohort", acc)
            geo_rows.append(geo)
    if geo_rows:
        pd.concat(geo_rows, ignore_index=True).to_csv(
            RESULTS / "tables" / "geo_text_label_validation_summary.tsv", sep="\t", index=False
        )
    print("Bulk validation complete.")


if __name__ == "__main__":
    main()

