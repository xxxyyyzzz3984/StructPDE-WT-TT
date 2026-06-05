from __future__ import annotations

from wtai.data.preprocess_bulk import build_target_matrix, score_signatures, write_geo_processed
from wtai.paths import PROCESSED, RAW, configure_local_cache


def main() -> None:
    configure_local_cache()
    target_expr, _ = build_target_matrix(RAW / "target_wt", PROCESSED / "target_wt")
    score_signatures(target_expr).to_csv(PROCESSED / "target_wt" / "target_wt_signature_scores.tsv", sep="\t")
    write_geo_processed(RAW / "geo", PROCESSED / "geo")
    print("Bulk preprocessing complete.")


if __name__ == "__main__":
    main()

