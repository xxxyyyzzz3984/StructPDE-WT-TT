from __future__ import annotations

import pandas as pd

from wtai.communication.lr_database import expressed_lr_scores, load_omnipath_lr
from wtai.paths import EXTERNAL, PROCESSED, RESULTS, configure_local_cache
from wtai.structure.structure_score import save_structure_outputs, score_structure


def main() -> None:
    configure_local_cache()
    expr = pd.read_csv(PROCESSED / "target_wt" / "target_wt_tpm.tsv", sep="\t", index_col=0)
    lr = load_omnipath_lr(EXTERNAL / "ligand_receptor" / "omnipath_ligrecextra.tsv")
    lr_scores = expressed_lr_scores(expr, lr, top_n=250)
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    lr_scores.to_csv(out / "target_wt_lr_expression_scores.tsv", sep="\t", index=False)
    struct = score_structure(lr_scores)
    structure_dir = RESULTS / "structure"
    save_structure_outputs(struct, structure_dir)
    struct.head(100).to_csv(structure_dir / "lr_structure_scores_top100.tsv", sep="\t", index=False)
    struct[struct["structure_evidence"].eq("curated_known_axis")].to_csv(
        structure_dir / "high_confidence_known_axes.tsv", sep="\t", index=False
    )
    print("LR and structure scoring complete.")


if __name__ == "__main__":
    main()
