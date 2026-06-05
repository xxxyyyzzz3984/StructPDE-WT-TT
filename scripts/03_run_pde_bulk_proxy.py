from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from wtai.pde.graph_laplacian import knn_graph, normalized_laplacian
from wtai.pde.reaction_diffusion import diffuse, receptor_activation
from wtai.paths import PROCESSED, RESULTS, configure_local_cache


def main() -> None:
    configure_local_cache()
    expr = pd.read_csv(PROCESSED / "target_wt" / "target_wt_tpm.tsv", sep="\t", index_col=0)
    lr = pd.read_csv(RESULTS / "structure" / "lr_structure_scores.tsv", sep="\t").head(50)
    x = np.log2(expr.T + 1.0)
    features = PCA(n_components=min(20, x.shape[0] - 1), random_state=20260524).fit_transform(
        StandardScaler().fit_transform(x)
    )
    lap = normalized_laplacian(knn_graph(features, k=12))
    gene_map = {g.upper(): g for g in expr.index.astype(str)}
    activations = {}
    for row in lr.itertuples(index=False):
        if row.ligand not in gene_map or row.receptor not in gene_map:
            continue
        ligand = np.log2(expr.loc[gene_map[row.ligand]].astype(float).to_numpy() + 1.0)
        receptor = np.log2(expr.loc[gene_map[row.receptor]].astype(float).to_numpy() + 1.0)
        field = diffuse(ligand, lap, steps=5)
        activations[f"{row.ligand}-{row.receptor}"] = receptor_activation(field, receptor, row.structure_score)
    act = pd.DataFrame(activations, index=expr.columns)
    out = RESULTS / "pde"
    out.mkdir(parents=True, exist_ok=True)
    act.to_csv(out / "target_wt_lr_activation_fields.tsv", sep="\t")
    if not act.empty:
        score = act.mean(axis=1).rename("structpde_activation_proxy")
        score.to_csv(out / "target_wt_structpde_activation_score.tsv", sep="\t")
    print("Bulk graph PDE proxy complete.")


if __name__ == "__main__":
    main()

