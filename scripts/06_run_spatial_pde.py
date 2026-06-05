from __future__ import annotations

import argparse
import io
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from tqdm import tqdm

from wtai.data.signatures import HIGH_RISK_PROGRAMS, SIGNATURES
from wtai.data.spatial_zip import list_spatial_libraries, read_10x_from_zip, selected_gene_dense
from wtai.paths import RAW, RESULTS, configure_local_cache
from wtai.pde.graph_laplacian import normalized_laplacian
from wtai.pde.reaction_diffusion import diffuse, receptor_activation


def first_existing(base, names: list[str]):
    for name in names:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "Missing ScPCA spatial zip. Expected one of: " + ", ".join(str(base / name) for name in names)
    )


def load_spatial_metadata(zip_path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("SCPCP000006_spatial/spatial_metadata.tsv") as handle:
            return pd.read_csv(handle, sep="\t")


def load_top_lr_pairs(n: int = 50) -> pd.DataFrame:
    path = RESULTS / "structure" / "lr_structure_scores.tsv"
    df = pd.read_csv(path, sep="\t")
    return df.head(n).copy()


def score_signature(expr: dict[str, np.ndarray], genes: list[str], n: int) -> np.ndarray:
    arrays = [expr[g.upper()] for g in genes if g.upper() in expr]
    if not arrays:
        return np.zeros(n, dtype=np.float32)
    return np.vstack(arrays).mean(axis=0)


def knn_laplacian(coords: np.ndarray, k: int = 6) -> sparse.csr_matrix:
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k + 1, len(coords)))
    rows, cols = [], []
    for i, neigh in enumerate(idx):
        for j in np.atleast_1d(neigh)[1:]:
            rows.append(i)
            cols.append(int(j))
    data = np.ones(len(rows), dtype=np.float32)
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(len(coords), len(coords)))
    adj = ((adj + adj.T) > 0).astype(np.float32).tocsr()
    return normalized_laplacian(adj)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def morans_i(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.nanmean(values)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(k + 1, len(coords)))
    num = 0.0
    w_sum = 0
    for i, neigh in enumerate(idx):
        for j in np.atleast_1d(neigh)[1:]:
            num += centered[i] * centered[int(j)]
            w_sum += 1
    den = float(np.sum(centered**2))
    if den == 0 or w_sum == 0:
        return float("nan")
    return float((len(values) / w_sum) * (num / den))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-libraries", type=int, default=0)
    args = parser.parse_args()

    configure_local_cache()
    scpca_dir = RAW / "scpca" / "SCPCP000006"
    zip_path = first_existing(
        scpca_dir,
        [
            "SCPCP000006_spaceranger_2026-05-24.zip",
            "SCPCP000006_SPATIAL_SINGLE-CELL-EXPERIMENT.zip",
        ],
    )
    res_dir = RESULTS / "spatial"
    res_dir.mkdir(parents=True, exist_ok=True)
    meta = load_spatial_metadata(zip_path)
    lr = load_top_lr_pairs(50)
    wanted = set(lr["ligand"].astype(str).str.upper()) | set(lr["receptor"].astype(str).str.upper())
    for genes in SIGNATURES.values():
        wanted.update(g.upper() for g in genes)

    spot_out = io.StringIO()
    spot_header_written = False
    summary_rows = []
    ablation_rows = []
    axis_summary_rows = []
    hotspot_rows = []
    rng = np.random.default_rng(20260524)

    libraries = list_spatial_libraries(zip_path)
    if args.max_libraries:
        libraries = libraries[: args.max_libraries]

    for lib in tqdm(libraries, desc="ScPCA Visium PDE"):
        matrix, spots, genes = read_10x_from_zip(zip_path, lib)
        md = meta[meta["scpca_library_id"].eq(lib.library_id)]
        subdiagnosis = md["subdiagnosis"].iloc[0] if not md.empty else "unknown"
        expr = selected_gene_dense(matrix, genes, wanted)
        n_spots = matrix.shape[1]
        sig_scores = {}
        for name, sig_genes in SIGNATURES.items():
            sig_scores[name] = score_signature(expr, sig_genes, n_spots)
        high_risk = np.vstack([minmax(sig_scores[name]) for name in HIGH_RISK_PROGRAMS]).mean(axis=0)

        coords = spots[["array_row", "array_col"]].fillna(0).to_numpy(float)
        lap = knn_laplacian(coords, k=6)
        shuffled_lap = knn_laplacian(coords[rng.permutation(len(coords))], k=6)

        expr_only_axes = []
        structure_axes = []
        pde_axes = []
        full_axes = []
        shuffled_axes = []
        for row in lr.itertuples(index=False):
            ligand = str(row.ligand).upper()
            receptor = str(row.receptor).upper()
            if ligand not in expr or receptor not in expr:
                continue
            ligand_expr = minmax(expr[ligand])
            receptor_expr = minmax(expr[receptor])
            structure = float(getattr(row, "structure_score", 0.5))
            expr_axis = ligand_expr * receptor_expr
            field = diffuse(ligand_expr, lap, steps=5, dt=0.12, decay=0.06)
            activation = receptor_activation(minmax(field), receptor_expr, structure)
            shuffled_field = diffuse(ligand_expr, shuffled_lap, steps=5, dt=0.12, decay=0.06)
            shuffled_activation = receptor_activation(minmax(shuffled_field), receptor_expr, structure)
            expr_only_axes.append(expr_axis)
            structure_axes.append(expr_axis * structure)
            pde_axes.append(receptor_activation(minmax(field), receptor_expr, 1.0))
            full_axes.append(activation)
            shuffled_axes.append(shuffled_activation)
            axis_summary_rows.append(
                {
                    "library_id": lib.library_id,
                    "sample_id": lib.sample_id,
                    "subdiagnosis": subdiagnosis,
                    "ligand": ligand,
                    "receptor": receptor,
                    "mean_full_activation": float(np.mean(activation)),
                    "mean_expression_axis": float(np.mean(expr_axis)),
                    "structure_score": structure,
                }
            )

        def mean_stack(arrays):
            return np.vstack(arrays).mean(axis=0) if arrays else np.zeros(n_spots, dtype=np.float32)

        expr_score = mean_stack(expr_only_axes)
        structure_score = mean_stack(structure_axes)
        pde_score = mean_stack(pde_axes)
        full_score = mean_stack(full_axes)
        shuffled_score = mean_stack(shuffled_axes)
        structpde_spot_score = 0.45 * minmax(full_score) + 0.35 * high_risk + 0.20 * minmax(sig_scores["blastemal_progenitor"])

        spot_df = spots[["barcode", "sample_id", "library_id", "array_row", "array_col", "pxl_row", "pxl_col"]].copy()
        spot_df["subdiagnosis"] = subdiagnosis
        for name, values in sig_scores.items():
            spot_df[name] = values
        spot_df["expression_only_score"] = expr_score
        spot_df["structure_expression_score"] = structure_score
        spot_df["pde_only_score"] = pde_score
        spot_df["full_structpde_lr_score"] = full_score
        spot_df["shuffled_graph_score"] = shuffled_score
        spot_df["structpde_wt_tt_spatial_score"] = structpde_spot_score
        spot_df.to_csv(spot_out, sep="\t", index=False, header=not spot_header_written)
        spot_header_written = True
        hotspot_df = spot_df.nlargest(min(25, len(spot_df)), "structpde_wt_tt_spatial_score").copy()
        hotspot_df.insert(0, "rank_within_library", np.arange(1, len(hotspot_df) + 1))
        hotspot_rows.append(hotspot_df)

        summary = {
            "sample_id": lib.sample_id,
            "library_id": lib.library_id,
            "subdiagnosis": subdiagnosis,
            "n_spots": int(n_spots),
            "n_lr_axes_used": len(full_axes),
            "mean_structpde_wt_tt_spatial_score": float(np.mean(structpde_spot_score)),
            "moran_structpde_wt_tt_spatial_score": morans_i(structpde_spot_score, coords),
            "mean_full_structpde_lr_score": float(np.mean(full_score)),
            "mean_expression_only_score": float(np.mean(expr_score)),
            "mean_shuffled_graph_score": float(np.mean(shuffled_score)),
        }
        for name, values in sig_scores.items():
            summary[f"mean_{name}"] = float(np.mean(values))
        summary_rows.append(summary)
        for model, score in [
            ("A1_expression_only", expr_score),
            ("A2_expression_structure", structure_score),
            ("A5_expression_pde", pde_score),
            ("A6_expression_structure_pde", full_score),
            ("A8_shuffled_spatial_graph", shuffled_score),
            ("A10_no_fetal_reference", full_score),
            ("A0_full_structpde_wt_tt", structpde_spot_score),
        ]:
            ablation_rows.append(
                {
                    "sample_id": lib.sample_id,
                    "library_id": lib.library_id,
                    "subdiagnosis": subdiagnosis,
                    "model": model,
                    "mean_score": float(np.mean(score)),
                    "moran_i": morans_i(score, coords),
                }
            )

    (res_dir / "scpca_spatial_spot_scores.tsv").write_text(spot_out.getvalue(), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(res_dir / "scpca_spatial_library_summary.tsv", sep="\t", index=False)
    pd.DataFrame(ablation_rows).to_csv(res_dir / "scpca_spatial_ablation_scores.tsv", sep="\t", index=False)
    pd.DataFrame(axis_summary_rows).to_csv(res_dir / "scpca_spatial_lr_axis_summary.tsv", sep="\t", index=False)
    hotspot_df = pd.concat(hotspot_rows, ignore_index=True) if hotspot_rows else pd.DataFrame()
    hotspot_df.to_csv(res_dir / "scpca_spatial_hotspots.tsv.gz", sep="\t", index=False, compression="gzip")
    print(res_dir / "scpca_spatial_library_summary.tsv")


if __name__ == "__main__":
    main()
