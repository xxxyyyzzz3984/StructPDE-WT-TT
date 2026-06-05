from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec



CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT.parent if CODE_ROOT.name in {"codes", "published_codes"} else CODE_ROOT
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
FIG_DIR = PAPER / "figures"


plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


COLORS = {
    "navy": "#12355B",
    "teal": "#00A6A6",
    "coral": "#F26D5B",
    "gold": "#E4B363",
    "green": "#5B8C5A",
    "blue": "#3C6EAA",
    "purple": "#7B5EA7",
    "gray": "#6B7280",
    "light": "#F5F7FA",
    "ink": "#1F2937",
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def mm(v: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def savefig(fig: plt.Figure, name: str) -> Path:
    out = FIG_DIR / name
    fig.savefig(out, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def fig1_method_overview() -> Path:
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(
        9,
        9.35,
        "StructPDE-WT-TT：结构约束图反应扩散细胞通讯模型\n"
        "Structure-constrained graph reaction-diffusion cell-communication model",
        ha="center",
        va="center",
        fontsize=22,
        color=COLORS["navy"],
        fontweight="bold",
    )

    modules = [
        (0.6, 6.8, 3.2, 1.35, "多模态WT数据\nMulti-modal WT data", "snRNA-seq | Visium | TARGET\nFetal kidney reference", COLORS["blue"]),
        (4.1, 6.8, 3.2, 1.35, "发育程序\nDevelopmental programs", "Blastemal | ECM | NOTCH\nIGF | BMP/TGF | immune", COLORS["green"]),
        (7.6, 6.8, 3.2, 1.35, "配体-受体先验\nLigand-receptor prior", "CellPhoneDB | CellChat\nOmniPath | LIANA", COLORS["gold"]),
        (11.1, 6.8, 3.2, 1.35, "结构证据层\nStructural evidence", "PDB | AlphaFold DB\nBoltz-2 | posterior score", COLORS["coral"]),
        (14.6, 6.8, 2.8, 1.35, "验证\nValidation", "Spatial | LOO classifier\nTARGET bulk proxy", COLORS["purple"]),
        (3.0, 3.8, 4.0, 1.55, "图反应扩散PDE\nGraph reaction-diffusion PDE", "source q_l  → diffusion f_l\nreceptor activation a_lr,i", COLORS["teal"]),
        (8.0, 3.6, 4.2, 1.85, "WT转变吸引场\nWT transition attractor field", "Z_i integrates LR activation,\ndevelopmental-risk programs,\nand blastemal state", COLORS["navy"]),
        (13.0, 3.8, 4.0, 1.55, "候选机制\nCandidate axes", "CXCL12-CXCR4 | FN1-ITGB1\nMDK-ITGB1 | LAMB2-RPSA", COLORS["green"]),
    ]

    for x, y, w, h, title, body, color in modules:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.18",
            linewidth=1.4,
            edgecolor=color,
            facecolor="white",
            alpha=0.98,
        )
        ax.add_patch(box)
        ax.add_patch(Circle((x + 0.32, y + h - 0.32), 0.12, color=color, alpha=0.95))
        ax.text(x + w / 2, y + h - 0.38, title, ha="center", va="center", fontsize=13, color=color, fontweight="bold")
        ax.text(x + w / 2, y + 0.46, body, ha="center", va="center", fontsize=10.5, color=COLORS["ink"])

    arrows = [
        ((3.8, 7.45), (4.1, 7.45)),
        ((7.3, 7.45), (7.6, 7.45)),
        ((10.8, 7.45), (11.1, 7.45)),
        ((14.3, 7.45), (14.6, 7.45)),
        ((5.7, 6.8), (5.2, 5.35)),
        ((9.2, 6.8), (9.7, 5.45)),
        ((12.7, 6.8), (10.9, 5.45)),
        ((7.0, 4.6), (8.0, 4.55)),
        ((12.2, 4.55), (13.0, 4.6)),
        ((15.0, 5.35), (15.8, 6.8)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#64748B"))

    formula_box = FancyBboxPatch(
        (4.2, 1.1),
        9.6,
        1.7,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        linewidth=1.2,
        edgecolor="#CBD5E1",
        facecolor="#EFF6FF",
    )
    ax.add_patch(formula_box)
    ax.text(
        9,
        2.25,
        "核心数学层 / Core mathematical layer",
        ha="center",
        va="center",
        fontsize=12.5,
        color=COLORS["navy"],
        fontweight="bold",
    )
    ax.text(
        9,
        1.65,
        r"$f_l^{(t+1)}=f_l^{(t)}-\Delta tLf_l^{(t)}-\Delta t\lambda f_l^{(t)}+\Delta t s_l$"
        "\n"
        r"$a_{lr,i}=r_i\,\sigma(\beta S_{lr}\operatorname{mm}(f_{l,i})),\quad "
        r"Z_i=0.45\operatorname{mm}(A_i^{full})+0.35H_i+0.20\operatorname{mm}(P_{blastemal,i})$",
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["ink"],
    )
    return savefig(fig, "fig1_method_overview.png")


def fig2_data_landscape() -> Path:
    sc = pd.read_csv(RESULTS / "singlecell" / "scpca_sample_signature_summary.tsv", sep="\t")
    sp = pd.read_csv(RESULTS / "spatial" / "scpca_spatial_library_summary.tsv", sep="\t")
    fetal = pd.read_csv(RESULTS / "fetal_reference" / "hca_fetal_reference_signature_summary.tsv", sep="\t")

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.9, 1.2], hspace=0.35, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.axis("off")
    cards = [
        ("40", "单核/单细胞库\nsnRNA libraries", COLORS["blue"]),
        ("200,514", "细胞/细胞核\ncells/nuclei", COLORS["teal"]),
        ("100", "Visium空间库\nspatial libraries", COLORS["green"]),
        ("270,873", "空间spot\nspatial spots", COLORS["gold"]),
        ("250", "结构约束LR轴\nstructure-scored LR axes", COLORS["coral"]),
        ("136", "TARGET bulk样本\nTARGET bulk samples", COLORS["purple"]),
    ]
    for i, (num, lab, color) in enumerate(cards):
        x = 0.02 + i * 0.16
        box = FancyBboxPatch((x, 0.18), 0.14, 0.64, transform=ax0.transAxes,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             facecolor="#F8FAFC", edgecolor=color, linewidth=1.6)
        ax0.add_patch(box)
        ax0.text(x + 0.07, 0.58, num, transform=ax0.transAxes, ha="center", fontsize=22, fontweight="bold", color=color)
        ax0.text(x + 0.07, 0.34, lab, transform=ax0.transAxes, ha="center", fontsize=10.5, color=COLORS["ink"])
    ax0.text(0.02, 0.95, "A  数据资源与规模 / Data resources", transform=ax0.transAxes,
             fontsize=14, fontweight="bold", color=COLORS["navy"])

    ax1 = fig.add_subplot(gs[1, 0])
    sc_group = sc.groupby("subdiagnosis")[["structpde_wt_tt_cell_score", "blastemal_progenitor", "igf"]].mean()
    sc_group = sc_group.loc[[x for x in ["Anaplastic", "Favorable"] if x in sc_group.index]]
    sc_group.plot(kind="bar", ax=ax1, color=[COLORS["coral"], COLORS["blue"], COLORS["gold"]], width=0.75)
    ax1.set_title("B  单细胞层发育/风险程序 / Single-cell programs", loc="left", fontweight="bold", color=COLORS["navy"])
    ax1.set_ylabel("Mean score")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=0)
    ax1.legend(["StructPDE-WT-TT", "Blastemal", "IGF"], frameon=False, fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 1])
    fetal_top = fetal.sort_values("mean_score", ascending=False).head(7)
    ax2.barh(fetal_top["signature"], fetal_top["mean_score"], color=[COLORS["green"], COLORS["blue"], COLORS["gold"], COLORS["teal"], COLORS["coral"], COLORS["purple"], "#94A3B8"])
    ax2.invert_yaxis()
    ax2.set_title("C  胎儿肾参照程序 / Fetal kidney reference programs", loc="left", fontweight="bold", color=COLORS["navy"])
    ax2.set_xlabel("Mean score")
    ax2.grid(axis="x", linestyle="--", alpha=0.25)

    return savefig(fig, "fig2_data_landscape.png")


def fig3_structure_axes() -> Path:
    lr = pd.read_csv(RESULTS / "structure" / "lr_structure_scores.tsv", sep="\t")
    post = pd.read_csv(RESULTS / "structure" / "structure_evidence_posterior.tsv", sep="\t")
    refined = pd.read_csv(RESULTS / "structure" / "refined_key_axis_summary.tsv", sep="\t")

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    top = lr.assign(axis=lr["ligand"] + "-" + lr["receptor"]).sort_values("struct_expression_score", ascending=False).head(10)
    palette = {
        "CHEMOKINE": COLORS["purple"], "IGF": COLORS["gold"], "NOTCH": COLORS["green"],
        "BMP_TGF": COLORS["coral"], "ECM_INTEGRIN": COLORS["blue"], "FGF": COLORS["teal"], "OTHER": "#94A3B8"
    }
    colors = [palette.get(f, "#94A3B8") for f in top["family"]]
    ax1.barh(top["axis"][::-1], top["struct_expression_score"][::-1], color=colors[::-1])
    ax1.set_title("A  结构约束表达分Top轴 / Top structure-constrained LR axes", loc="left", fontweight="bold", color=COLORS["navy"])
    ax1.set_xlabel("Structure-constrained expression score")
    ax1.grid(axis="x", linestyle="--", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    fam = lr.groupby("family").size().sort_values(ascending=True)
    ax2.barh(fam.index, fam.values, color=[palette.get(f, "#94A3B8") for f in fam.index])
    ax2.set_title("B  LR家族构成 / LR family composition", loc="left", fontweight="bold", color=COLORS["navy"])
    ax2.set_xlabel("Number of axes")
    ax2.grid(axis="x", linestyle="--", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    post_top = post.assign(axis=post["ligand"] + "-" + post["receptor"]).sort_values("posterior_struct_expression_score", ascending=False).head(10)
    ax3.scatter(post_top["posterior_structure_score"], post_top["posterior_struct_expression_score"],
                s=post_top["expression_lr_score"] * 11, c=[palette.get(f, "#94A3B8") for f in post_top["family"]],
                alpha=0.78, edgecolor="white", linewidth=1)
    for _, row in post_top.head(6).iterrows():
        ax3.text(row["posterior_structure_score"] + 0.005, row["posterior_struct_expression_score"], row["axis"], fontsize=8.5)
    ax3.set_title("C  后验结构证据重评分 / Posterior structural evidence", loc="left", fontweight="bold", color=COLORS["navy"])
    ax3.set_xlabel(r"$S_{lr}^{post}$")
    ax3.set_ylabel("Posterior structural-expression score")
    ax3.grid(linestyle="--", alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    refined = refined.sort_values("best_interface_confidence", ascending=True)
    ax4.barh(refined["axis"], refined["best_interface_confidence"], color=COLORS["teal"])
    ax4.axvline(0.70, color=COLORS["coral"], linestyle="--", linewidth=1.4, label="0.70")
    ax4.set_title("D  关键轴精修界面置信度 / Refined interface confidence", loc="left", fontweight="bold", color=COLORS["navy"])
    ax4.set_xlabel("Best interface confidence")
    ax4.set_xlim(0, 1.0)
    ax4.grid(axis="x", linestyle="--", alpha=0.25)
    ax4.legend(frameon=False, loc="lower right")

    return savefig(fig, "fig3_structure_axes.png")


def fig4_spatial_fields() -> Path:
    spots = pd.read_csv(RESULTS / "spatial" / "scpca_spatial_spot_scores.tsv", sep="\t")
    cases = [
        ("SCPCL000448", "Anaplastic reference\nSCPCS000184/SCPCL000448"),
        ("SCPCL000422", "Favorable spatial outlier\nSCPCS000204/SCPCL000422"),
    ]
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.18, wspace=0.12)
    metrics = [
        ("structpde_wt_tt_spatial_score", "StructPDE-WT-TT Z_i", "magma"),
        ("stromal_ecm", "Stromal ECM", "viridis"),
    ]
    for r, (lib, case_title) in enumerate(cases):
        df = spots[spots["library_id"] == lib].copy()
        for c, (metric, title, cmap) in enumerate(metrics):
            ax = fig.add_subplot(gs[r, c])
            sc = ax.scatter(df["array_col"], -df["array_row"], c=df[metric], s=13, cmap=cmap, linewidths=0, alpha=0.92)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(("A" if r == 0 and c == 0 else "B" if r == 0 and c == 1 else "C" if c == 0 else "D")
                         + f"  {case_title} | {title}", loc="left", fontsize=11.5, fontweight="bold", color=COLORS["navy"])
            cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
    return savefig(fig, "fig4_spatial_fields.png")


def _roc_curve_np(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    y = y_true[order]
    pos = y.sum()
    neg = len(y) - pos
    tps = np.r_[0, np.cumsum(y)]
    fps = np.r_[0, np.cumsum(1 - y)]
    tpr = tps / pos if pos else tps
    fpr = fps / neg if neg else fps
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, float(auc)


def fig5_model_performance() -> Path:
    pred = pd.read_csv(RESULTS / "models" / "spatial_library_classifier_predictions.tsv", sep="\t")
    coef = pd.read_csv(RESULTS / "models" / "spatial_logistic_coefficients.tsv", sep="\t")
    ab = pd.read_csv(RESULTS / "tables" / "full_spatial_ablation_metrics.tsv", sep="\t")

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    y = (pred["subdiagnosis"] == "Anaplastic").astype(int).values
    fpr, tpr, auc_val = _roc_curve_np(y, pred["pred_anaplastic_probability"].values)
    ax1.plot(fpr, tpr, color=COLORS["coral"], linewidth=3, label=f"LOO AUC={auc_val:.3f}")
    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--")
    ax1.set_title("A  多特征LOO分类器 / Multi-feature LOO classifier", loc="left", fontweight="bold", color=COLORS["navy"])
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.legend(frameon=False)
    ax1.grid(linestyle="--", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    data = [pred[pred["subdiagnosis"] == g]["pred_anaplastic_probability"] for g in ["Anaplastic", "Favorable"]]
    parts = ax2.violinplot(data, showmedians=True, widths=0.75)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([COLORS["coral"], COLORS["blue"]][i])
        pc.set_edgecolor("white")
        pc.set_alpha(0.75)
    parts["cmedians"].set_color(COLORS["ink"])
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Anaplastic", "Favorable"])
    ax2.set_ylabel("Predicted probability")
    ax2.set_title("B  预测概率分布 / Predicted probability", loc="left", fontweight="bold", color=COLORS["navy"])
    ax2.grid(axis="y", linestyle="--", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    coef_show = pd.concat([coef.head(6), coef.tail(4)]).sort_values("coefficient")
    colors = [COLORS["coral"] if x > 0 else COLORS["blue"] for x in coef_show["coefficient"]]
    ax3.barh(coef_show["feature"], coef_show["coefficient"], color=colors)
    ax3.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax3.set_title("C  关键空间-发育特征 / Spatial-developmental features", loc="left", fontweight="bold", color=COLORS["navy"])
    ax3.set_xlabel("Logistic coefficient")
    ax3.grid(axis="x", linestyle="--", alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    ab_show = ab.copy()
    ab_show["short"] = ab_show["model"].str.replace("A0_full_structpde_wt_tt", "A0 full", regex=False).str.replace("A1_expression_only", "A1 expr", regex=False).str.replace("A8_shuffled_spatial_graph", "A8 shuffled", regex=False).str.replace("A5_expression_pde", "A5 expr+PDE", regex=False).str.replace("A6_expression_structure_pde", "A6 expr+struct+PDE", regex=False).str.replace("A2_expression_structure", "A2 expr+struct", regex=False).str.replace("A10_no_fetal_reference", "A10 no fetal", regex=False)
    ax4.scatter(ab_show["auc_anaplastic_vs_favorable"], ab_show["mean_moran_i"],
                s=150, c=[COLORS["coral"] if "full" in s else COLORS["teal"] for s in ab_show["short"]],
                edgecolor="white", linewidth=1.2)
    for _, row in ab_show.iterrows():
        ax4.text(row["auc_anaplastic_vs_favorable"] + 0.002, row["mean_moran_i"], row["short"], fontsize=8.5)
    ax4.set_title("D  消融：单变量AUC与空间聚集性 / Ablation", loc="left", fontweight="bold", color=COLORS["navy"])
    ax4.set_xlabel("Single-score AUC")
    ax4.set_ylabel("Mean Moran's I")
    ax4.grid(linestyle="--", alpha=0.25)

    return savefig(fig, "fig5_model_performance.png")


def fig6_case_and_validation() -> Path:
    cases = pd.read_csv(RESULTS / "case_studies" / "integrated_paper_case_studies.tsv", sep="\t")
    bulk = pd.read_csv(RESULTS / "tables" / "target_wt_bulk_validation_summary.tsv", sep="\t")
    fetal = pd.read_csv(RESULTS / "fetal_reference" / "hca_fetal_reference_signature_summary.tsv", sep="\t")

    fig = plt.figure(figsize=(16, 10), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    primary = cases[cases["paper_case_tier"] == "primary_case"].copy().head(10)
    primary = primary.sort_values("integrated_case_score")
    colors = [COLORS["coral"] if x == "Anaplastic" else COLORS["blue"] if x == "Favorable" else "#94A3B8" for x in primary["subdiagnosis"]]
    ax1.barh(primary["sample_id"], primary["integrated_case_score"], color=colors)
    ax1.set_title("A  整合case study / Integrated case studies", loc="left", fontweight="bold", color=COLORS["navy"])
    ax1.set_xlabel("Integrated case score")
    ax1.grid(axis="x", linestyle="--", alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    bulk_show = bulk.sort_values("spearman_days_followup", ascending=True).tail(8)
    sig_colors = [COLORS["coral"] if p < 0.05 else "#94A3B8" for p in bulk_show["pvalue"]]
    ax2.barh(bulk_show["score"], bulk_show["spearman_days_followup"], color=sig_colors)
    ax2.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax2.set_title("B  TARGET bulk弱一致性 / TARGET bulk proxy support", loc="left", fontweight="bold", color=COLORS["navy"])
    ax2.set_xlabel("Spearman rho with follow-up days")
    ax2.grid(axis="x", linestyle="--", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    fetal_show = fetal.sort_values("p95_score", ascending=False).head(8)
    ax3.scatter(fetal_show["mean_score"], fetal_show["p95_score"], s=160, c=COLORS["green"], alpha=0.8, edgecolor="white")
    for _, row in fetal_show.iterrows():
        ax3.text(row["mean_score"] + 0.005, row["p95_score"], row["signature"], fontsize=9)
    ax3.set_title("C  胎儿肾参照的发育背景 / Fetal kidney reference context", loc="left", fontweight="bold", color=COLORS["navy"])
    ax3.set_xlabel("Mean score")
    ax3.set_ylabel("p95 score")
    ax3.grid(linestyle="--", alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    bullets = [
        ("空间聚集性", "A0 Moran's I=0.611，高于表达-only与打乱图"),
        ("多特征判别", "15特征LOO分类器AUC=0.836，AP=0.782"),
        ("结构优先级", "CXCL12-CXCR4、FN1-ITGB1、MDK-ITGB1获得精修支持"),
        ("机制提示", "高分区域更多呈ECM/基质、发育和免疫逃逸复合生态位"),
    ]
    ax4.text(0.02, 0.95, "D  结果主张边界 / Main claims with guardrails", transform=ax4.transAxes,
             fontsize=13, fontweight="bold", color=COLORS["navy"], va="top")
    y = 0.78
    for title, body in bullets:
        ax4.add_patch(Circle((0.05, y + 0.01), 0.012, transform=ax4.transAxes, color=COLORS["coral"]))
        ax4.text(0.08, y, f"{title}：", transform=ax4.transAxes, fontsize=11.5, fontweight="bold", color=COLORS["ink"], va="center")
        ax4.text(0.25, y, body, transform=ax4.transAxes, fontsize=11.3, color=COLORS["gray"], va="center")
        y -= 0.16

    return savefig(fig, "fig6_case_and_validation.png")

def main() -> None:
    ensure_dirs()
    generated = [
        fig1_method_overview(),
        fig2_data_landscape(),
        fig3_structure_axes(),
        fig4_spatial_fields(),
        fig5_model_performance(),
        fig6_case_and_validation(),
    ]
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
