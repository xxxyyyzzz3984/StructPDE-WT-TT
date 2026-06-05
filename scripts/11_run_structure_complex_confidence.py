from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from wtai.paths import RESULTS, configure_local_cache


BOLTZ_DEPS = Path(".deps_structure")
BOLTZ_CACHE = Path("models") / "structure_predictors" / "boltz"
BOLTZ_CHECKPOINT = BOLTZ_CACHE / "boltz2_conf.ckpt"


# 1-based inclusive regions. These are intentionally domain-level constructs:
# full-length NOTCH/FN1/COL/LAMB complexes are too large for robust 3090 screening.
DOMAIN_REGIONS: dict[str, tuple[int, int, str]] = {
    "CXCL12": (22, 93, "mature chemokine ligand"),
    "CXCR4": (1, 352, "full-length GPCR receptor"),
    "ACKR3": (1, 362, "full-length GPCR receptor"),
    "IGF1": (49, 118, "mature IGF ligand"),
    "IGF2": (25, 91, "mature IGF ligand"),
    "IGF1R": (31, 932, "extracellular receptor domain"),
    "IGF2R": (1508, 1650, "IGF2-binding extracellular repeat window"),
    "JAG1": (34, 540, "DSL/EGF ligand-binding extracellular module"),
    "DLL4": (27, 525, "DSL/EGF ligand-binding extracellular module"),
    "NOTCH1": (400, 520, "EGF11-13 ligand-binding receptor window"),
    "NOTCH2": (390, 510, "EGF11-13 ligand-binding receptor window"),
    "NOTCH3": (390, 510, "EGF11-13 ligand-binding receptor window"),
    "BMP2": (283, 396, "mature BMP ligand"),
    "BMP4": (293, 408, "mature BMP ligand"),
    "BMP7": (293, 431, "mature BMP ligand"),
    "BMPR1A": (24, 152, "extracellular ligand-binding domain"),
    "BMPR2": (27, 151, "extracellular ligand-binding domain"),
    "TGFB1": (279, 390, "mature TGF-beta ligand"),
    "TGFBR1": (24, 126, "extracellular ligand-binding domain"),
    "FN1": (1410, 1605, "fibronectin III9-10 integrin-binding module"),
    "ITGB1": (21, 470, "integrin beta-1 extracellular headpiece"),
    "ITGA5": (1, 620, "integrin alpha-5 extracellular headpiece"),
    "SPP1": (17, 314, "secreted osteopontin mature region"),
    "LAMB2": (1520, 1798, "laminin beta-2 C-terminal LG-region window"),
    "RPSA": (1, 295, "laminin receptor full-length protein"),
    "MDK": (23, 143, "mature midkine ligand"),
    "COL1A1": (780, 920, "collagen triple-helix receptor-binding window"),
    "COL2A1": (780, 920, "collagen triple-helix receptor-binding window"),
    "COL3A1": (780, 920, "collagen triple-helix receptor-binding window"),
    "DDR1": (22, 417, "DDR1 extracellular discoidin domain"),
    "FGF2": (1, 288, "FGF2 protein isoform window"),
    "FGFR1": (22, 377, "FGFR1 extracellular domain"),
}


def slug(ligand: str, receptor: str) -> str:
    return f"{ligand.upper()}__{receptor.upper()}".replace("/", "_")


def read_top_axes(limit_structure: int, limit_spatial: int) -> pd.DataFrame:
    structure = pd.read_csv(RESULTS / "structure" / "lr_structure_scores.tsv", sep="\t")
    structure_top = structure.head(limit_structure).copy()
    structure_top["selection_source"] = "structure_top"
    structure_top["selection_metric"] = structure_top["struct_expression_score"]

    spatial_path = RESULTS / "spatial" / "scpca_spatial_lr_axis_summary.tsv"
    spatial_top = pd.DataFrame()
    if spatial_path.exists():
        spatial = pd.read_csv(spatial_path, sep="\t")
        spatial_top = (
            spatial.groupby(["ligand", "receptor"], dropna=False)
            .agg(
                spatial_n_libraries=("library_id", "nunique"),
                spatial_mean_full_activation=("mean_full_activation", "mean"),
                spatial_mean_expression_axis=("mean_expression_axis", "mean"),
                structure_score=("structure_score", "mean"),
            )
            .reset_index()
            .sort_values("spatial_mean_full_activation", ascending=False)
            .head(limit_spatial)
        )
        spatial_top["selection_source"] = "spatial_activation_top"
        spatial_top["selection_metric"] = spatial_top["spatial_mean_full_activation"]
        spatial_top = spatial_top.merge(
            structure,
            on=["ligand", "receptor", "structure_score"],
            how="left",
            suffixes=("", "_structure"),
        )

    combined = pd.concat([structure_top, spatial_top], ignore_index=True, sort=False)
    combined["ligand"] = combined["ligand"].astype(str).str.upper()
    combined["receptor"] = combined["receptor"].astype(str).str.upper()
    combined["axis"] = combined["ligand"] + "-" + combined["receptor"]
    combined = combined.drop_duplicates(["ligand", "receptor"], keep="first").reset_index(drop=True)
    combined.insert(0, "complex_rank", np.arange(1, len(combined) + 1))
    return combined


def fetch_uniprot_sequence(gene: str, cache_dir: Path) -> dict[str, str | int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    gene = gene.upper()
    cache_file = cache_dir / f"{gene}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    params = {
        "query": f"(gene_exact:{gene}) AND (organism_id:9606) AND (reviewed:true)",
        "format": "json",
        "fields": "accession,gene_primary,protein_name,length,sequence",
        "size": 5,
    }
    response = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    if not results:
        raise RuntimeError(f"No reviewed human UniProt record found for {gene}")

    chosen = results[0]
    sequence = chosen["sequence"]["value"]
    record = {
        "gene": gene,
        "accession": chosen.get("primaryAccession", ""),
        "protein_name": chosen.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", ""),
        "length": len(sequence),
        "sequence": sequence,
    }
    cache_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def domain_sequence(record: dict[str, str | int], max_default_len: int = 700) -> dict[str, str | int]:
    gene = str(record["gene"]).upper()
    seq = str(record["sequence"])
    length = len(seq)
    if gene in DOMAIN_REGIONS:
        start, end, label = DOMAIN_REGIONS[gene]
        start = max(1, min(start, length))
        end = max(start, min(end, length))
    elif length <= max_default_len:
        start, end, label = 1, length, "full-length sequence"
    else:
        start, end, label = 1, max_default_len, f"N-terminal {max_default_len}-aa window"
    region_seq = seq[start - 1 : end]
    return {
        "gene": gene,
        "accession": record["accession"],
        "protein_name": record["protein_name"],
        "full_length": length,
        "region_start": start,
        "region_end": end,
        "region_label": label,
        "region_length": len(region_seq),
        "sequence": region_seq,
    }


def write_yaml(path: Path, ligand: dict[str, str | int], receptor: dict[str, str | int]) -> None:
    path.write_text(
        "\n".join(
            [
                "version: 1",
                "sequences:",
                "- protein:",
                "    id: A",
                f"    sequence: {ligand['sequence']}",
                "    msa: empty",
                "- protein:",
                "    id: B",
                f"    sequence: {receptor['sequence']}",
                "    msa: empty",
                "",
            ]
        ),
        encoding="utf-8",
    )


def prepare_inputs(axes: pd.DataFrame, input_dir: Path, cache_dir: Path) -> pd.DataFrame:
    input_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for row in axes.itertuples(index=False):
        ligand_record = domain_sequence(fetch_uniprot_sequence(row.ligand, cache_dir))
        receptor_record = domain_sequence(fetch_uniprot_sequence(row.receptor, cache_dir))
        input_name = slug(row.ligand, row.receptor)
        yaml_path = input_dir / f"{input_name}.yaml"
        write_yaml(yaml_path, ligand_record, receptor_record)
        rows.append(
            {
                "complex_rank": int(row.complex_rank),
                "selection_source": row.selection_source,
                "selection_metric": float(row.selection_metric),
                "ligand": row.ligand,
                "receptor": row.receptor,
                "family": getattr(row, "family", ""),
                "input_name": input_name,
                "yaml_path": str(yaml_path),
                "ligand_uniprot": ligand_record["accession"],
                "ligand_region": f"{ligand_record['region_start']}-{ligand_record['region_end']}",
                "ligand_region_label": ligand_record["region_label"],
                "ligand_full_length": ligand_record["full_length"],
                "ligand_region_length": ligand_record["region_length"],
                "receptor_uniprot": receptor_record["accession"],
                "receptor_region": f"{receptor_record['region_start']}-{receptor_record['region_end']}",
                "receptor_region_label": receptor_record["region_label"],
                "receptor_full_length": receptor_record["full_length"],
                "receptor_region_length": receptor_record["region_length"],
                "structure_score_prior": getattr(row, "structure_score", np.nan),
                "struct_expression_score": getattr(row, "struct_expression_score", np.nan),
                "spatial_mean_full_activation": getattr(row, "spatial_mean_full_activation", np.nan),
            }
        )
    return pd.DataFrame(rows)


def boltz_env() -> dict[str, str]:
    env = os.environ.copy()
    deps = str((Path.cwd() / BOLTZ_DEPS).resolve())
    previous_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = deps if not previous_pythonpath else f"{deps}:{previous_pythonpath}"
    env["PATH"] = f"{deps}/bin:" + env.get("PATH", "")
    env["BOLTZ_CACHE"] = str((Path.cwd() / BOLTZ_CACHE).resolve())
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def confidence_json_path(out_dir: Path, input_name: str) -> Path:
    return out_dir / f"boltz_results_{input_name}" / "predictions" / input_name / f"confidence_{input_name}_model_0.json"


def run_boltz_predictions(
    manifest: pd.DataFrame,
    out_dir: Path,
    logs_dir: Path,
    sampling_steps: int,
    recycling_steps: int,
    override: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not BOLTZ_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing Boltz checkpoint: {BOLTZ_CHECKPOINT}")

    for row in manifest.itertuples(index=False):
        output_json = confidence_json_path(out_dir, row.input_name)
        if output_json.exists() and not override:
            continue
        cmd = [
            "boltz",
            "predict",
            row.yaml_path,
            "--out_dir",
            str(out_dir),
            "--cache",
            str(BOLTZ_CACHE),
            "--checkpoint",
            str(BOLTZ_CHECKPOINT),
            "--model",
            "boltz2",
            "--devices",
            "1",
            "--accelerator",
            "gpu",
            "--recycling_steps",
            str(recycling_steps),
            "--sampling_steps",
            str(sampling_steps),
            "--diffusion_samples",
            "1",
            "--max_parallel_samples",
            "1",
            "--num_workers",
            "0",
            "--preprocessing-threads",
            "1",
            "--no_kernels",
            "--output_format",
            "mmcif",
        ]
        if override:
            cmd.append("--override")
        log_path = logs_dir / f"{row.input_name}.log"
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=Path.cwd(), env=boltz_env(), stdout=log, stderr=subprocess.STDOUT)
            log.write(f"\nexit_code={proc.returncode}\nelapsed_seconds={time.time() - start:.1f}\n")
        if proc.returncode != 0:
            print(f"Boltz failed for {row.input_name}; see {log_path}")


def read_confidence(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def offdiag_pair_iptm(conf: dict) -> float:
    pairs = conf.get("pair_chains_iptm", {}) or {}
    vals = []
    for i, inner in pairs.items():
        for j, value in (inner or {}).items():
            if str(i) != str(j):
                try:
                    vals.append(float(value))
                except (TypeError, ValueError):
                    pass
    return float(np.mean(vals)) if vals else float("nan")


def interpretation(row: pd.Series) -> str:
    confidence = pd.to_numeric(row.get("confidence_score"), errors="coerce")
    iptm = pd.to_numeric(row.get("protein_iptm"), errors="coerce")
    iplddt = pd.to_numeric(row.get("complex_iplddt"), errors="coerce")
    pair_iptm = pd.to_numeric(row.get("pair_chains_iptm_offdiag_mean"), errors="coerce")
    if pd.notna(iptm) and pd.notna(iplddt) and pd.notna(confidence):
        if iptm >= 0.75 and iplddt >= 0.70:
            return "high_confidence_complex"
        if (iptm >= 0.45 or pair_iptm >= 0.45) and iplddt >= 0.50:
            return "moderate_interface_support"
        if confidence >= 0.65 and iplddt >= 0.70:
            return "fold_confident_interface_uncertain"
        if confidence >= 0.55 and iplddt >= 0.55:
            return "screening_level_support"
        return "low_confidence_or_uncertain_interface"
    return "prediction_failed_or_missing"


def summarize_outputs(manifest: pd.DataFrame, out_dir: Path, report_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        conf_path = confidence_json_path(out_dir, row.input_name)
        conf = read_confidence(conf_path)
        result = row._asdict()
        result.update(
            {
                "model": "boltz2",
                "checkpoint": str(BOLTZ_CHECKPOINT),
                "msa_mode": "single_sequence_empty_msa",
                "sampling_steps": args.sampling_steps,
                "recycling_steps": args.recycling_steps,
                "confidence_json": str(conf_path) if conf_path.exists() else "",
                "confidence_score": conf.get("confidence_score", np.nan),
                "ptm": conf.get("ptm", np.nan),
                "iptm": conf.get("iptm", np.nan),
                "protein_iptm": conf.get("protein_iptm", np.nan),
                "complex_plddt": conf.get("complex_plddt", np.nan),
                "complex_iplddt": conf.get("complex_iplddt", np.nan),
                "complex_pde": conf.get("complex_pde", np.nan),
                "complex_ipde": conf.get("complex_ipde", np.nan),
                "pair_chains_iptm_offdiag_mean": offdiag_pair_iptm(conf),
            }
        )
        rows.append(result)
    df = pd.DataFrame(rows)
    df["confidence_interpretation"] = df.apply(interpretation, axis=1)
    df = df.sort_values(["confidence_score", "protein_iptm"], ascending=[False, False], na_position="last")
    out_tsv = report_dir / "structure_complex_confidence.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)

    summary = {
        "experiment": "11_structure_complex_confidence",
        "model": "Boltz-2",
        "checkpoint": str(BOLTZ_CHECKPOINT),
        "n_complexes_selected": int(len(manifest)),
        "n_complexes_completed": int(df["confidence_json"].astype(str).ne("").sum()),
        "n_high_confidence": int(df["confidence_interpretation"].eq("high_confidence_complex").sum()),
        "n_moderate_interface_support": int(df["confidence_interpretation"].eq("moderate_interface_support").sum()),
        "n_fold_confident_interface_uncertain": int(
            df["confidence_interpretation"].eq("fold_confident_interface_uncertain").sum()
        ),
        "n_screening_level_support": int(df["confidence_interpretation"].eq("screening_level_support").sum()),
        "n_low_or_uncertain": int(df["confidence_interpretation"].eq("low_confidence_or_uncertain_interface").sum()),
        "mean_confidence_score": float(pd.to_numeric(df["confidence_score"], errors="coerce").mean()),
        "mean_protein_iptm": float(pd.to_numeric(df["protein_iptm"], errors="coerce").mean()),
        "mean_complex_iplddt": float(pd.to_numeric(df["complex_iplddt"], errors="coerce").mean()),
        "top_complexes": df[
            [
                "ligand",
                "receptor",
                "confidence_score",
                "protein_iptm",
                "complex_iplddt",
                "confidence_interpretation",
            ]
        ]
        .head(10)
        .to_dict(orient="records"),
    }
    (report_dir / "structure_complex_confidence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown_report(df, summary, report_dir / "structure_complex_confidence_report.md")
    return df


def write_markdown_report(df: pd.DataFrame, summary: dict, path: Path) -> None:
    lines = [
        "# Boltz-2 top LR complex confidence analysis",
        "",
        "This report summarizes domain-level protein-protein complex confidence predictions for the top StructPDE-WT-TT LR axes.",
        "",
        "## Summary",
        "",
        f"- Selected complexes: {summary['n_complexes_selected']}",
        f"- Completed complexes: {summary['n_complexes_completed']}",
        f"- High-confidence complexes: {summary['n_high_confidence']}",
        f"- Moderate interface support: {summary['n_moderate_interface_support']}",
        f"- Fold-confident but interface-uncertain complexes: {summary['n_fold_confident_interface_uncertain']}",
        f"- Screening-level support: {summary['n_screening_level_support']}",
        f"- Low/uncertain interfaces: {summary['n_low_or_uncertain']}",
        f"- Mean confidence_score: {summary['mean_confidence_score']:.4f}",
        f"- Mean protein_iptm: {summary['mean_protein_iptm']:.4f}",
        f"- Mean complex_iplddt: {summary['mean_complex_iplddt']:.4f}",
        "",
        "## Top complexes",
        "",
        "| Axis | Source | Ligand region | Receptor region | confidence_score | protein_iptm | complex_iplddt | Interpretation |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    top = df.head(20)
    for row in top.itertuples(index=False):
        lines.append(
            "| "
            f"{row.ligand}-{row.receptor} | {row.selection_source} | "
            f"{row.ligand_uniprot}:{row.ligand_region} ({row.ligand_region_label}) | "
            f"{row.receptor_uniprot}:{row.receptor_region} ({row.receptor_region_label}) | "
            f"{float(row.confidence_score):.4f} | {float(row.protein_iptm):.4f} | "
            f"{float(row.complex_iplddt):.4f} | {row.confidence_interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Predictions use `msa: empty`, i.e. single-sequence mode, to avoid external MSA services and keep the run self-contained.",
            "- Long receptors and ECM ligands are represented by biologically motivated extracellular or ligand-binding windows.",
            "- Confidence interpretation is heuristic: high if protein_iptm >= 0.75 and complex_iplddt >= 0.70; moderate interface support if protein_iptm or pair-chain ipTM >= 0.45 and complex_iplddt >= 0.50; fold-confident/interface-uncertain if confidence_score >= 0.65 and complex_iplddt >= 0.70 but interface ipTM remains low.",
            "- The resulting confidence values are intended as structural support for prioritizing LR axes, not as experimental binding constants.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def update_summary_json(report_dir: Path) -> None:
    summary_path = RESULTS / "full_experiment_summary.json"
    boltz_summary = json.loads((report_dir / "structure_complex_confidence_summary.json").read_text(encoding="utf-8"))
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload["structure_complex_confidence"] = boltz_summary
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-structure", type=int, default=12)
    parser.add_argument("--limit-spatial", type=int, default=12)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--recycling-steps", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    configure_local_cache()
    report_dir = RESULTS / "structure"
    input_dir = report_dir / "boltz_inputs"
    cache_dir = report_dir / "uniprot_cache"
    out_dir = report_dir / "boltz_complex_predictions"
    logs_dir = report_dir / "boltz_logs"

    axes = read_top_axes(args.limit_structure, args.limit_spatial)
    manifest = prepare_inputs(axes, input_dir, cache_dir)
    manifest.to_csv(report_dir / "structure_complex_confidence_manifest.tsv", sep="\t", index=False)

    if not args.prepare_only and not args.summarize_only:
        run_boltz_predictions(manifest, out_dir, logs_dir, args.sampling_steps, args.recycling_steps, args.override)
    if not args.prepare_only:
        summarize_outputs(manifest, out_dir, report_dir, args)
        update_summary_json(report_dir)
    print(report_dir / "structure_complex_confidence.tsv")


if __name__ == "__main__":
    main()
