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
SEEDS = [101, 202]


def fetch_uniprot_sequence(gene: str, cache_dir: Path) -> dict:
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
    results = response.json().get("results", [])
    if not results:
        raise RuntimeError(f"No reviewed human UniProt record found for {gene}")
    chosen = results[0]
    record = {
        "gene": gene,
        "accession": chosen.get("primaryAccession", ""),
        "protein_name": chosen.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", ""),
        "length": len(chosen["sequence"]["value"]),
        "sequence": chosen["sequence"]["value"],
    }
    cache_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def crop(record: dict, start: int, end: int, label: str) -> dict:
    seq = record["sequence"]
    start = max(1, min(start, len(seq)))
    end = max(start, min(end, len(seq)))
    return {
        "gene": record["gene"],
        "accession": record["accession"],
        "full_length": len(seq),
        "region_start": start,
        "region_end": end,
        "region_label": label,
        "sequence": seq[start - 1 : end],
    }


def variant_specs() -> list[dict]:
    return [
        {
            "axis": "CXCL12-CXCR4",
            "variant": "pocket_multiseed",
            "seeds": [101, 202, 303],
            "chains": [
                ("A", "CXCL12", 22, 93, "mature chemokine ligand"),
                ("B", "CXCR4", 1, 352, "full-length GPCR receptor"),
            ],
            "constraints": [
                {"pocket": {"binder": "A", "contacts": [["B", 1], ["B", 8], ["B", 12], ["B", 187]], "max_distance": 12.0}}
            ],
        },
        {
            "axis": "JAG1-NOTCH1",
            "variant": "compact_pocket_multiseed",
            "seeds": SEEDS,
            "chains": [
                ("A", "JAG1", 180, 360, "compact DSL/EGF ligand-binding window"),
                ("B", "NOTCH1", 400, 520, "EGF11-13 receptor window"),
            ],
            "constraints": [
                {"pocket": {"binder": "A", "contacts": [["B", 40], ["B", 60], ["B", 80]], "max_distance": 12.0}}
            ],
        },
        {
            "axis": "JAG1-NOTCH2",
            "variant": "compact_pocket_multiseed",
            "seeds": SEEDS,
            "chains": [
                ("A", "JAG1", 180, 360, "compact DSL/EGF ligand-binding window"),
                ("B", "NOTCH2", 390, 510, "EGF11-13 receptor window"),
            ],
            "constraints": [
                {"pocket": {"binder": "A", "contacts": [["B", 40], ["B", 60], ["B", 80]], "max_distance": 12.0}}
            ],
        },
        {
            "axis": "FN1-ITGB1",
            "variant": "alpha5_beta1_trimer",
            "seeds": SEEDS,
            "chains": [
                ("A", "FN1", 1410, 1605, "fibronectin III9-10 integrin-binding module"),
                ("B", "ITGA5", 1, 560, "integrin alpha-5 extracellular headpiece"),
                ("C", "ITGB1", 21, 470, "integrin beta-1 extracellular headpiece"),
            ],
            "constraints": [],
        },
        {
            "axis": "MDK-ITGB1",
            "variant": "alpha5_beta1_trimer",
            "seeds": SEEDS,
            "chains": [
                ("A", "MDK", 23, 143, "mature midkine ligand"),
                ("B", "ITGA5", 1, 560, "integrin alpha-5 extracellular headpiece"),
                ("C", "ITGB1", 21, 470, "integrin beta-1 extracellular headpiece"),
            ],
            "constraints": [],
        },
        {
            "axis": "IGF2-IGF1R",
            "variant": "ecd_multiseed",
            "seeds": SEEDS,
            "chains": [
                ("A", "IGF2", 25, 91, "mature IGF2 ligand"),
                ("B", "IGF1R", 31, 932, "IGF1R extracellular domain"),
            ],
            "constraints": [],
        },
    ]


def slug(text: str) -> str:
    return text.replace("-", "__").replace("/", "_").replace(" ", "_")


def write_yaml(path: Path, chains: list[dict], constraints: list[dict]) -> None:
    lines = ["version: 1", "sequences:"]
    for chain in chains:
        lines.extend(
            [
                "- protein:",
                f"    id: {chain['chain_id']}",
                f"    sequence: {chain['sequence']}",
                "    msa: empty",
            ]
        )
    if constraints:
        lines.append("constraints:")
        for item in constraints:
            pocket = item.get("pocket")
            if pocket:
                lines.extend(
                    [
                        "- pocket:",
                        f"    binder: {pocket['binder']}",
                        "    contacts:",
                    ]
                )
                for contact in pocket["contacts"]:
                    lines.append(f"    - [{contact[0]}, {contact[1]}]")
                lines.append(f"    max_distance: {pocket.get('max_distance', 12.0)}")
                lines.append(f"    force: {str(pocket.get('force', False)).lower()}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_inputs(input_dir: Path, cache_dir: Path) -> pd.DataFrame:
    input_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for spec in variant_specs():
        for seed in spec["seeds"]:
            chains = []
            for chain_id, gene, start, end, label in spec["chains"]:
                record = fetch_uniprot_sequence(gene, cache_dir)
                region = crop(record, start, end, label)
                chains.append({"chain_id": chain_id, **region})
            input_name = f"{slug(spec['axis'])}__{spec['variant']}__seed{seed}"
            yaml_path = input_dir / f"{input_name}.yaml"
            write_yaml(yaml_path, chains, spec.get("constraints", []))
            row = {
                "axis": spec["axis"],
                "variant": spec["variant"],
                "seed": seed,
                "input_name": input_name,
                "yaml_path": str(yaml_path),
                "n_chains": len(chains),
                "total_residues": int(sum(len(c["sequence"]) for c in chains)),
                "constraints": json.dumps(spec.get("constraints", []), ensure_ascii=False),
            }
            for chain in chains:
                prefix = f"chain_{chain['chain_id']}"
                row[f"{prefix}_gene"] = chain["gene"]
                row[f"{prefix}_uniprot"] = chain["accession"]
                row[f"{prefix}_region"] = f"{chain['region_start']}-{chain['region_end']}"
                row[f"{prefix}_region_label"] = chain["region_label"]
                row[f"{prefix}_length"] = len(chain["sequence"])
            rows.append(row)
    return pd.DataFrame(rows)


def boltz_env() -> dict[str, str]:
    env = os.environ.copy()
    deps = str((Path.cwd() / BOLTZ_DEPS).resolve())
    env["PYTHONPATH"] = deps if not env.get("PYTHONPATH") else f"{deps}:{env['PYTHONPATH']}"
    env["PATH"] = f"{deps}/bin:" + env.get("PATH", "")
    env["BOLTZ_CACHE"] = str((Path.cwd() / BOLTZ_CACHE).resolve())
    env["WANDB_MODE"] = "offline"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def confidence_json_path(out_dir: Path, input_name: str) -> Path:
    return out_dir / f"boltz_results_{input_name}" / "predictions" / input_name / f"confidence_{input_name}_model_0.json"


def run_predictions(manifest: pd.DataFrame, out_dir: Path, logs_dir: Path, override: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest.itertuples(index=False):
        conf = confidence_json_path(out_dir, row.input_name)
        if conf.exists() and not override:
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
            "3",
            "--sampling_steps",
            "80",
            "--diffusion_samples",
            "1",
            "--max_parallel_samples",
            "1",
            "--num_workers",
            "0",
            "--preprocessing-threads",
            "1",
            "--no_kernels",
            "--use_potentials",
            "--seed",
            str(row.seed),
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
            print(f"Boltz refined run failed for {row.input_name}; see {log_path}")


def read_confidence(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ligand_pair_iptm(conf: dict) -> float:
    pairs = conf.get("pair_chains_iptm", {}) or {}
    vals = []
    for src, inner in pairs.items():
        for dst, value in (inner or {}).items():
            if str(src) == "0" and str(dst) != "0":
                vals.append(float(value))
            elif str(dst) == "0" and str(src) != "0":
                vals.append(float(value))
    return float(np.mean(vals)) if vals else float("nan")


def summarize(manifest: pd.DataFrame, out_dir: Path, report_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        conf_path = confidence_json_path(out_dir, row.input_name)
        conf = read_confidence(conf_path)
        rec = row._asdict()
        rec.update(
            {
                "model": "boltz2_refined",
                "confidence_json": str(conf_path) if conf_path.exists() else "",
                "confidence_score": conf.get("confidence_score", np.nan),
                "ptm": conf.get("ptm", np.nan),
                "iptm": conf.get("iptm", np.nan),
                "protein_iptm": conf.get("protein_iptm", np.nan),
                "complex_plddt": conf.get("complex_plddt", np.nan),
                "complex_iplddt": conf.get("complex_iplddt", np.nan),
                "complex_pde": conf.get("complex_pde", np.nan),
                "complex_ipde": conf.get("complex_ipde", np.nan),
                "ligand_pair_iptm_mean": ligand_pair_iptm(conf),
            }
        )
        protein_iptm = pd.to_numeric(pd.Series([rec.get("protein_iptm", np.nan)]), errors="coerce").fillna(0.0).iloc[0]
        ligand_pair_iptm_value = pd.to_numeric(
            pd.Series([rec.get("ligand_pair_iptm_mean", np.nan)]), errors="coerce"
        ).fillna(0.0).iloc[0]
        rec["interface_confidence"] = max(float(protein_iptm), float(ligand_pair_iptm_value))
        rows.append(rec)
    df = pd.DataFrame(rows)
    baseline = pd.read_csv(RESULTS / "structure" / "structure_complex_confidence.tsv", sep="\t")
    baseline["axis"] = baseline["ligand"] + "-" + baseline["receptor"]
    baseline["baseline_interface_confidence"] = baseline[
        ["protein_iptm", "pair_chains_iptm_offdiag_mean"]
    ].max(axis=1)
    baseline_map = baseline.drop_duplicates("axis").set_index("axis")[
        ["confidence_score", "baseline_interface_confidence", "complex_iplddt"]
    ].rename(
        columns={
            "confidence_score": "baseline_confidence_score",
            "complex_iplddt": "baseline_complex_iplddt",
        }
    )
    df = df.merge(baseline_map, on="axis", how="left")
    df["interface_delta_vs_baseline"] = df["interface_confidence"] - df["baseline_interface_confidence"]
    df["confidence_delta_vs_baseline"] = df["confidence_score"] - df["baseline_confidence_score"]
    df["refined_label"] = np.select(
        [
            (df["interface_confidence"] >= 0.55) & (df["complex_iplddt"] >= 0.55),
            (df["interface_delta_vs_baseline"] >= 0.08) & (df["complex_iplddt"] >= 0.60),
            (df["confidence_score"] >= 0.65) & (df["complex_iplddt"] >= 0.70),
        ],
        ["refined_moderate_interface", "refined_interface_improved", "refined_fold_confident"],
        default="refined_uncertain",
    )
    df.to_csv(report_dir / "refined_key_axis_boltz2.tsv", sep="\t", index=False)

    summary = (
        df.groupby("axis", dropna=False)
        .agg(
            n_runs=("input_name", "count"),
            n_completed=("confidence_json", lambda s: int(s.astype(str).ne("").sum())),
            best_confidence_score=("confidence_score", "max"),
            mean_confidence_score=("confidence_score", "mean"),
            best_interface_confidence=("interface_confidence", "max"),
            mean_interface_confidence=("interface_confidence", "mean"),
            best_complex_iplddt=("complex_iplddt", "max"),
            mean_interface_delta_vs_baseline=("interface_delta_vs_baseline", "mean"),
            best_interface_delta_vs_baseline=("interface_delta_vs_baseline", "max"),
            best_refined_label=("refined_label", lambda s: s.value_counts().index[0] if len(s) else ""),
        )
        .reset_index()
        .sort_values(["best_interface_confidence", "best_confidence_score"], ascending=False)
    )
    summary.to_csv(report_dir / "refined_key_axis_summary.tsv", sep="\t", index=False)
    write_report(df, summary, report_dir / "refined_key_axis_report.md")
    update_summary_json(summary)
    return df, summary


def write_report(df: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Refined Boltz-2 key-axis predictions",
        "",
        "Refinement used multi-seed Boltz-2 runs with potentials, compact domain windows, pocket conditioning for CXCL12/JAG1 axes, and alpha5-beta1 trimer variants for integrin axes.",
        "",
        "## Axis-level summary",
        "",
        "| Axis | Runs | Best confidence | Best interface | Best complex ipLDDT | Best delta vs baseline | Label |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.axis} | {row.n_runs} | {row.best_confidence_score:.4f} | "
            f"{row.best_interface_confidence:.4f} | {row.best_complex_iplddt:.4f} | "
            f"{row.best_interface_delta_vs_baseline:.4f} | {row.best_refined_label} |"
        )
    lines.extend(["", "## Best individual runs", "", "| Axis | Variant | Seed | confidence | interface | ipLDDT | delta | Label |", "|---|---|---:|---:|---:|---:|---:|---|"])
    best = df.sort_values(["axis", "interface_confidence"], ascending=[True, False]).drop_duplicates("axis")
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.axis} | {row.variant} | {row.seed} | {row.confidence_score:.4f} | "
            f"{row.interface_confidence:.4f} | {row.complex_iplddt:.4f} | "
            f"{row.interface_delta_vs_baseline:.4f} | {row.refined_label} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Refined runs are used as robustness and sensitivity checks, not as experimental binding proof.",
            "- Positive deltas indicate that topology-aware variants or pocket conditioning improved the predicted interface confidence relative to the first-pass domain-level screen.",
            "- Persistent low interface confidence should down-weight the axis in posterior structural evidence even if expression or spatial activation is high.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def update_summary_json(summary: pd.DataFrame) -> None:
    summary_path = RESULTS / "full_experiment_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload["refined_key_axis_predictions"] = {
        "experiment": "13_refined_key_axis_predictions",
        "model": "Boltz-2 refined multi-seed/domain-topology variants",
        "n_axes": int(summary["axis"].nunique()),
        "n_runs": int(summary["n_runs"].sum()),
        "n_completed": int(summary["n_completed"].sum()),
        "top_axes": summary.head(10).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    configure_local_cache()
    report_dir = RESULTS / "structure"
    input_dir = report_dir / "refined_boltz_inputs"
    cache_dir = report_dir / "uniprot_cache"
    out_dir = report_dir / "refined_boltz_predictions"
    logs_dir = report_dir / "refined_boltz_logs"
    manifest = prepare_inputs(input_dir, cache_dir)
    manifest.to_csv(report_dir / "refined_key_axis_manifest.tsv", sep="\t", index=False)
    if not args.prepare_only and not args.summarize_only:
        run_predictions(manifest, out_dir, logs_dir, args.override)
    if not args.prepare_only:
        summarize(manifest, out_dir, report_dir)
    print(report_dir / "refined_key_axis_summary.tsv")


if __name__ == "__main__":
    main()
