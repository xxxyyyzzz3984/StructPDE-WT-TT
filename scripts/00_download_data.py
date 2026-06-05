from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from wtai.data.download import (
    DownloadRecord,
    Downloader,
    create_scpca_token,
    gdc_clinical,
    gdc_files,
    geo_urls,
    scpca_file_download_url,
    scpca_project_metadata,
    write_scpca_manifest,
)
from wtai.paths import EXTERNAL, RAW, ROOT, configure_local_cache


def proxies_from_args(args: argparse.Namespace) -> dict[str, str]:
    if not args.proxy:
        return {}
    return {"http": args.proxy, "https": args.proxy}


def download_scpca(args: argparse.Namespace, dl: Downloader) -> None:
    project_dir = RAW / "scpca" / args.scpca_project
    project_dir.mkdir(parents=True, exist_ok=True)
    project = scpca_project_metadata(dl.session, args.scpca_project)
    manifest = write_scpca_manifest(project, project_dir / "computed_files.tsv")
    (project_dir / "project_metadata.json").write_text(
        json.dumps(project, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    token = args.scpca_token or os.environ.get("SCPCA_TOKEN")
    if not token and args.scpca_email and args.accept_scpca_terms:
        token = create_scpca_token(dl, args.scpca_email)
        (ROOT / ".cache" / "scpca_token.txt").write_text(token, encoding="utf-8")
    if not token:
        dl.record(
            DownloadRecord(
                "ScPCA protected downloads",
                f"https://scpca.alexslemonade.org/projects/{args.scpca_project}",
                str(project_dir),
                "skipped",
                note="Set SCPCA_TOKEN or pass --scpca-email with --accept-scpca-terms to download protected files.",
            )
        )
        return
    selected = manifest[manifest["s3_key"].isin(args.scpca_files)]
    if selected.empty:
        selected = manifest[(manifest["scope"] == "project") & (manifest["modality"].isin(["SINGLE_CELL", "SPATIAL"]))]
    for _, row in selected.iterrows():
        url = scpca_file_download_url(dl, int(row["id"]), token)
        dl.download(str(row["s3_key"]), url, project_dir / str(row["s3_key"]), headers={})


def download_target(dl: Downloader, include_expression: bool) -> None:
    target_dir = RAW / "target_wt"
    target_dir.mkdir(parents=True, exist_ok=True)
    clinical = gdc_clinical("TARGET-WT")
    clinical.to_csv(target_dir / "clinical.tsv", sep="\t", index=False)
    files = gdc_files("TARGET-WT")
    files.to_csv(target_dir / "gdc_open_expression_files.tsv", sep="\t", index=False)
    if include_expression:
        for _, row in files.iterrows():
            url = f"https://api.gdc.cancer.gov/data/{row['file_id']}"
            dl.download(row["file_name"], url, target_dir / "expression" / row["file_name"])


def download_geo(dl: Downloader, accessions: list[str]) -> None:
    for acc in accessions:
        out = RAW / "geo" / acc
        out.mkdir(parents=True, exist_ok=True)
        for name, url in geo_urls(acc):
            dl.download(name, url, out / name)


def download_lr_resources(dl: Downloader) -> None:
    out = EXTERNAL / "ligand_receptor"
    out.mkdir(parents=True, exist_ok=True)
    omnipath = (
        "https://omnipathdb.org/interactions?"
        "datasets=ligrecextra&organisms=9606&genesymbols=1&fields=sources,references,curation_effort,type"
    )
    dl.download("omnipath_ligrecextra.tsv", omnipath, out / "omnipath_ligrecextra.tsv")


def download_fetal_kidney_reference(dl: Downloader) -> None:
    project_id = "d8ae869c-39c2-4cdd-b3fc-2d0d8f60e7b8"
    out = RAW / "fetal_kidney_hca" / project_id
    out.mkdir(parents=True, exist_ok=True)
    filters = {"projectId": {"is": [project_id]}}
    files = dl.get_json(
        "https://service.azul.data.humancellatlas.org/index/files",
        params={"filters": __import__("json").dumps(filters), "size": "25"},
    )
    rows = []
    for hit in files.get("hits", []):
        for file_info in hit.get("files", []):
            rows.append(file_info)
    pd.DataFrame(rows).to_csv(out / "hca_files.tsv", sep="\t", index=False)
    wanted = {
        "tableOfCounts.mtx",
        "tableOfCounts_colLabels.tsv",
        "tableOfCounts_rowLabels.tsv",
        "Haniffa-Human-10x3pv2_metadata_04-07-2023.xlsx",
    }
    for file_info in rows:
        name = file_info.get("name")
        if name in wanted:
            dl.download(name, file_info["azul_url"], out / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or "")
    parser.add_argument("--scpca-project", default="SCPCP000006")
    parser.add_argument("--scpca-token", default="")
    parser.add_argument("--scpca-email", default=os.environ.get("SCPCA_EMAIL") or "")
    parser.add_argument("--accept-scpca-terms", action="store_true")
    parser.add_argument(
        "--scpca-files",
        nargs="*",
        default=[
            "SCPCP000006_ALL_METADATA.zip",
            "SCPCP000006_SINGLE-CELL_ANN-DATA.zip",
            "SCPCP000006_SPATIAL_SINGLE-CELL-EXPERIMENT.zip",
        ],
    )
    parser.add_argument("--download-target-expression", action="store_true")
    parser.add_argument("--geo", nargs="*", default=["GSE31403", "GSE10320"])
    parser.add_argument("--skip-scpca", action="store_true")
    parser.add_argument("--skip-fetal-kidney", action="store_true")
    args = parser.parse_args()

    configure_local_cache()
    dl = Downloader(ROOT / "data" / "download_manifest.json", proxies_from_args(args))
    if not args.skip_scpca:
        download_scpca(args, dl)
    download_target(dl, include_expression=args.download_target_expression)
    download_geo(dl, args.geo)
    if not args.skip_fetal_kidney:
        download_fetal_kidney_reference(dl)
    download_lr_resources(dl)
    print(f"Manifest: {ROOT / 'data' / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
