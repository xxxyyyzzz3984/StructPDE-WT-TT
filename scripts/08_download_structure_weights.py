from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from wtai.paths import MODELS, ROOT, configure_local_cache


@dataclass
class WeightItem:
    family: str
    name: str
    url: str
    relative_path: str
    expected_size: int | None = None
    extract: bool = False


WEIGHTS = [
    WeightItem(
        "boltz",
        "boltz2_conf",
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
        "boltz/boltz2_conf.ckpt",
        2286561469,
    ),
    WeightItem(
        "boltz",
        "boltz2_aff",
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt",
        "boltz/boltz2_aff.ckpt",
        2062139170,
    ),
    WeightItem(
        "boltz",
        "boltz2_mols",
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar",
        "boltz/mols.tar",
        1855662080,
        extract=True,
    ),
    WeightItem(
        "boltz",
        "boltz1_conf",
        "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
        "boltz/boltz1_conf.ckpt",
        3595352714,
    ),
    WeightItem(
        "boltz",
        "boltz1_ccd",
        "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl",
        "boltz/ccd.pkl",
        345859128,
    ),
    WeightItem(
        "protenix",
        "protenix-v2",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt",
        "protenix/checkpoint/protenix-v2.pt",
        1859785497,
    ),
    WeightItem(
        "protenix",
        "protenix_base_default_v1.0.0",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt",
        "protenix/checkpoint/protenix_base_default_v1.0.0.pt",
        1475950125,
    ),
    WeightItem(
        "protenix",
        "protenix_base_20250630_v1.0.0",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_20250630_v1.0.0.pt",
        "protenix/checkpoint/protenix_base_20250630_v1.0.0.pt",
        1475945945,
    ),
    WeightItem(
        "protenix",
        "protenix_base_default_v0.5.0",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v0.5.0.pt",
        "protenix/checkpoint/protenix_base_default_v0.5.0.pt",
        1474265486,
    ),
    WeightItem(
        "protenix",
        "protenix_base_constraint_v0.5.0",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_constraint_v0.5.0.pt",
        "protenix/checkpoint/protenix_base_constraint_v0.5.0.pt",
        1475206741,
    ),
    WeightItem(
        "protenix",
        "protenix_mini_default_v0.5.0",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_mini_default_v0.5.0.pt",
        "protenix/checkpoint/protenix_mini_default_v0.5.0.pt",
        537049294,
    ),
    WeightItem(
        "protenix",
        "protenix_tiny_default_v0.5.0",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_tiny_default_v0.5.0.pt",
        "protenix/checkpoint/protenix_tiny_default_v0.5.0.pt",
        443171586,
    ),
    WeightItem(
        "protenix",
        "protenix_mini_esm_v0.5.0",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_mini_esm_v0.5.0.pt",
        "protenix/checkpoint/protenix_mini_esm_v0.5.0.pt",
        541640990,
    ),
    WeightItem(
        "protenix",
        "esm2_t36_3B_UR50D",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/esm2_t36_3B_UR50D.pt",
        "protenix/checkpoint/esm2_t36_3B_UR50D.pt",
        5678116398,
    ),
    WeightItem(
        "protenix",
        "esm2_t36_3B_UR50D-contact-regression",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/esm2_t36_3B_UR50D-contact-regression.pt",
        "protenix/checkpoint/esm2_t36_3B_UR50D-contact-regression.pt",
        6759,
    ),
    WeightItem(
        "protenix",
        "protenix_mini_ism_v0.5.0",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_mini_ism_v0.5.0.pt",
        "protenix/checkpoint/protenix_mini_ism_v0.5.0.pt",
        541640990,
    ),
    WeightItem(
        "protenix",
        "esm2_t36_3B_UR50D_ism",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/esm2_t36_3B_UR50D_ism.pt",
        "protenix/checkpoint/esm2_t36_3B_UR50D_ism.pt",
        11356246722,
    ),
    WeightItem(
        "protenix",
        "esm2_t36_3B_UR50D_ism-contact-regression",
        "https://protenix.tos-cn-beijing.volces.com/checkpoint/esm2_t36_3B_UR50D_ism-contact-regression.pt",
        "protenix/checkpoint/esm2_t36_3B_UR50D_ism-contact-regression.pt",
        7582,
    ),
    WeightItem(
        "protenix",
        "ccd_components_file",
        "https://protenix.tos-cn-beijing.volces.com/common/components.cif",
        "protenix/common/components.cif",
        490777362,
    ),
    WeightItem(
        "protenix",
        "ccd_components_rdkit_mol_file",
        "https://protenix.tos-cn-beijing.volces.com/common/components.cif.rdkit_mol.pkl",
        "protenix/common/components.cif.rdkit_mol.pkl",
        142498117,
    ),
    WeightItem(
        "protenix",
        "pdb_cluster_file",
        "https://protenix.tos-cn-beijing.volces.com/common/clusters-by-entity-40.txt",
        "protenix/common/clusters-by-entity-40.txt",
        21699572,
    ),
    WeightItem(
        "protenix",
        "obsolete_release_data_csv",
        "https://protenix.tos-cn-beijing.volces.com/common/obsolete_release_date.csv",
        "protenix/common/obsolete_release_date.csv",
        134716,
    ),
    WeightItem(
        "protenix",
        "obsolete_pdbs_path",
        "https://protenix.tos-cn-beijing.volces.com/common/obsolete_to_successor.json",
        "protenix/common/obsolete_to_successor.json",
        86882,
    ),
    WeightItem(
        "protenix",
        "release_dates_path",
        "https://protenix.tos-cn-beijing.volces.com/common/release_date_cache.json",
        "protenix/common/release_date_cache.json",
        12754898,
    ),
    WeightItem(
        "alphafold_multimer",
        "alphafold_params_2022-12-06",
        "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
        "alphafold_multimer/alphafold_params_2022-12-06.tar",
        5587968000,
        extract=True,
    ),
]


def should_bypass_proxy(url: str) -> bool:
    return "tos-cn-beijing.volces.com" in url


def wanted_items(families: set[str], names: set[str]) -> list[WeightItem]:
    items = WEIGHTS
    if families:
        items = [item for item in items if item.family in families]
    if names:
        items = [item for item in items if item.name in names]
    return items


def download_file(item: WeightItem, dest: Path, retries: int = 3) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and item.expected_size and dest.stat().st_size == item.expected_size:
        return {"status": "exists", "bytes": dest.stat().st_size}
    if dest.exists() and not item.expected_size and dest.stat().st_size > 0:
        return {"status": "exists", "bytes": dest.stat().st_size}

    part = dest.with_suffix(dest.suffix + ".part")
    session = requests.Session()
    if should_bypass_proxy(item.url):
        session.trust_env = False
    session.headers.update({"User-Agent": "StructPDE-WT-TT-weights/0.1"})
    last_error = ""
    for attempt in range(1, retries + 1):
        existing = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with session.get(item.url, stream=True, headers=headers, timeout=120, allow_redirects=True) as response:
                if response.status_code == 416 and part.exists():
                    part.replace(dest)
                    return {"status": "downloaded", "bytes": dest.stat().st_size}
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
                if existing and response.status_code != 206:
                    existing = 0
                    part.unlink(missing_ok=True)
                total = int(response.headers.get("content-length") or 0) + existing
                mode = "ab" if existing else "wb"
                with part.open(mode) as handle, tqdm(
                    total=total or item.expected_size,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    desc=item.name,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024 * 4):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bar.update(len(chunk))
            part.replace(dest)
            if item.expected_size and dest.stat().st_size != item.expected_size:
                raise RuntimeError(
                    f"Downloaded size mismatch: got {dest.stat().st_size}, expected {item.expected_size}"
                )
            return {"status": "downloaded", "bytes": dest.stat().st_size}
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(5 * attempt)
    return {"status": "failed", "bytes": dest.stat().st_size if dest.exists() else 0, "note": last_error}


def size_from_content_range(content_range: str | None) -> int | None:
    if not content_range:
        return None
    match = re.search(r"/(\d+)$", content_range)
    return int(match.group(1)) if match else None


def extract_if_needed(item: WeightItem, dest: Path) -> str:
    if not item.extract or not dest.exists():
        return ""
    if item.name == "boltz2_mols":
        marker = dest.parent / "mols" / ".extracted_from_mols_tar"
        if marker.exists():
            return "already_extracted"
        with tarfile.open(dest) as tar:
            tar.extractall(dest.parent)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        return "extracted"
    if item.name.startswith("alphafold_params"):
        extract_dir = dest.parent / "params"
        marker = extract_dir / ".extracted_from_alphafold_params"
        if marker.exists():
            return "already_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest) as tar:
            tar.extractall(extract_dir)
        marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
        return "extracted"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="*", default=[])
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    configure_local_cache()
    root = MODELS / "structure_predictors"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for item in wanted_items(set(args.families), set(args.names)):
        dest = root / item.relative_path
        result = download_file(item, dest)
        extract_status = ""
        if result["status"] in {"downloaded", "exists"} and not args.skip_extract:
            extract_status = extract_if_needed(item, dest)
        records.append(
            {
                **asdict(item),
                "path": str(dest),
                **result,
                "extract_status": extract_status,
            }
        )
        (root / "weights_manifest.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(root / "weights_manifest.json")
    if shutil.which("du") and os.name != "nt":
        os.system(f"du -sh {root}")


if __name__ == "__main__":
    main()
