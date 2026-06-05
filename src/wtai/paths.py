from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
ROOT = CODE_ROOT.parent if CODE_ROOT.name in {"codes", "published_codes"} else CODE_ROOT
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXTERNAL = DATA / "external"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"


def ensure_project_dirs() -> None:
    for path in [
        DATA,
        RAW,
        PROCESSED,
        EXTERNAL,
        MODELS,
        RESULTS,
        RESULTS / "figures",
        RESULTS / "tables",
        RESULTS / "models",
        CACHE,
        CACHE / "pip",
        CACHE / "hf",
        CACHE / "torch",
        CACHE / "matplotlib",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def configure_local_cache() -> None:
    """Force common libraries to keep cache files inside this repository."""
    ensure_project_dirs()
    env = {
        "PIP_CACHE_DIR": CACHE / "pip",
        "HF_HOME": CACHE / "hf",
        "HUGGINGFACE_HUB_CACHE": CACHE / "hf" / "hub",
        "TRANSFORMERS_CACHE": CACHE / "hf" / "transformers",
        "TORCH_HOME": CACHE / "torch",
        "MPLCONFIGDIR": CACHE / "matplotlib",
        "XDG_CACHE_HOME": CACHE / "xdg",
        "TMPDIR": CACHE / "tmp",
        "TEMP": CACHE / "tmp",
        "TMP": CACHE / "tmp",
    }
    for value in env.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    for key, value in env.items():
        os.environ.setdefault(key, str(value))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))
