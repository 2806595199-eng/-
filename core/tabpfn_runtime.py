"""TabPFN runtime configuration helpers."""

import os
from pathlib import Path

from core import config as cfg


def configure_tabpfn_cache(cache_dir=None) -> Path:
    """Point TabPFN at a project-local model cache before importing tabpfn."""
    if cache_dir is None:
        cache_dir = cfg.TABPFN_MODEL_CACHE_DIR

    path = Path(cache_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(path)
    return path


def resolve_tabpfn_device(device: str | None = None) -> str:
    """Resolve 'auto' into the best available TabPFN device."""
    requested = (device or cfg.DEVICE or "auto").lower()
    if requested != "auto":
        return requested

    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
