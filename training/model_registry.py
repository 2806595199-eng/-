"""Versioned model artifact registry."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_MODEL_ARTIFACTS = ["feature_config.json", "scaler.pkl", "train_data.pkl"]
ACTIVE_MODEL_FILE = "active_model.json"


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def create_version_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def create_version_dir(models_root: Path | str = "models",
                       version_id: Optional[str] = None) -> Path:
    root = Path(models_root)
    version = version_id or create_version_id()
    version_dir = root / "versions" / version
    version_dir.mkdir(parents=True, exist_ok=False)
    return version_dir


def missing_artifacts(model_dir: Path | str) -> list[str]:
    path = Path(model_dir)
    return [name for name in REQUIRED_MODEL_ARTIFACTS if not (path / name).exists()]


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def publish_active_model(models_root: Path | str,
                         version_dir: Path | str,
                         metrics: Optional[Dict[str, Any]] = None,
                         note: Optional[str] = None) -> Dict[str, Any]:
    root = Path(models_root)
    path = Path(version_dir)
    missing = missing_artifacts(path)
    if missing:
        raise ValueError(f"Cannot publish model version; missing artifacts: {missing}")

    active = {
        "active_version": path.name,
        "active_path": _relative_to_root(path, root),
        "activated_at": datetime.now().isoformat(),
        "metrics": metrics or {},
    }
    if note:
        active["note"] = note

    root.mkdir(parents=True, exist_ok=True)
    (root / ACTIVE_MODEL_FILE).write_text(
        json.dumps(active, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return active


def read_active_model(models_root: Path | str = "models") -> Optional[Dict[str, Any]]:
    active_path = Path(models_root) / ACTIVE_MODEL_FILE
    if not active_path.exists():
        return None
    return json.loads(active_path.read_text(encoding="utf-8"))


def resolve_model_dir(models_root: Path | str = "models") -> Path:
    """Resolve active version path, falling back to legacy root layout."""
    root = Path(models_root)
    active = read_active_model(root)
    if not active:
        return root

    active_path = Path(active["active_path"])
    if not active_path.is_absolute():
        active_path = root / active_path
    return active_path
