from pathlib import Path

from serving.inference_engine import InferenceEngine
from training.model_registry import (
    create_version_dir,
    publish_active_model,
    read_active_model,
    resolve_model_dir,
)


def _write_minimal_artifacts(version_dir: Path):
    (version_dir / "feature_config.json").write_text("{}", encoding="utf-8")
    (version_dir / "scaler.pkl").write_bytes(b"placeholder")
    (version_dir / "train_data.pkl").write_bytes(b"placeholder")


def test_publish_and_resolve_active_model(tmp_path):
    version_dir = create_version_dir(tmp_path, version_id="20260527_170000")
    _write_minimal_artifacts(version_dir)

    publish_active_model(tmp_path, version_dir, metrics={"r2_test": 0.81})

    active = read_active_model(tmp_path)
    assert active["active_version"] == "20260527_170000"
    assert active["metrics"]["r2_test"] == 0.81
    assert resolve_model_dir(tmp_path) == version_dir


def test_inference_engine_uses_active_model_pointer(tmp_path):
    version_dir = create_version_dir(tmp_path, version_id="v1")
    _write_minimal_artifacts(version_dir)
    publish_active_model(tmp_path, version_dir)

    engine = InferenceEngine(tmp_path)

    assert engine.models_dir == version_dir
