import json
from pathlib import Path

import pandas as pd

from core import config as cfg
from training.model_update import update_model_from_file


def _valid_rows():
    base = {
        "influent_flow": 100.0,
        "influent_ph": 7.2,
        "conductivity": 6500.0,
        "influent_f": 18.0,
        "pacl_dose": 500.0,
        "defluor_dose": 0.5,
        "pacl_tank_ph": 7.0,
        "defluor_tank_ph": 6.0,
        "recycle_flow": 0.3,
        "waste_flow": 2.5,
        "pam_dose": 3.8,
        "effluent_f": 0.7,
    }
    rows = []
    for i in range(5):
        row = dict(base)
        row["timestamp"] = f"2026-05-27 11:0{i}:00"
        row["effluent_f"] = 0.7 + i * 0.01
        rows.append(row)
    return rows


def test_update_model_from_file_creates_version_and_publishes(tmp_path, monkeypatch):
    source = tmp_path / "site.csv"
    pd.DataFrame(_valid_rows()).to_csv(source, index=False, encoding="utf-8-sig")
    captured = {}

    def fake_train_main(data_path=None, device="cpu", output_dir="models"):
        captured["device"] = device
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "feature_config.json").write_text("{}", encoding="utf-8")
        (out / "scaler.pkl").write_bytes(b"placeholder")
        (out / "train_data.pkl").write_bytes(b"placeholder")
        (out / "model_metadata.json").write_text(
            json.dumps({"r2_test": 0.9, "rmse": 0.08}),
            encoding="utf-8",
        )
        return {"r2_test": 0.9, "rmse": 0.08}

    monkeypatch.setattr("training.model_update.train_main", fake_train_main)

    result = update_model_from_file(
        source,
        models_root=tmp_path / "models",
        raw_dir=tmp_path / "raw",
        prepared_dir=tmp_path / "prepared",
        report_dir=tmp_path / "reports",
        version_id="test_version",
        auto_publish=True,
    )

    version_dir = Path(result["version_dir"])
    assert result["published"] is True
    assert captured["device"] == cfg.DEVICE
    assert (version_dir / "data_quality_report.json").exists()
    assert (version_dir / "update_report.json").exists()
    active = json.loads((tmp_path / "models" / "active_model.json").read_text(encoding="utf-8"))
    assert active["active_version"] == "test_version"


def test_update_model_requires_better_than_active_to_publish(tmp_path, monkeypatch):
    source = tmp_path / "site.csv"
    pd.DataFrame(_valid_rows()).to_csv(source, index=False, encoding="utf-8-sig")
    models_root = tmp_path / "models"
    models_root.mkdir()
    (models_root / "active_model.json").write_text(
        json.dumps({
            "active_version": "old",
            "active_path": "versions/old",
            "metrics": {"r2_test": 0.95, "rmse": 0.05},
        }),
        encoding="utf-8",
    )

    def fake_train_main(data_path=None, device="cpu", output_dir="models"):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "feature_config.json").write_text("{}", encoding="utf-8")
        (out / "scaler.pkl").write_bytes(b"placeholder")
        (out / "train_data.pkl").write_bytes(b"placeholder")
        (out / "model_metadata.json").write_text(
            json.dumps({"r2_test": 0.9, "rmse": 0.08}),
            encoding="utf-8",
        )
        return {"r2_test": 0.9, "rmse": 0.08}

    monkeypatch.setattr("training.model_update.train_main", fake_train_main)

    result = update_model_from_file(
        source,
        models_root=models_root,
        raw_dir=tmp_path / "raw",
        prepared_dir=tmp_path / "prepared",
        report_dir=tmp_path / "reports",
        version_id="worse_version",
        auto_publish=True,
        require_better_than_active=True,
    )

    active = json.loads((models_root / "active_model.json").read_text(encoding="utf-8"))
    assert result["published"] is False
    assert result["publish_decision"]["passed"] is False
    assert active["active_version"] == "old"
