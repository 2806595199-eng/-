import pandas as pd
import json

from core import config as cfg
from training import train as train_module


class FakeBackupModel:
    def predict(self, values):
        return [0.5 for _ in values]


def _training_rows(n=40):
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": f"2026-01-01 00:{i:02d}:00",
            "influent_flow": 100 + i * 0.1,
            "influent_ph": 7.2,
            "conductivity": 6500,
            "influent_f": 18 + i * 0.01,
            "pacl_dose": 900,
            "defluor_dose": 1.5,
            "pacl_tank_ph": 7.0,
            "defluor_tank_ph": 6.2,
            "recycle_flow": 4.0,
            "waste_flow": 15.0,
            "pam_dose": 0.5,
            "effluent_f": 0.8,
        })
    return rows


def test_training_exports_artifacts_when_tabpfn_unavailable(tmp_path, monkeypatch):
    data_path = tmp_path / "site.csv"
    pd.DataFrame(_training_rows()).to_csv(data_path, index=False)

    def failing_tabpfn(*args, **kwargs):
        raise PermissionError("tabpfn checkpoint denied")

    def fake_xgboost(*args, **kwargs):
        return {
            "model": FakeBackupModel(),
            "mape": 0.1,
            "r2": 0.8,
            "rmse": 0.1,
            "best_params": {"model_type": "fake"},
            "importance": [],
            "train_time": 0.0,
        }

    monkeypatch.setattr(train_module, "train_tabpfn", failing_tabpfn)
    monkeypatch.setattr(train_module, "cross_validate", lambda *a, **k: {
        "cv_r2_mean": 0,
        "cv_r2_std": 0,
    })
    monkeypatch.setattr(train_module, "train_xgboost", fake_xgboost)

    result = train_module.main(str(data_path), device="cpu", output_dir=str(tmp_path / "models"))

    assert result["tabpfn_available"] is False
    assert (tmp_path / "models" / "feature_config.json").exists()
    assert (tmp_path / "models" / "backup_model.pkl").exists()


def test_training_default_device_uses_config(tmp_path, monkeypatch):
    data_path = tmp_path / "site.csv"
    pd.DataFrame(_training_rows()).to_csv(data_path, index=False)
    captured = {}

    def fake_tabpfn(*args, **kwargs):
        captured["device"] = kwargs.get("device")
        return {
            "model": object(),
            "r2_train": 0.9,
            "r2_test": 0.8,
            "rmse": 0.1,
            "mae": 0.05,
            "residual_std": 0.02,
            "y_pred": [],
            "y_test": [],
        }

    def fake_xgboost(*args, **kwargs):
        return {
            "model": FakeBackupModel(),
            "mape": 0.1,
            "r2": 0.8,
            "rmse": 0.1,
            "best_params": {"model_type": "fake"},
            "importance": [],
            "train_time": 0.0,
        }

    monkeypatch.setattr(train_module, "train_tabpfn", fake_tabpfn)
    monkeypatch.setattr(train_module, "cross_validate", lambda *a, **k: {
        "cv_r2_mean": 0,
        "cv_r2_std": 0,
    })
    monkeypatch.setattr(train_module, "train_xgboost", fake_xgboost)

    train_module.main(str(data_path), output_dir=str(tmp_path / "models"))

    assert captured["device"] == cfg.DEVICE
    metadata = json.loads((tmp_path / "models" / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["xgb_r2"] == 0.8
    assert metadata["xgb_rmse"] == 0.1
    assert metadata["xgb_mape"] == 0.1
