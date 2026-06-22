"""测试：推理引擎基础"""
import json
import pickle

import numpy as np
import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from serving.inference_engine import InferenceEngine


class FakeBackupModel:
    def predict(self, values):
        return [0.55 for _ in values]


def test_no_adaptation_attrs():
    """不应有 calibrator/monitor/drift_detector/ensemble"""
    eng = InferenceEngine("models")
    assert not hasattr(eng, "calibrator"), "calibrator should be removed"
    assert not hasattr(eng, "monitor"), "monitor should be removed"
    assert not hasattr(eng, "drift_detector"), "drift_detector should be removed"
    assert not hasattr(eng, "ensemble"), "ensemble should be removed"


def test_no_adaptation_methods():
    """不应有 predict_with_adaptation"""
    eng = InferenceEngine("models")
    assert not hasattr(eng, "predict_with_adaptation"), \
        "predict_with_adaptation should be removed"


def test_no_adaptive_state_methods():
    """不应有自适应状态方法"""
    eng = InferenceEngine("models")
    assert not hasattr(eng, "save_adaptive_state")
    assert not hasattr(eng, "_load_adaptive_state")


def test_load_requires_core_model_artifacts(tmp_path, monkeypatch):
    """缺少训练产物时不能被 health 误判为可预测。"""
    import serving.inference_engine as inference_engine

    class FakeTabPFN:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(inference_engine, "_get_tabpfn", lambda: FakeTabPFN)
    eng = InferenceEngine(tmp_path)

    with pytest.raises(RuntimeError, match="缺少模型产物"):
        eng.load()


def test_load_reports_active_model_version(tmp_path, monkeypatch):
    import serving.inference_engine as inference_engine

    class FakeTabPFN:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            pass

    version_dir = tmp_path / "versions" / "20260529_000012_120382"
    version_dir.mkdir(parents=True)
    (tmp_path / "active_model.json").write_text(
        json.dumps({
            "active_version": "20260529_000012_120382",
            "active_path": "versions/20260529_000012_120382",
            "metrics": {"r2_test": 0.9883},
        }),
        encoding="utf-8",
    )
    (version_dir / "feature_config.json").write_text(
        json.dumps({"feature_names": []}),
        encoding="utf-8",
    )
    with open(version_dir / "scaler.pkl", "wb") as f:
        pickle.dump(StandardScaler().fit(np.array([[0.0], [1.0]])), f)
    with open(version_dir / "train_data.pkl", "wb") as f:
        pickle.dump({"X_train": np.array([[0.0]]), "y_train": np.array([0.5])}, f)
    (version_dir / "model_metadata.json").write_text(
        json.dumps({"r2_test": 0.9883, "residual_std": 0.04}),
        encoding="utf-8",
    )

    monkeypatch.setattr(inference_engine, "_get_tabpfn", lambda: FakeTabPFN)

    eng = InferenceEngine(tmp_path)
    eng.load()

    assert eng.version == "20260529_000012_120382"


def test_predict_projects_current_setpoint_to_feature_delay_horizon():
    class FakeEngineer:
        min_history = 10

        def __init__(self):
            self.seen = None

        def prediction_horizon_steps(self):
            return 2

        def transform(self, df):
            self.seen = df.copy()
            return pd.DataFrame({"x": [1.0] * len(df)})

    class FakeModel:
        def predict(self, values):
            return [0.6]

    history = pd.DataFrame([
        {"pacl_dose": 100.0, "influent_f": 10.0},
        {"pacl_dose": 120.0, "influent_f": 11.0},
    ])
    water_quality = {"pacl_dose": 300.0, "influent_f": 12.0}

    eng = InferenceEngine("models")
    eng.engineer = FakeEngineer()
    eng.model = FakeModel()
    eng.main_model_fitted = True
    eng.feature_names = ["x"]

    eng.predict(water_quality, history=history)

    assert len(eng.engineer.seen) == len(history) + 1 + 2
    assert eng.engineer.seen.tail(3)["pacl_dose"].tolist() == [300.0, 300.0, 300.0]


def test_predict_uses_backup_when_main_model_is_not_fitted():
    class UnfittedMainModel:
        def predict(self, values):
            raise AssertionError("unfitted main model should not be called")

    eng = InferenceEngine("models")
    eng.model = UnfittedMainModel()
    eng.main_model_fitted = False
    eng.backup_model = FakeBackupModel()

    result = eng.predict({"influent_flow": 100.0, "influent_f": 2.0})

    assert result["model_used"] == "xgboost"
    assert result["predicted_f"] == 0.55


def test_load_keeps_backup_available_when_tabpfn_fit_fails(tmp_path, monkeypatch):
    import serving.inference_engine as inference_engine

    class FailingTabPFN:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X, y):
            raise PermissionError("tabpfn checkpoint denied")

    (tmp_path / "feature_config.json").write_text(
        json.dumps({"feature_names": []}),
        encoding="utf-8",
    )
    with open(tmp_path / "scaler.pkl", "wb") as f:
        pickle.dump(StandardScaler().fit(np.array([[0.0], [1.0]])), f)
    with open(tmp_path / "train_data.pkl", "wb") as f:
        pickle.dump({"X_train": np.array([[0.0]]), "y_train": np.array([0.5])}, f)
    with open(tmp_path / "backup_model.pkl", "wb") as f:
        pickle.dump(FakeBackupModel(), f)

    monkeypatch.setattr(inference_engine, "_get_tabpfn", lambda: FailingTabPFN)

    eng = InferenceEngine(tmp_path)
    eng.load()

    assert eng.main_model_fitted is False
    assert eng.backup_model is not None


def test_load_skips_tabpfn_when_backup_is_forced(tmp_path, monkeypatch):
    import serving.inference_engine as inference_engine

    called = {"tabpfn": False}

    class FakeTabPFN:
        def __init__(self, *args, **kwargs):
            called["tabpfn"] = True

        def fit(self, X, y):
            pass

    (tmp_path / "feature_config.json").write_text(
        json.dumps({"feature_names": []}),
        encoding="utf-8",
    )
    with open(tmp_path / "scaler.pkl", "wb") as f:
        pickle.dump(StandardScaler().fit(np.array([[0.0], [1.0]])), f)
    with open(tmp_path / "train_data.pkl", "wb") as f:
        pickle.dump({"X_train": np.array([[0.0]]), "y_train": np.array([0.5])}, f)
    with open(tmp_path / "backup_model.pkl", "wb") as f:
        pickle.dump(FakeBackupModel(), f)

    monkeypatch.setenv("USE_BACKUP", "true")
    monkeypatch.setattr(inference_engine, "_get_tabpfn", lambda: FakeTabPFN)

    eng = InferenceEngine(tmp_path)
    eng.load()

    assert called["tabpfn"] is False
    assert eng.main_model_fitted is False
    assert eng.backup_model is not None


def test_load_uses_resolved_tabpfn_device(tmp_path, monkeypatch):
    import serving.inference_engine as inference_engine

    captured = {}

    class FakeTabPFN:
        def __init__(self, *args, **kwargs):
            captured["device"] = kwargs.get("device")

        def fit(self, X, y):
            captured["fit_shape"] = X.shape

    (tmp_path / "feature_config.json").write_text(
        json.dumps({"feature_names": []}),
        encoding="utf-8",
    )
    with open(tmp_path / "scaler.pkl", "wb") as f:
        pickle.dump(StandardScaler().fit(np.array([[0.0], [1.0]])), f)
    with open(tmp_path / "train_data.pkl", "wb") as f:
        pickle.dump({"X_train": np.array([[0.0]]), "y_train": np.array([0.5])}, f)

    monkeypatch.setattr(inference_engine, "_get_tabpfn", lambda: FakeTabPFN)
    monkeypatch.setattr(inference_engine, "resolve_tabpfn_device", lambda device=None: "cuda")

    eng = InferenceEngine(tmp_path)
    eng.load()

    assert captured["device"] == "cuda"
    assert eng.main_model_fitted is True
