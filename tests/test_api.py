"""测试：FastAPI 接口（不依赖真实 TabPFN）"""
from pathlib import Path

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from core import config as cfg


class FakeRecipe:
    def __init__(self, p, d):
        self.pacl_dose_setpoint = p
        self.defluor_dose_setpoint = d


class FakeEngine:
    main_model_fitted = True
    backup_model = True         # health 检查
    backup_model_loaded = True
    version = "0.0.0"
    model = True

    def predict(self, wq, history=None):
        return {"predicted_f": 0.5, "q05": 0.3, "q95": 0.7,
                "risk_level": "safe", "model_used": "tabpfn", "warnings": []}

    def predict_batch(self, samples, prefer_model=None, history=None):
        return [self.predict(s) for s in samples]


class FakeOptimizer:
    def optimize(self, wq, engine, mode="balanced"):
        return {
            "recommended_mode": mode,
            "recommended": {
                "scheme_label": mode,
                "pacl_dose_setpoint": 500.0,
                "defluor_dose_setpoint": 0.5,
                "pacl_pump_flow_l_h": 500.0,
                "defluor_pump_flow_l_h": 50.0,
                "pump_flow_unit": "L/h",
                "formula_basis": {"pacl": "mg_L_product", "defluor": "mL_L_stock"},
                "predicted_f": 0.5,
                "q05": 0.3,
                "q95": 0.7,
                "risk_level": "safe",
                "model_used": "xgboost",
                "cost_per_ton": 1.0,
                "cost_per_hour_yuan": 100.0,
                "cost_breakdown": {"pacl_yuan_per_ton": 0.4, "defluor_yuan_per_ton": 0.6,
                                   "pam_yuan_per_ton": 0, "magnetic_yuan_per_ton": 0},
                "dose_score": 0.5,
                "quality_score": 0.0,
                "balanced_score": 0.5,
                "selection_reason": "balanced_min_score",
                "warnings": [],
            },
            "alternatives": {
                "economic": {"pacl_dose_setpoint": 400, "defluor_dose_setpoint": 0.4,
                             "predicted_f": 0.6, "q95": 0.8, "risk_level": "safe",
                             "cost_per_ton": 0.8, "model_used": "xgboost"},
                "balanced": {"pacl_dose_setpoint": 500, "defluor_dose_setpoint": 0.5,
                             "predicted_f": 0.5, "q95": 0.7, "risk_level": "safe",
                             "cost_per_ton": 1.0, "model_used": "xgboost"},
                "safe": {"pacl_dose_setpoint": 600, "defluor_dose_setpoint": 0.6,
                         "predicted_f": 0.4, "q95": 0.6, "risk_level": "safe",
                         "cost_per_ton": 1.2, "model_used": "xgboost"},
            },
            "pareto_front": [{"pacl_dose_setpoint": 400, "defluor_dose_setpoint": 0.4,
                              "predicted_f": 0.6, "q95": 0.8, "cost_per_ton": 0.8,
                              "balanced_score": 0.3, "risk_level": "safe"}],
            "warnings": [],
            "assumptions": {"optimizer_model": "xgboost", "cost_prices_need_confirmation": True},
        }


@pytest.fixture
def client(monkeypatch, tmp_path):
    from serving.serve import app
    monkeypatch.setattr("serving.serve.engine", FakeEngine())
    monkeypatch.setattr("serving.serve.optimizer", FakeOptimizer())
    monkeypatch.setattr("serving.serve.model_loaded", True)
    monkeypatch.setattr("serving.serve.ONLINE_HISTORY_DIR", tmp_path)
    return TestClient(app)


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    d = r.json()
    assert "status" in d
    assert "model_loaded" in d
    assert "uptime" in d


def test_ready_reports_model_readiness(client):
    r = client.get("/api/v1/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


def test_ready_fails_when_model_is_not_loaded(client, monkeypatch):
    monkeypatch.setattr("serving.serve.model_loaded", False)

    r = client.get("/api/v1/ready")

    assert r.status_code == 503
    assert r.json()["status"] == "not_ready"
    assert r.json()["checks"]["model_loaded"] is False


def test_client_fixture_uses_isolated_online_history(client, tmp_path):
    from serving import serve

    assert Path(serve.ONLINE_HISTORY_DIR) == tmp_path


def test_predict(client):
    r = client.post("/api/v1/predict", json={
        "influent_flow": 100, "influent_ph": 7.2, "conductivity": 6500,
        "influent_f": 18, "pacl_dose": 500, "defluor_dose": 0.5,
        "pacl_tank_ph": 7, "defluor_tank_ph": 6, "recycle_flow": 4,
        "waste_flow": 2.5, "pam_dose": 0.5,
    })
    assert r.status_code == 200
    d = r.json()
    assert "predicted_f" in d
    assert "risk_level" in d
    assert "model_used" in d
    assert "record_id" in d


def test_predict_no_effluent_f(client):
    """WQInput 不应包含 effluent_f 作为预测输入"""
    r = client.post("/api/v1/predict", json={
        "influent_flow": 100, "influent_ph": 7.2, "conductivity": 6500,
        "influent_f": 18, "pacl_dose": 500, "defluor_dose": 0.5,
        "pacl_tank_ph": 7, "defluor_tank_ph": 6, "recycle_flow": 4,
        "waste_flow": 2.5, "pam_dose": 0.5, "effluent_f": 1.5,
    })
    assert r.status_code == 200


def test_recommend(client):
    r = client.post("/api/v1/dose/recommend", json={
        "mode": "balanced",
        "water_quality": {
            "influent_flow": 100, "influent_ph": 7.2, "conductivity": 6500,
            "influent_f": 18, "pacl_dose": 500, "defluor_dose": 0.5,
            "pacl_tank_ph": 7, "defluor_tank_ph": 6, "recycle_flow": 4,
            "waste_flow": 2.5, "pam_dose": 0.5,
        }
    })
    assert r.status_code == 200
    d = r.json()
    assert "recommended" in d
    assert "alternatives" in d
    assert "pareto_front" in d
    assert "record_id" in d
    rec = d["recommended"]
    assert "pacl_dose_setpoint" in rec
    assert "defluor_dose_setpoint" in rec
    assert "cost_per_ton" in rec
    assert "model_used" in rec


def test_batch_recommend(client):
    r = client.post("/api/v1/dose/recommend/batch", json={
        "mode": "balanced",
        "records": [
            {"influent_flow": 100, "influent_ph": 7.2, "conductivity": 6500,
             "influent_f": 18, "pacl_dose": 500, "defluor_dose": 0.5}
        ]
    })
    assert r.status_code == 200
    d = r.json()
    assert "pacl_dose_setpoint" in d
    assert "defluor_dose_setpoint" in d
    assert "based_on_records" in d


def test_recommend_feedback_training_row_uses_recommended_doses(client, tmp_path, monkeypatch):
    monkeypatch.setattr("serving.serve.ONLINE_HISTORY_DIR", tmp_path)

    r = client.post("/api/v1/dose/recommend", json={
        "mode": "balanced",
        "water_quality": {
            "influent_flow": 100, "influent_ph": 7.2, "conductivity": 6500,
            "influent_f": 18, "pacl_dose": 100, "defluor_dose": 0.1,
            "pacl_tank_ph": 7, "defluor_tank_ph": 6, "recycle_flow": 4,
            "waste_flow": 2.5, "pam_dose": 0.5,
        }
    })
    assert r.status_code == 200
    record_id = r.json()["record_id"]

    fb = client.post("/api/v1/feedback", json={
        "record_id": record_id,
        "effluent_f": 0.72,
    })
    assert fb.status_code == 200

    from serving.online_history import build_feedback_training_file

    built = build_feedback_training_file(tmp_path)
    df = pd.read_csv(built["training_path"])

    assert built["row_count"] == 1
    assert df.loc[0, "pacl_dose"] == 500.0
    assert df.loc[0, "defluor_dose"] == 0.5


def test_save_result_does_not_overwrite_same_second(tmp_path, monkeypatch):
    from serving import serve

    monkeypatch.chdir(tmp_path)
    payload = {"predicted_f": 0.7, "q95": 0.8,
               "risk_level": "warning", "model_used": "tabpfn"}

    serve._save_result("predict", payload)
    serve._save_result("predict", payload)

    files = list((tmp_path / "logs" / "results").glob("predict_*.json"))
    assert len(files) == 2


def test_plc_schema_404(client):
    r = client.get("/api/v1/plc/schema")
    assert r.status_code == 404


def test_debug_404(client):
    r = client.get("/api/v1/debug/model_status")
    assert r.status_code == 404


def test_feedback_endpoint_records_effluent(client, tmp_path, monkeypatch):
    monkeypatch.setattr("serving.serve.ONLINE_HISTORY_DIR", tmp_path)
    r = client.post("/api/v1/feedback", json={
        "influent_flow": 100, "influent_ph": 7.2, "conductivity": 6500,
        "influent_f": 18, "pacl_dose": 500, "defluor_dose": 0.5,
        "pacl_tank_ph": 7, "defluor_tank_ph": 6, "recycle_flow": 4,
        "waste_flow": 2.5, "pam_dose": 0.5,
        "effluent_f": 0.72,
    })

    assert r.status_code == 200
    assert r.json()["status"] == "recorded"


def test_history_status_endpoint(client, tmp_path, monkeypatch):
    monkeypatch.setattr("serving.serve.ONLINE_HISTORY_DIR", tmp_path)
    client.post("/api/v1/feedback", json={
        "influent_flow": 100, "influent_ph": 7.2, "conductivity": 6500,
        "influent_f": 18, "pacl_dose": 500, "defluor_dose": 0.5,
        "pacl_tank_ph": 7, "defluor_tank_ph": 6, "recycle_flow": 4,
        "waste_flow": 2.5, "pam_dose": 0.5,
        "effluent_f": 0.72,
    })

    r = client.get("/api/v1/history/status")

    assert r.status_code == 200
    assert r.json()["feedback_count"] == 1


def test_model_update_from_feedback_endpoint(client, tmp_path, monkeypatch):
    monkeypatch.setattr("serving.serve.ONLINE_HISTORY_DIR", tmp_path)
    captured = {}

    def fake_run_scheduled_update(**kwargs):
        captured["kwargs"] = kwargs
        return {"status": "trained", "published": True, "feedback_row_count": 10}

    monkeypatch.setattr("serving.serve.run_scheduled_update", fake_run_scheduled_update)

    r = client.post("/api/v1/model/update/from-feedback", json={
        "min_rows": 10,
        "publish": True,
        "min_r2": 0.7,
    })

    assert r.status_code == 200
    assert r.json()["status"] == "trained"
    assert r.json()["published"] is True
    assert captured["kwargs"]["device"] == cfg.DEVICE
