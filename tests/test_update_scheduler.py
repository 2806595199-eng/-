from pathlib import Path

from core import config as cfg
from serving.online_history import record_feedback
from training.update_scheduler import run_scheduled_update


def _sample(i):
    return {
        "timestamp": f"2026-05-28 11:{i:02d}:00",
        "influent_flow": 100.0,
        "influent_ph": 7.2,
        "conductivity": 6500.0,
        "influent_f": 18.0 + i,
        "pacl_dose": 500.0,
        "defluor_dose": 0.5,
        "pacl_tank_ph": 7.0,
        "defluor_tank_ph": 6.0,
        "recycle_flow": 4.0,
        "waste_flow": 15.0,
        "pam_dose": 0.5,
        "effluent_f": 0.7,
    }


def test_scheduled_update_skips_when_feedback_rows_are_not_enough(tmp_path):
    record_feedback(_sample(0), history_dir=tmp_path / "history")

    result = run_scheduled_update(
        history_dir=tmp_path / "history",
        models_root=tmp_path / "models",
        min_rows=2,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "not_enough_feedback_rows"


def test_scheduled_update_builds_training_file_and_calls_update(tmp_path, monkeypatch):
    for i in range(2):
        record_feedback(_sample(i), history_dir=tmp_path / "history")

    calls = {}

    def fake_update_model_from_file(source_path, **kwargs):
        calls["source_path"] = source_path
        calls["kwargs"] = kwargs
        return {"status": "trained", "published": True, "version_dir": "v1"}

    monkeypatch.setattr("training.update_scheduler.update_model_from_file",
                        fake_update_model_from_file)

    result = run_scheduled_update(
        history_dir=tmp_path / "history",
        models_root=tmp_path / "models",
        min_rows=2,
        auto_publish=True,
        min_r2=0.7,
    )

    assert result["status"] == "trained"
    assert Path(calls["source_path"]).exists()
    assert calls["kwargs"]["require_better_than_active"] is True
    assert calls["kwargs"]["auto_publish"] is True
    assert calls["kwargs"]["min_r2"] == 0.7
    assert calls["kwargs"]["device"] == cfg.DEVICE
