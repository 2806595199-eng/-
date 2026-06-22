import json
from pathlib import Path

import pandas as pd

from serving.online_history import (
    build_feedback_training_file,
    history_status,
    record_feedback,
    record_runtime_event,
)
from core import config as cfg


def _water_quality():
    return {
        "influent_flow": 100.0,
        "influent_ph": 7.2,
        "conductivity": 6500.0,
        "influent_f": 18.0,
        "pacl_dose": 500.0,
        "defluor_dose": 0.5,
        "pacl_tank_ph": 7.0,
        "defluor_tank_ph": 6.0,
        "recycle_flow": 4.0,
        "waste_flow": 15.0,
        "pam_dose": 0.5,
    }


def test_runtime_event_and_feedback_by_record_id_create_training_row(tmp_path):
    event = record_runtime_event(
        "predict",
        _water_quality(),
        {"predicted_f": 0.6, "q95": 0.8, "risk_level": "warning"},
        history_dir=tmp_path,
    )

    feedback = record_feedback({
        "record_id": event["record_id"],
        "effluent_f": 0.72,
        "timestamp": "2026-05-28 10:00:00",
    }, history_dir=tmp_path)
    output = build_feedback_training_file(tmp_path, output_path=tmp_path / "feedback.csv")

    df = pd.read_csv(output["training_path"], encoding="utf-8-sig")
    assert feedback["record_id"] == event["record_id"]
    assert len(df) == 1
    assert df.loc[0, "influent_f"] == 18.0
    assert df.loc[0, "effluent_f"] == 0.72


def test_direct_feedback_full_sample_does_not_need_record_id(tmp_path):
    payload = {**_water_quality(), "effluent_f": 0.68, "timestamp": "2026-05-28 10:05:00"}

    record_feedback(payload, history_dir=tmp_path)
    output = build_feedback_training_file(tmp_path, output_path=tmp_path / "feedback.csv")

    df = pd.read_csv(output["training_path"], encoding="utf-8-sig")
    assert len(df) == 1
    assert df.loc[0, "pacl_dose"] == 500.0
    assert df.loc[0, "effluent_f"] == 0.68


def test_feedback_timestamp_matches_event_by_hrt_when_record_id_missing(tmp_path):
    event_time = pd.Timestamp("2026-05-28 10:00:00")
    feedback_time = event_time + pd.Timedelta(
        minutes=max(cfg.FEATURE_DELAY_STEPS.values()) * cfg.MODEL_SAMPLE_INTERVAL_MIN
    )

    record_runtime_event(
        "predict",
        {**_water_quality(), "timestamp": event_time.isoformat()},
        {"predicted_f": 0.6},
        history_dir=tmp_path,
    )
    record_feedback({
        "effluent_f": 0.71,
        "timestamp": feedback_time.isoformat(),
    }, history_dir=tmp_path)

    output = build_feedback_training_file(tmp_path, output_path=tmp_path / "feedback.csv")
    df = pd.read_csv(output["training_path"], encoding="utf-8-sig")

    assert output["row_count"] == 1
    assert df.loc[0, "influent_f"] == 18.0
    assert df.loc[0, "effluent_f"] == 0.71


def test_history_status_counts_events_and_feedback(tmp_path):
    record_runtime_event("predict", _water_quality(), {"predicted_f": 0.6}, history_dir=tmp_path)
    record_feedback({**_water_quality(), "effluent_f": 0.7}, history_dir=tmp_path)

    status = history_status(tmp_path)

    assert status["event_count"] == 1
    assert status["feedback_count"] == 1


def test_history_status_skips_malformed_jsonl_rows(tmp_path):
    events_path = tmp_path / "runtime_events.jsonl"
    valid = record_runtime_event(
        "predict",
        _water_quality(),
        {"predicted_f": 0.6},
        history_dir=tmp_path,
    )
    with events_path.open("a", encoding="utf-8") as f:
        f.write("{bad json\n")
        f.write(json.dumps({
            "record_id": "valid_after_bad",
            "kind": "predict",
            "water_quality": _water_quality(),
            "result": {"predicted_f": 0.7},
        }) + "\n")

    status = history_status(tmp_path)

    assert valid["record_id"]
    assert status["event_count"] == 2
