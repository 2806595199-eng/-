import json
from pathlib import Path

import pandas as pd

from core import config as cfg
from training.data_quality import ingest_source_file, validate_dataset


def _valid_rows():
    row = {
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
    for i in range(3):
        item = dict(row)
        item["timestamp"] = f"2026-05-27 10:0{i}:00"
        item["effluent_f"] = 0.7 + i * 0.01
        rows.append(item)
    return rows


def test_validate_dataset_catches_range_violations():
    df = pd.DataFrame(_valid_rows())
    df.loc[0, "influent_ph"] = 16.0

    report = validate_dataset(df, for_training=True)

    assert report["passed"] is False
    assert any(
        issue["type"] == "range" and issue["field"] == "influent_ph"
        for issue in report["issues"]
    )


def test_validate_dataset_accepts_good_training_frame():
    report = validate_dataset(pd.DataFrame(_valid_rows()), for_training=True)

    assert report["passed"] is True
    assert report["row_count"] == 3
    assert report["required_columns_present"] is True


def test_ingest_source_file_archives_original_and_writes_report(tmp_path):
    source = tmp_path / "site_batch.csv"
    pd.DataFrame(_valid_rows()).to_csv(source, index=False, encoding="utf-8-sig")

    result = ingest_source_file(
        source,
        raw_dir=tmp_path / "raw",
        prepared_dir=tmp_path / "prepared",
        report_dir=tmp_path / "reports",
    )

    assert Path(result["raw_path"]).exists()
    assert Path(result["prepared_path"]).exists()
    report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
    assert report["passed"] is True
    prepared = pd.read_csv(result["prepared_path"], encoding="utf-8-sig")
    for col in cfg.MODEL_INPUT_COLS + [cfg.TARGET_COL]:
        assert col in prepared.columns
