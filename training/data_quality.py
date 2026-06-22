"""Data ingestion and quality checks for site operation data."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from core import config as cfg
from training.data_loader import canonicalize_columns, load_data


# 基于中试数据实际分布设置（~3倍观测范围），用于检测传感器故障异常值
NUMERIC_RANGES = {
    "influent_flow": (5.0, 200.0),       # 中试 16-52 m3/h
    "influent_ph": (4.0, 10.0),          # 中试 5.8-8.4
    "conductivity": (300.0, 10000.0),    # 中试 1211-1924 μS/cm
    "influent_f": (1.0, 50.0),           # 中试 4.0-25.2 mg/L
    "effluent_f": (0.0, 5.0),            # 中试 0.6-2.87 mg/L
    "pacl_dose": (0.0, 1500.0),          # 中试 70-240 mg/L
    "defluor_dose": (0.0, 10.0),         # 中试 0.26-4.63 mL/L
    "pacl_tank_ph": (4.0, 12.0),         # 中试 6.3-11.5
    "defluor_tank_ph": (4.0, 9.0),       # 中试 5.9-7.0
    "recycle_flow": (0.0, 2.0),          # 中试 0.26-0.40 m3/h
    "waste_flow": (0.0, 10.0),           # 中试 2.30-2.55 m3/h
    "pam_dose": (0.0, 10.0),             # 中试 3.77-4.0 mg/L
}


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "site_data"


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _issue(severity: str, issue_type: str, message: str,
           field: Optional[str] = None, **extra) -> Dict[str, Any]:
    item = {
        "severity": severity,
        "type": issue_type,
        "message": message,
    }
    if field is not None:
        item["field"] = field
    item.update(extra)
    return item


def _required_columns(for_training: bool) -> list[str]:
    required = list(cfg.MODEL_INPUT_COLS)
    if for_training:
        required.append(cfg.TARGET_COL)
    return required


def validate_dataset(df: pd.DataFrame, for_training: bool = True,
                     timestamp_col: str = "timestamp",
                     max_missing_fraction: float = 0.2) -> Dict[str, Any]:
    """Validate canonical field presence, time order, missingness and ranges."""
    issues = []
    try:
        checked = canonicalize_columns(df)
    except Exception as exc:
        return {
            "passed": False,
            "row_count": int(len(df)),
            "required_columns_present": False,
            "issues": [_issue("error", "schema", str(exc))],
        }

    required = _required_columns(for_training)
    missing = [col for col in required if col not in checked.columns]
    if missing:
        issues.append(_issue(
            "error", "missing_columns",
            f"Missing required columns: {missing}",
            missing_columns=missing,
        ))

    if len(checked) == 0:
        issues.append(_issue("error", "empty_dataset", "Dataset has no rows"))

    if timestamp_col in checked.columns:
        ts = pd.to_datetime(checked[timestamp_col], errors="coerce")
        invalid_ts = int(ts.isna().sum())
        if invalid_ts:
            issues.append(_issue(
                "error", "timestamp_parse",
                f"{invalid_ts} timestamps cannot be parsed",
                field=timestamp_col,
                count=invalid_ts,
            ))
        duplicate_ts = int(ts.duplicated().sum())
        if duplicate_ts:
            issues.append(_issue(
                "error", "duplicate_timestamp",
                f"{duplicate_ts} duplicate timestamps found",
                field=timestamp_col,
                count=duplicate_ts,
            ))
        if invalid_ts == 0 and not ts.is_monotonic_increasing:
            issues.append(_issue(
                "error", "timestamp_order",
                "Timestamps must be monotonically increasing",
                field=timestamp_col,
            ))

    columns_to_check = [c for c in required if c in checked.columns]
    optional_numeric = [c for c in cfg.OPTIONAL_INPUT_COLS if c in checked.columns]
    for col in columns_to_check + optional_numeric:
        series = pd.to_numeric(checked[col], errors="coerce")
        missing_count = int(series.isna().sum())
        if missing_count:
            fraction = missing_count / max(len(series), 1)
            severity = "error" if fraction > max_missing_fraction else "warning"
            issues.append(_issue(
                severity, "missing_values",
                f"{missing_count} missing values in {col}",
                field=col,
                count=missing_count,
                fraction=round(fraction, 4),
            ))

        if col in NUMERIC_RANGES:
            lo, hi = NUMERIC_RANGES[col]
            out_of_range = series.notna() & ((series < lo) | (series > hi))
            count = int(out_of_range.sum())
            if count:
                issues.append(_issue(
                    "error", "range",
                    f"{count} values outside [{lo}, {hi}]",
                    field=col,
                    count=count,
                    min_allowed=lo,
                    max_allowed=hi,
                ))

    error_count = sum(1 for item in issues if item["severity"] == "error")
    return {
        "passed": error_count == 0,
        "row_count": int(len(checked)),
        "column_count": int(len(checked.columns)),
        "required_columns_present": not missing,
        "error_count": error_count,
        "warning_count": sum(1 for item in issues if item["severity"] == "warning"),
        "issues": issues,
    }


def write_quality_report(report: Dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return out


def ingest_source_file(source_path: Path | str,
                       raw_dir: Path | str = "data/raw/site",
                       prepared_dir: Path | str = "data/prepared",
                       report_dir: Path | str = "data/reports",
                       for_training: bool = True) -> Dict[str, Any]:
    """Archive a site data file, canonicalize it and create a quality report."""
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(src)

    batch_id = _now_id()
    raw_root = Path(raw_dir)
    prepared_root = Path(prepared_dir)
    report_root = Path(report_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    raw_path = raw_root / f"{batch_id}_{_safe_filename(src.stem)}{src.suffix.lower()}"
    shutil.copy2(src, raw_path)

    df = load_data(str(raw_path))
    report = validate_dataset(df, for_training=for_training)
    report_path = report_root / f"{batch_id}_quality_report.json"
    write_quality_report(report, report_path)

    prepared_path = None
    if report["passed"]:
        prepared_path = prepared_root / f"{batch_id}_canonical.csv"
        df.to_csv(prepared_path, index=False, encoding="utf-8-sig")

    return {
        "batch_id": batch_id,
        "source_path": str(src),
        "raw_path": str(raw_path),
        "prepared_path": str(prepared_path) if prepared_path is not None else None,
        "report_path": str(report_path),
        "quality_report": report,
    }
