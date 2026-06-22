"""Persist online prediction/recommendation events and effluent feedback."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd

from core import config as cfg


EVENTS_FILE = "runtime_events.jsonl"
FEEDBACK_FILE = "effluent_feedback.jsonl"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    """容错读取在线日志，避免单行损坏拖垮状态查询或模型更新。"""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _clean_water_quality(water_quality: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v for k, v in dict(water_quality).items()
        if k in set(cfg.MODEL_INPUT_COLS) | {"timestamp"}
    }


def record_runtime_event(kind: str,
                         water_quality: Dict[str, Any],
                         result: Dict[str, Any],
                         history_dir: Path | str = "data/online") -> Dict[str, Any]:
    """记录一次在线预测或加药推荐，并返回 record_id。

    后续人工化验出水氟回来后，可以用这个 record_id 把“当时输入/推荐方案”和“实际出水”拼成训练样本。
    """
    record = {
        "record_id": uuid4().hex,
        "created_at": _now_iso(),
        "kind": kind,
        "water_quality": _clean_water_quality(water_quality),
        "result": result,
    }
    _append_jsonl(Path(history_dir) / EVENTS_FILE, record)
    return record


def record_feedback(feedback: Dict[str, Any],
                    history_dir: Path | str = "data/online") -> Dict[str, Any]:
    """记录一次人工/仪表反馈的实际出水氟。

    推荐用 record_id 关联历史预测/推荐记录；如果没有 record_id，也可以直接上传完整输入字段加 effluent_f。
    支持 executed_pacl_dose / executed_defluor_dose 记录实际执行加药量，
    以区分"推荐值"与"实际值"，避免运维未按推荐执行时训练错误关系。
    """
    payload = dict(feedback)
    payload.setdefault("feedback_id", uuid4().hex)
    payload.setdefault("created_at", _now_iso())
    if "effluent_f" not in payload:
        raise ValueError("feedback must include effluent_f")
    # 保留推荐与实际的区分：训练时优先用 executed_dose，回退到推荐值
    if "executed_pacl_dose" not in payload and "pacl_dose" in payload:
        payload["executed_pacl_dose"] = payload["pacl_dose"]
    if "executed_defluor_dose" not in payload and "defluor_dose" in payload:
        payload["executed_defluor_dose"] = payload["defluor_dose"]
    _append_jsonl(Path(history_dir) / FEEDBACK_FILE, payload)
    return payload


def _event_lookup(history_dir: Path | str) -> Dict[str, Dict[str, Any]]:
    events = _read_jsonl(Path(history_dir) / EVENTS_FILE)
    return {row["record_id"]: row for row in events if "record_id" in row}


def _event_timestamp(event: Dict[str, Any]):
    water_quality = event.get("water_quality", {})
    timestamp = water_quality.get("timestamp") or event.get("created_at")
    return pd.to_datetime(timestamp, errors="coerce")


def _match_event_by_hrt(feedback: Dict[str, Any],
                        events: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """反馈没有 record_id 时，按时间戳做尽力匹配。

    优先使用 record_id 精确匹配；这里用最大特征延迟粗略代表 HRT，
    并允许一个采样间隔的容差。
    """
    feedback_ts = pd.to_datetime(feedback.get("timestamp"), errors="coerce")
    if pd.isna(feedback_ts):
        return None

    delay_min = max(cfg.FEATURE_DELAY_STEPS.values(), default=0) * cfg.MODEL_SAMPLE_INTERVAL_MIN
    target_ts = feedback_ts - pd.Timedelta(minutes=delay_min)
    tolerance = pd.Timedelta(minutes=max(cfg.MODEL_SAMPLE_INTERVAL_MIN, 1))

    best_event = None
    best_delta = None
    for event in events.values():
        event_ts = _event_timestamp(event)
        if pd.isna(event_ts):
            continue
        delta = abs(event_ts - target_ts)
        if best_delta is None or delta < best_delta:
            best_event = event
            best_delta = delta

    if best_event is not None and best_delta <= tolerance:
        return best_event
    return None


def _training_row_from_feedback(feedback: Dict[str, Any],
                                events: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """把反馈记录转换成可训练的一行数据。

    拼不齐 cfg.MODEL_INPUT_COLS 时直接丢弃，避免用半缺失样本污染在线更新训练集。
    """
    row = {k: feedback.get(k) for k in cfg.MODEL_INPUT_COLS if k in feedback}
    if len(row) < len(cfg.MODEL_INPUT_COLS):
        event = events.get(feedback.get("record_id"))
        if event is None:
            # 没有 record_id 时才按 HRT 时间反推，降低同一时刻拼错训练样本的风险。
            event = _match_event_by_hrt(feedback, events)
        if event:
            row.update(event.get("water_quality", {}))

    if not all(k in row and row[k] is not None for k in cfg.MODEL_INPUT_COLS):
        return None

    row["effluent_f"] = feedback["effluent_f"]
    if feedback.get("timestamp") is not None:
        row["timestamp"] = feedback["timestamp"]
    return row


def build_feedback_training_file(history_dir: Path | str = "data/online",
                                 output_path: Path | str | None = None) -> Dict[str, Any]:
    """把在线反馈日志整理成标准训练 CSV。

    这是在线闭环更新的入口：runtime_events.jsonl 提供当时输入，effluent_feedback.jsonl 提供事后真实出水。
    """
    history = Path(history_dir)
    feedback_rows = _read_jsonl(history / FEEDBACK_FILE)
    events = _event_lookup(history)
    rows = [
        row for row in (
            _training_row_from_feedback(feedback, events)
            for feedback in feedback_rows
        )
        if row is not None
    ]

    if output_path is None:
        output_path = history / "feedback_training.csv"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = ["timestamp"] + cfg.MODEL_INPUT_COLS + [cfg.TARGET_COL]
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return {
        "training_path": str(output),
        "row_count": int(len(df)),
        "feedback_count": len(feedback_rows),
    }


def history_status(history_dir: Path | str = "data/online") -> Dict[str, Any]:
    history = Path(history_dir)
    event_count = len(_read_jsonl(history / EVENTS_FILE))
    feedback_count = len(_read_jsonl(history / FEEDBACK_FILE))
    trainable = build_feedback_training_file(
        history,
        output_path=history / "_status_feedback_training.csv",
    )
    status_path = history / "_status_feedback_training.csv"
    if status_path.exists():
        status_path.unlink()
    return {
        "history_dir": str(history),
        "event_count": event_count,
        "feedback_count": feedback_count,
        "trainable_feedback_rows": trainable["row_count"],
    }
