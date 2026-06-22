"""基于在线反馈的定期更新入口。

定时任务或 API 会调用这里：先把反馈日志整理成训练 CSV，达到最小样本量后再训练候选模型。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core import config as cfg
from serving.online_history import build_feedback_training_file
from training.model_update import update_model_from_file


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def run_scheduled_update(history_dir: Path | str = "data/online",
                         models_root: Path | str = "models",
                         min_rows: int = 100,
                         auto_publish: bool = False,
                         min_r2: Optional[float] = None,
                         device: Optional[str] = None) -> Dict[str, Any]:
    """用在线反馈触发一次候选模型训练。

    min_rows 是保护阈值：反馈样本太少时不更新，避免模型被少量偶然数据带偏。
    """
    if device is None:
        device = cfg.DEVICE

    history = Path(history_dir)
    prepared_dir = history / "prepared"
    training_path = prepared_dir / "feedback_training.csv"
    # 把 runtime_events + effluent_feedback 拼成标准训练表。
    built = build_feedback_training_file(history, output_path=training_path)
    if built["row_count"] < min_rows:
        return {
            "status": "skipped",
            "reason": "not_enough_feedback_rows",
            "row_count": built["row_count"],
            "min_rows": min_rows,
            "training_path": built["training_path"],
        }

    # 在线更新默认要求候选模型不差于当前 active，防止自动发布退化版本。
    result = update_model_from_file(
        built["training_path"],
        models_root=models_root,
        raw_dir=history / "raw",
        prepared_dir=history / "canonical",
        report_dir=history / "reports",
        device=device,
        auto_publish=auto_publish,
        min_r2=min_r2,
        require_better_than_active=True,
    )
    result.setdefault("status", "trained")
    result["scheduled_at"] = datetime.now().isoformat()
    result["feedback_training_path"] = built["training_path"]
    result["feedback_row_count"] = built["row_count"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", default="data/online")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--min-r2", type=float, default=None)
    parser.add_argument("--device", default=cfg.DEVICE)
    args = parser.parse_args(argv)

    result = run_scheduled_update(
        history_dir=args.history_dir,
        models_root=args.models_root,
        min_rows=args.min_rows,
        auto_publish=args.publish,
        min_r2=args.min_r2,
        device=args.device,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
