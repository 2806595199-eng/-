"""第一阶段模型更新闭环。

用途:
- 接收现场新增数据或在线反馈整理出的训练 CSV；
- 先做数据质量校验，再训练候选版本；
- 只有在 auto_publish=True 且发布门槛通过时，才把候选版本切为 active。
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from core import config as cfg
from training.data_quality import ingest_source_file
from training.model_registry import (create_version_dir, publish_active_model,
                                     read_active_model)
from training.train import main as train_main


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _read_metadata(version_dir: Path) -> Dict[str, Any]:
    path = version_dir / "model_metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_decision(metrics: Dict[str, Any],
                      active_metrics: Optional[Dict[str, Any]] = None,
                      min_r2: Optional[float] = None,
                      require_better_than_active: bool = False) -> Dict[str, Any]:
    """判断候选模型是否允许发布。

    这里不只看“训练成功”，还会检查最低 R2、是否劣于当前 active 模型等发布门槛。
    """
    reasons = []
    r2 = metrics.get("r2_test")
    rmse = metrics.get("rmse")
    if min_r2 is not None:
        if r2 is None or float(r2) < min_r2:
            reasons.append(f"r2_test below min_r2: {r2} < {min_r2}")

    if require_better_than_active and active_metrics:
        old_r2 = active_metrics.get("r2_test")
        old_rmse = active_metrics.get("rmse")
        if old_r2 is not None and r2 is not None and float(r2) < float(old_r2):
            reasons.append(f"r2_test worse than active: {r2} < {old_r2}")
        if old_rmse is not None and rmse is not None and float(rmse) > float(old_rmse):
            reasons.append(f"rmse worse than active: {rmse} > {old_rmse}")

    return {
        "passed": not reasons,
        "reasons": reasons,
        "candidate_metrics": metrics,
        "active_metrics": active_metrics or {},
    }


def update_model_from_file(source_path: Path | str,
                           models_root: Path | str = "models",
                           raw_dir: Path | str = "data/raw/site",
                           prepared_dir: Path | str = "data/prepared",
                           report_dir: Path | str = "data/reports",
                           device: Optional[str] = None,
                           version_id: Optional[str] = None,
                           auto_publish: bool = False,
                           min_r2: Optional[float] = None,
                           require_better_than_active: bool = False) -> Dict[str, Any]:
    """从一个数据文件训练候选模型，并按规则决定是否发布。

    人工 review 时重点看返回的 update_report.json：里面有数据质量报告、指标、发布决策和 active 模型信息。
    """
    if device is None:
        device = cfg.DEVICE

    # ingest 会复制原始数据、生成 canonical CSV、输出质量报告；质量不通过则不训练。
    ingest = ingest_source_file(
        source_path,
        raw_dir=raw_dir,
        prepared_dir=prepared_dir,
        report_dir=report_dir,
        for_training=True,
    )
    if not ingest["quality_report"]["passed"]:
        return {
            "status": "failed_quality_check",
            "published": False,
            "ingest": ingest,
            "quality_report_path": ingest["report_path"],
        }

    # 每次更新都写入独立版本目录，避免覆盖当前线上模型。
    version_dir = create_version_dir(models_root, version_id=version_id)
    shutil.copy2(ingest["report_path"], version_dir / "data_quality_report.json")

    train_result = train_main(
        data_path=ingest["prepared_path"],
        device=device,
        output_dir=str(version_dir),
    )
    metrics = {**_read_metadata(version_dir), **(train_result or {})}
    active_before = read_active_model(models_root)
    active_metrics = (active_before or {}).get("metrics", {})
    decision = _publish_decision(
        metrics,
        active_metrics=active_metrics,
        min_r2=min_r2,
        require_better_than_active=require_better_than_active,
    )
    # 默认只训练候选版本；只有显式 publish 且指标门槛通过，才修改 active_model.json。
    publish_allowed = auto_publish and decision["passed"]
    active = None
    if publish_allowed:
        active = publish_active_model(
            models_root,
            version_dir,
            metrics=metrics,
            note="auto-published by model_update",
        )

    update_report = {
        "status": "trained",
        "created_at": datetime.now().isoformat(),
        "source_path": str(source_path),
        "prepared_path": ingest["prepared_path"],
        "quality_report_path": str(version_dir / "data_quality_report.json"),
        "version_dir": str(version_dir),
        "metrics": metrics,
        "published": active is not None,
        "publish_decision": decision,
        "active_model": active,
    }
    (version_dir / "update_report.json").write_text(
        json.dumps(update_report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return update_report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="CSV/XLSX site data file")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--raw-dir", default="data/raw/site")
    parser.add_argument("--prepared-dir", default="data/prepared")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--device", default=cfg.DEVICE)
    parser.add_argument("--version-id", default=None)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--min-r2", type=float, default=None)
    parser.add_argument("--require-better-than-active", action="store_true")
    args = parser.parse_args(argv)

    result = update_model_from_file(
        args.source,
        models_root=args.models_root,
        raw_dir=args.raw_dir,
        prepared_dir=args.prepared_dir,
        report_dir=args.report_dir,
        device=args.device,
        version_id=args.version_id,
        auto_publish=args.publish,
        min_r2=args.min_r2,
        require_better_than_active=args.require_better_than_active,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
