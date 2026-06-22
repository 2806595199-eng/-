"""模型导出 / 加载"""

import pickle
import json
from pathlib import Path
from datetime import datetime


def export(output_dir, scaler, feature_engineer, metadata, model=None):
    """导出 scaler + 特征配置 + 元数据"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    feature_engineer.save_config(str(out / "feature_config.json"))
    meta = {"export_time": datetime.now().isoformat(),
            "model_type": "TabPFNRegressor", **metadata}
    with open(out / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Model exported → {out}/")
    for x in sorted(out.iterdir()):
        print(f"  {x.name}")


def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
