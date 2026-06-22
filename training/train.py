"""离线训练入口。

工作链条:
1. 读取并校验原始 CSV/Excel；
2. 按时间顺序切分训练段/测试段；
3. 构建 HRT 延迟、lag、rolling 等时序特征；
4. 训练主模型 TabPFN，并训练 XGBoost 作为优化器/兜底模型；
5. 导出 scaler、feature_config、train_data、metadata 等模型产物。
"""

import argparse
import os
import pickle

import pandas as pd

from core import config as cfg
from core.feature_engineer import FeatureEngineer
from training.backup_trainer import train_xgboost, save_backup_model
from training.data_loader import (
    align_target_by_delay,
    clean_data,
    load_data,
    validate_required_columns,
)
from training.model_exporter import export
from training.model_trainer import cross_validate, train_tabpfn


def _tabpfn_unavailable_result(error: Exception, y_test) -> dict:
    residual_std = float(y_test.std() or 0.12)
    return {
        "model": None,
        "r2_train": 0.0,
        "r2_test": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "residual_std": residual_std,
        "tabpfn_available": False,
        "tabpfn_error": str(error)[:500],
    }


def main(data_path=None, device=None, output_dir="models"):
    if device is None:
        device = cfg.DEVICE
    os.makedirs(output_dir, exist_ok=True)

    if data_path is None:
        data_path = "data/raw/sim_300.csv"

    print(f"[1/6] Load: {data_path}")
    df_raw = load_data(data_path)
    validate_required_columns(df_raw, for_training=True)

    has_feature_delays = any(getattr(cfg, "FEATURE_DELAY_STEPS", {}).values())
    if cfg.OUTPUT_DELAY_STEPS and has_feature_delays:
        # 当前项目采用“按变量独立延迟”的口径；如果再整体平移 target，会重复计算 HRT。
        print("  WARNING: FEATURE_DELAY_STEPS is nonzero; skipping OUTPUT_DELAY_STEPS")
    else:
        df_raw = align_target_by_delay(df_raw, cfg.OUTPUT_DELAY_STEPS)

    # 时序项目默认按时间前 80% 训练、后 20% 测试，不能随机打乱。
    n_train_raw = int(len(df_raw) * 0.8)
    df_train = clean_data(df_raw.iloc[:n_train_raw].reset_index(drop=True), for_training=True)
    df_test = clean_data(df_raw.iloc[n_train_raw:].reset_index(drop=True), for_training=True)
    n_train = len(df_train)
    if n_train == 0 or len(df_test) == 0:
        raise ValueError("training or test set is empty after cleaning; check effluent_f")

    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"      {df.shape[0]} rows ({n_train} train + {len(df_test)} test)")

    print("[2/6] Feature engineering...")
    # 特征工程会在完整时间轴上生成 lag/rolling，但 scaler 和缺失值统计只 fit 训练段。
    eng = FeatureEngineer(
        lag_steps=cfg.LAG_STEPS,
        rolling_windows=cfg.ROLLING_WINDOWS,
        min_history=cfg.MIN_HISTORY,
        feature_delay_steps=cfg.FEATURE_DELAY_STEPS,
    )
    df_feat = eng.fit_transform(
        df,
        cfg.TARGET_COL,
        feat_cols=cfg.TS_FEAT_COLS,
        train_idx=list(range(n_train)),
    )
    print(f"      {df_feat.shape[1] - 1} features")

    print("[3/6] TabPFN training...")
    # TabPFN 是正式预测主模型；训练数据会同时保存，供服务启动时重新 fit。
    X_tr = df_feat.iloc[:n_train].drop(columns=[cfg.TARGET_COL])
    X_te = df_feat.iloc[n_train:].drop(columns=[cfg.TARGET_COL])
    y_tr = df_feat.iloc[:n_train][cfg.TARGET_COL]
    y_te = df_feat.iloc[n_train:][cfg.TARGET_COL]

    try:
        res = train_tabpfn(X_tr, y_tr, X_te, y_te, device=device, seed=cfg.RANDOM_SEED)
        res["tabpfn_available"] = True
    except Exception as exc:
        print(f"  WARNING: TabPFN unavailable, continuing with backup model: {exc}")
        res = _tabpfn_unavailable_result(exc, y_te)

    print("[4/6] CV (train only)...")
    if res.get("tabpfn_available") and len(y_tr) >= 30:
        cv = cross_validate(X_tr, y_tr, n_splits=5, device=device)
    elif not res.get("tabpfn_available"):
        print("  WARNING: TabPFN unavailable, skipping CV")
        cv = {"cv_r2_mean": 0, "cv_r2_std": 0}
    else:
        print("  WARNING: train rows < 30, skipping CV")
        cv = {"cv_r2_mean": 0, "cv_r2_std": 0}

    print("[5/6] XGBoost backup...")
    # XGBoost 不替代 TabPFN，主要用于加药优化时快速扫大量候选组合。
    base_cols = cfg.XGB_BASE_COLS
    xgb_eng = FeatureEngineer(
        lag_steps=(),
        rolling_windows=(),
        min_history=cfg.MIN_HISTORY,
        feature_delay_steps=cfg.FEATURE_DELAY_STEPS,
    )
    df_xgb = xgb_eng._build(df, feat_cols=cfg.TS_FEAT_COLS)
    # 备用 XGBoost 路径同样只用训练段填充值，避免测试段统计量泄漏到评估。
    xgb_fill_values = df_xgb.iloc[:len(X_tr)].median().fillna(0.0).to_dict()
    df_xgb = df_xgb.fillna(xgb_fill_values).fillna(0.0)
    available = [c for c in base_cols if c in df_xgb.columns]
    xgb_train = df_xgb.iloc[:len(X_tr)][available]
    xgb_test = df_xgb.iloc[len(X_tr):len(X_tr) + len(X_te)][available]
    assert len(xgb_train) == len(y_tr), f"XGB train size mismatch: {len(xgb_train)} vs {len(y_tr)}"
    assert len(xgb_test) == len(y_te), f"XGB test size mismatch: {len(xgb_test)} vs {len(y_te)}"
    xgb_res = train_xgboost(xgb_train, y_tr, xgb_test, y_te, seed=cfg.RANDOM_SEED)
    save_backup_model(xgb_res["model"], xgb_res["importance"], output_dir)

    print("[6/6] Export...")
    # train_data.pkl 保存的是标准化后的 TabPFN 训练/测试数组，推理服务加载 active 版本时会用它重建主模型。
    with open(os.path.join(output_dir, "train_data.pkl"), "wb") as f:
        pickle.dump({
            "X_train": X_tr.values.astype("float32"),
            "y_train": y_tr.values.astype("float32"),
            "X_test": X_te.values.astype("float32"),
            "y_test": y_te.values.astype("float32"),
        }, f)

    meta = {k: v for k, v in {**res, **cv}.items()
            if k not in ("model", "y_pred", "y_test")}
    meta.update({
        "model_type": "TabPFNRegressor" if res.get("tabpfn_available")
        else xgb_res.get("best_params", {}).get("model_type", "backup_model"),
        "serving_model_mode": "tabpfn_with_backup" if res.get("tabpfn_available")
        else "backup_only",
        "model_input_cols": cfg.MODEL_INPUT_COLS,
        "target_col": cfg.TARGET_COL,
        "xgb_feature_cols": available,
        "influent_flow_mean": float(df_train["influent_flow"].mean()),
        "influent_ph_mean": float(df_train["influent_ph"].mean()),
        "conductivity_mean": float(df_train["conductivity"].mean()),
        "influent_f_mean": float(df_train["influent_f"].mean()),
        "xgb_r2": xgb_res["r2"],
        "xgb_rmse": xgb_res["rmse"],
        "xgb_mape": xgb_res["mape"],
        "xgb_model_type": xgb_res.get("best_params", {}).get("model_type", "XGBRegressor"),
    })
    export(output_dir, eng.scaler, eng, meta)

    print(f"\n{'=' * 50}")
    if res.get("tabpfn_available"):
        tab_mape = round(res["rmse"] / max(y_te.mean(), 0.01), 4)
        print(f"  TabPFN  R2={res['r2_test']:.4f}  RMSE={res['rmse']:.4f}  MAPE~{tab_mape:.4f}")
    else:
        print("  TabPFN  unavailable; exported backup-only artifacts")
    print(f"  XGBoost R2={xgb_res['r2']:.4f}  RMSE={xgb_res['rmse']:.4f}  MAPE={xgb_res['mape']:.4f}")
    if res.get("tabpfn_available") and len(y_tr) >= 30:
        print(f"  CV R2={cv['cv_r2_mean']:.4f}+-{cv['cv_r2_std']:.4f}")
    print(f"{'=' * 50}")
    print("Next: python serve.py")

    return {
        **meta,
        "xgb_r2": xgb_res["r2"],
        "xgb_rmse": xgb_res["rmse"],
        "xgb_mape": xgb_res["mape"],
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--device", type=str, default=cfg.DEVICE)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()
    main(args.data, args.device, args.output_dir)
