"""XGBoost 备用模型训练 — 与 TabPFN 共用特征矩阵

训练时间控制在 <30 秒，适合定期重训。
"""

import json
import pickle
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def _score_model(model, X_test, y_test):
    X_ts = X_test.values.astype(np.float32)
    y_ts = y_test.values.astype(np.float32)
    yp = model.predict(X_ts)
    return {
        "mape": mean_absolute_percentage_error(y_ts, yp),
        "r2": r2_score(y_ts, yp),
        "rmse": float(np.sqrt(((y_ts - yp) ** 2).mean())),
    }


def _train_sklearn_backup(X_train, y_train, X_test, y_test, seed=42) -> dict:
    from sklearn.ensemble import RandomForestRegressor

    t0 = time.time()
    params = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 120,
        "max_depth": 8,
        "min_samples_leaf": 2,
    }
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.float32))

    scores = _score_model(model, X_test, y_test)
    importance = getattr(model, "feature_importances_", np.zeros(len(X_train.columns)))
    feat_imp = sorted(
        zip(X_train.columns, importance),
        key=lambda x: x[1], reverse=True,
    )[:15]

    elapsed = time.time() - t0
    print(f"[Backup] XGBoost unavailable; trained RandomForest in {elapsed:.1f}s")
    print(f"  MAPE={scores['mape']:.4f}  R2={scores['r2']:.4f}  RMSE={scores['rmse']:.4f}")

    return {
        "model": model,
        "mape": round(scores["mape"], 4),
        "r2": round(scores["r2"], 4),
        "rmse": round(scores["rmse"], 4),
        "best_params": params,
        "importance": feat_imp,
        "train_time": round(elapsed, 1),
    }


def train_xgboost(X_train, y_train, X_test, y_test,
                  n_trials=30, seed=42) -> dict:
    """网格搜索 XGBoost 最优参数

    Args:
        X_train, y_train: 训练集（已特征工程 + 标准化）
        X_test, y_test:   测试集
        n_trials: 保留参数（兼容接口）
        seed: 随机种子

    Returns:
        dict: {model, mape, r2, rmse, best_params, importance}
    """
    try:
        import xgboost as xgb
    except ImportError:
        return _train_sklearn_backup(X_train, y_train, X_test, y_test, seed=seed)

    from sklearn.model_selection import TimeSeriesSplit
    t0 = time.time()

    # 网格搜索空间：3×3×3=27，比随机搜索快且可控
    param_grid = [
        {"max_depth": d, "learning_rate": lr, "n_estimators": n}
        for d in [3, 6, 9]
        for lr in [0.05, 0.1, 0.2]
        for n in [50, 100, 200]
    ]

    # 用训练集内部 TimeSeriesSplit 选参数，避免测试集泄漏
    best_mdl = None
    best_score = float("inf")
    best_params = None
    X_tr = X_train.values.astype(np.float32)
    y_tr = y_train.values.astype(np.float32)
    tscv = TimeSeriesSplit(n_splits=3)

    for params in param_grid:
        scores = []
        for fold_train_idx, fold_val_idx in tscv.split(X_tr):
            mdl = xgb.XGBRegressor(**params, random_state=seed, verbosity=0, n_jobs=1)
            mdl.fit(X_tr[fold_train_idx], y_tr[fold_train_idx])
            yp = mdl.predict(X_tr[fold_val_idx])
            scores.append(mean_absolute_percentage_error(y_tr[fold_val_idx], yp))
        avg_mape = float(np.mean(scores))
        if avg_mape < best_score:
            best_score = avg_mape
            best_params = params

    # 用最优参数在全量训练集上训练
    best_mdl = xgb.XGBRegressor(**best_params, random_state=seed, verbosity=0, n_jobs=1)
    best_mdl.fit(X_tr, y_tr)

    # 用测试集做最终评估（仅一次）
    X_ts = X_test.values.astype(np.float32)
    y_ts = y_test.values.astype(np.float32)
    yp = best_mdl.predict(X_ts)
    mape = mean_absolute_percentage_error(y_ts, yp)
    r2 = r2_score(y_ts, yp)
    rmse = float(np.sqrt(((y_ts - yp) ** 2).mean()))

    # 特征重要性
    importance = best_mdl.feature_importances_
    feat_imp = sorted(
        zip(X_train.columns, importance),
        key=lambda x: x[1], reverse=True
    )[:15]

    elapsed = time.time() - t0
    print(f"[XGBoost] Grid search {len(param_grid)} combos in {elapsed:.1f}s")
    print(f"  MAPE={mape:.4f}  R2={r2:.4f}  RMSE={rmse:.4f}")
    print(f"  Best: {best_params}")

    return {
        "model": best_mdl,
        "mape": round(mape, 4),
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "best_params": best_params,
        "importance": feat_imp,
        "train_time": round(elapsed, 1),
    }


def save_backup_model(model, importance, output_dir="models"):
    """保存 XGBoost 模型 + 特征重要性"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 模型
    with open(out / "backup_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 特征重要性
    imp_list = [(name, float(imp)) for name, imp in importance]
    with open(out / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(imp_list, f, indent=2, ensure_ascii=False)

    print(f"[XGBoost] Saved → {out / 'backup_model.pkl'}")
