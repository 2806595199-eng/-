"""TabPFN 模型训练与评估"""

import numpy as np
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
from core.tabpfn_runtime import configure_tabpfn_cache, resolve_tabpfn_device

configure_tabpfn_cache()
from tabpfn import TabPFNRegressor


def train_tabpfn(X_train, y_train, X_test, y_test, device=None, seed=42):
    """TabPFN 回归：前向传播推理"""
    print(f"[TabPFN] train: {X_train.shape}, test: {X_test.shape}")
    resolved_device = resolve_tabpfn_device(device)
    print(f"[TabPFN] device: {resolved_device}")

    t0 = time.time()
    model = TabPFNRegressor(
        device=resolved_device, ignore_pretraining_limits=True, random_state=seed)
    model.fit(X_train.values.astype(np.float32), y_train.values.astype(np.float32))
    fit_t = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test.values.astype(np.float32))
    pred_t = time.time() - t0

    y_train_pred = model.predict(X_train.values.astype(np.float32))
    r2_tr = r2_score(y_train, y_train_pred)
    r2_te = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  R2(train)={r2_tr:.4f}  R2(test)={r2_te:.4f}  "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}")
    print(f"  fit={fit_t:.1f}s  predict={pred_t:.1f}s")

    # 计算验证集残差的标准差，供推理时构建置信区间
    # 动态计算比硬编码更准确反映当前模型的预测波动
    residual_std = float(np.std(y_test.values - y_pred))
    return {
        "model": model, "r2_train": round(r2_tr, 4),
        "r2_test": round(r2_te, 4), "rmse": round(rmse, 4),
        "mae": round(mae, 4), "residual_std": round(residual_std, 6),
        "y_pred": y_pred, "y_test": y_test.values,
    }


def cross_validate(X, y, n_splits=5, device=None, seed=42):
    """5 折交叉验证"""
    # 时序数据使用 TimeSeriesSplit，保持时间顺序不被破坏
    tscv = TimeSeriesSplit(n_splits=n_splits)
    r2_list = []
    resolved_device = resolve_tabpfn_device(device)
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model = TabPFNRegressor(
            device=resolved_device, ignore_pretraining_limits=True, random_state=seed)
        model.fit(X.iloc[tr].values.astype(np.float32),
                  y.iloc[tr].values.astype(np.float32))
        yp = model.predict(X.iloc[te].values.astype(np.float32))
        r2_list.append(r2_score(y.iloc[te], yp))
        print(f"  Fold {fold}: R2={r2_list[-1]:.4f}")
    print(f"  CV: R2={np.mean(r2_list):.4f} +- {np.std(r2_list):.4f}")
    return {"cv_r2_mean": round(np.mean(r2_list), 4),
            "cv_r2_std": round(np.std(r2_list), 4)}
