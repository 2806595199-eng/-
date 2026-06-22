import builtins

import numpy as np
import pandas as pd

from training.backup_trainer import train_xgboost


def test_train_xgboost_falls_back_to_sklearn_when_xgboost_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("forced missing xgboost")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    X_train = pd.DataFrame({
        "influent_flow": np.linspace(90, 110, 20),
        "influent_f": np.linspace(10, 20, 20),
    })
    y_train = pd.Series(np.linspace(0.4, 0.9, 20))
    X_test = pd.DataFrame({
        "influent_flow": np.linspace(92, 108, 6),
        "influent_f": np.linspace(11, 19, 6),
    })
    y_test = pd.Series(np.linspace(0.45, 0.85, 6))

    result = train_xgboost(X_train, y_train, X_test, y_test, seed=42)

    assert result["model"] is not None
    assert result["best_params"]["model_type"] == "RandomForestRegressor"
    assert len(result["model"].predict(X_test.values.astype(np.float32))) == len(y_test)
