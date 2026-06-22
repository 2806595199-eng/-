"""测试：时序特征工程"""
import pandas as pd

from core.feature_engineer import FeatureEngineer


def test_feature_delay_steps_shift_each_variable_independently():
    df = pd.DataFrame({
        "influent_f": [10.0, 20.0, 30.0, 40.0],
        "pacl_dose": [100.0, 200.0, 300.0, 400.0],
    })
    eng = FeatureEngineer(
        lag_steps=(),
        rolling_windows=(),
        feature_delay_steps={"influent_f": 2, "pacl_dose": 1},
    )

    built = eng._build(df, feat_cols=["influent_f", "pacl_dose"])

    assert built.loc[2, "influent_f"] == 10.0
    assert built.loc[2, "pacl_dose"] == 200.0
    assert built.loc[3, "influent_f"] == 20.0
    assert built.loc[3, "pacl_dose"] == 300.0


def test_prediction_horizon_uses_largest_feature_delay():
    eng = FeatureEngineer(
        lag_steps=(),
        rolling_windows=(),
        feature_delay_steps={"influent_f": 3, "defluor_dose": 1},
    )

    assert eng.prediction_horizon_steps() == 3


def test_fit_transform_fill_values_are_fit_on_train_split_only():
    df = pd.DataFrame({
        "influent_f": [1.0, 2.0, 3.0, 999.0],
        "effluent_f": [0.5, 0.6, 0.7, 0.8],
    })
    eng = FeatureEngineer(
        lag_steps=(),
        rolling_windows=(),
        feature_delay_steps={"influent_f": 3},
    )

    eng.fit_transform(
        df,
        target_col="effluent_f",
        feat_cols=["influent_f"],
        train_idx=[0, 1],
    )

    assert eng.fill_values_["influent_f"] == 0.0
