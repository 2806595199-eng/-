"""测试夹具"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_water_data():
    """生成 50 条模拟水质时序数据"""
    rng = np.random.default_rng(42)
    records = []
    for i in range(50):
        records.append({
            "timestamp": f"2025-12-10 {(i % 24):02d}:00:00",
            "influent_f": round(18.0 + rng.normal(0, 2), 2),
            "conductivity": round(6500 + rng.normal(0, 200), 0),
            "orp": round(200 + rng.normal(0, 10), 1),
            "turbidity": round(2.5 + rng.normal(0, 0.3), 2),
            "cod": round(450 + rng.normal(0, 50), 0),
            "ph": round(7.0 + rng.normal(0, 0.2), 1),
            "pacl_dose": round(400 + rng.normal(0, 100), 1),
            "defluor_dose": round(0.5 + rng.normal(0, 0.1), 4),
            "magnetic_dose": 0.15,
            "pam_dose": 0.5,
            "mud_recycle_mi": 0.356,
            "ph_removal": 6.0,
            "ph_floc": 6.0,
            "effluent_f": round(0.5 + rng.normal(0, 0.15), 4),
        })
    return pd.DataFrame(records)
