import numpy as np
import pandas as pd

from core import config as cfg
from training import gen_sim_data


def test_effluent_formula_uses_delayed_process_values():
    rows = []
    for i in range(10):
        rows.append({
            "influent_flow": 100.0,
            "influent_ph": 7.2,
            "conductivity": 6500.0,
            "influent_f": 20.0,
            "pacl_dose": 100.0 + i * 100.0,
            "defluor_dose": 0.5 + i * 0.2,
            "pacl_tank_ph": 7.0,
            "defluor_tank_ph": 6.0,
            "recycle_flow": 4.0,
            "waste_flow": 15.0,
            "pam_dose": 0.5,
        })
    df = pd.DataFrame(rows)
    delays = {"influent_f": 2, "pacl_dose": 2, "defluor_dose": 1,
              "defluor_tank_ph": 1}

    effluent = gen_sim_data.compute_effluent_with_delays(
        df,
        rng=np.random.default_rng(1),
        feature_delay_steps=delays,
        noise_scale=0.0,
    )

    expected = gen_sim_data.effluent_formula(
        influent_f=df.loc[3, "influent_f"],
        pacl=df.loc[3, "pacl_dose"],
        defluor=df.loc[4, "defluor_dose"],
        defluor_ph=df.loc[4, "defluor_tank_ph"],
    )
    assert effluent.loc[5] == expected


def test_generate_returns_required_training_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    df = gen_sim_data.generate(n=30, seed=42)

    for column in cfg.MODEL_INPUT_COLS + [cfg.TARGET_COL]:
        assert column in df.columns
    assert (tmp_path / "data" / "raw" / "sim_30.csv").exists()
