"""Generate simulated process data aligned with configured HRT delays."""

import argparse
import os

import numpy as np
import pandas as pd

from core import config as cfg


def _delayed(series: pd.Series, delay_steps: int) -> pd.Series:
    if delay_steps <= 0:
        return series
    return series.shift(delay_steps).fillna(series.iloc[0])


def effluent_formula(influent_f: float, pacl: float, defluor: float,
                     defluor_ph: float) -> float:
    pacl_eff = (1 - np.exp(-pacl / 600.0)) * 0.92
    after_pacl = influent_f * (1 - pacl_eff)
    defluor_eff = (1 - np.exp(-defluor / 1.0)) * 0.93
    after_defluor = after_pacl * (1 - defluor_eff)
    ph_dev = abs(defluor_ph - 6.0)
    return max(0.01, after_defluor * (1 + ph_dev * 0.25))


def compute_effluent_with_delays(df: pd.DataFrame, rng=None,
                                 feature_delay_steps=None,
                                 noise_scale: float = 0.04) -> pd.Series:
    if rng is None:
        rng = np.random.default_rng(42)
    if feature_delay_steps is None:
        feature_delay_steps = cfg.FEATURE_DELAY_STEPS

    delayed_influent = _delayed(df["influent_f"],
                                feature_delay_steps.get("influent_f", 0))
    delayed_pacl = _delayed(df["pacl_dose"],
                            feature_delay_steps.get("pacl_dose", 0))
    delayed_defluor = _delayed(df["defluor_dose"],
                               feature_delay_steps.get("defluor_dose", 0))
    delayed_defluor_ph = _delayed(
        df["defluor_tank_ph"],
        feature_delay_steps.get("defluor_tank_ph", 0),
    )

    values = []
    for i in range(len(df)):
        base = effluent_formula(
            influent_f=float(delayed_influent.iloc[i]),
            pacl=float(delayed_pacl.iloc[i]),
            defluor=float(delayed_defluor.iloc[i]),
            defluor_ph=float(delayed_defluor_ph.iloc[i]),
        )
        if noise_scale <= 0:
            noise = 0.0
        else:
            noise = rng.normal(0, max(0.02, base * noise_scale))
        values.append(max(0.01, base + noise))
    return pd.Series(values, index=df.index, name=cfg.TARGET_COL)


def generate(n=200, seed=42):
    rng = np.random.default_rng(seed)

    def drift(base, amp, period=100, i=0):
        trend = base + amp * np.sin(2 * np.pi * i / period)
        return float(trend + rng.normal(0, amp * 0.15))

    records = []
    for i in range(n):
        flow = max(5.0, drift(100, 30, 120, i))
        influent_ph = drift(7.2, 0.5, 60, i)
        cond = drift(6500, 500, 90, i)
        influent_f = max(4.0, drift(18.0, 6.0, 80, i))

        pacl = max(100, (300 + influent_f * 45) * rng.uniform(0.6, 1.4))
        defluor = max(0.1, (0.3 + influent_f * 0.10) * rng.uniform(0.6, 1.4))
        pacl_ph = drift(7.0, 0.3, 50, i)
        defluor_ph = drift(6.0, 0.15, 40, i)
        recycle = flow * rng.uniform(0.02, 0.06)
        waste = flow * 0.15 * rng.uniform(0.8, 1.2)
        pam = 0.5 * rng.uniform(0.9, 1.1)

        records.append({
            "timestamp": f"2025-12-{10 + i // 24:02d} {(i % 24):02d}:00:00",
            "influent_flow": round(flow, 1),
            "influent_ph": round(influent_ph, 2),
            "conductivity": round(cond, 0),
            "influent_f": round(influent_f, 2),
            "pacl_dose": round(pacl, 1),
            "defluor_dose": round(defluor, 4),
            "pacl_tank_ph": round(pacl_ph, 2),
            "defluor_tank_ph": round(defluor_ph, 2),
            "recycle_flow": round(recycle, 1),
            "waste_flow": round(waste, 1),
            "pam_dose": round(pam, 3),
        })

    df = pd.DataFrame(records)
    df[cfg.TARGET_COL] = compute_effluent_with_delays(df, rng=rng).round(4)

    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/sim_{n}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{n} records -> {path}")
    print(f"  effluent_f: [{df['effluent_f'].min():.4f}, {df['effluent_f'].max():.4f}]"
          f"  mean={df['effluent_f'].mean():.4f}"
          f"  <1.0: {100 * (df['effluent_f'] < 1.0).mean():.0f}%")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.n, args.seed)
