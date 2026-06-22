"""数据加载与清洗 — 支持中文列名映射 + 字段校验"""

import pandas as pd
import numpy as np
from pathlib import Path
from core import config as cfg

# ── 中文列名 → 英文字段映射 ──
# 现场表格可能使用中文列名，模型内部统一使用英文 canonical 字段。
# 人工 review 时如果发现甲方表头变化，优先在这里补映射，而不是到处改业务代码。
CN_TO_EN = {
    "入水口流量": "influent_flow",
    "入水PH值": "influent_ph", "入水 PH 值": "influent_ph",
    "入水电导率": "conductivity",
    "入水氟化物浓度": "influent_f",
    "混凝剂投加量": "pacl_dose", "PAC投加量": "pacl_dose",
    "除氟剂投加量": "defluor_dose",
    "混凝剂投加池的PH值": "pacl_tank_ph", "混凝剂投加池PH值": "pacl_tank_ph",
    "除氟剂投加池的PH值": "defluor_tank_ph", "除氟剂投加池PH值": "defluor_tank_ph",
    "回流流量": "recycle_flow",
    "剩余流量": "waste_flow",
    "PAM投加量": "pam_dose",
    "沉后出水氟化物浓度": "effluent_f",
    "出水氟浓度": "effluent_f", "出水氟化物浓度": "effluent_f",
    # 兼容旧列名
    "进水氟浓度": "influent_f", "进水pH": "influent_ph",
    "PACl投加量": "pacl_dose", "PACl_dose": "pacl_dose",
    "电导率": "conductivity",
    "浊度": "turbidity", "COD": "cod", "ORP": "orp",
    "磁粉投加量": "magnetic_dose", "mud_recycle_mi": "mud_recycle_mi",
    "pH_removal": "pacl_tank_ph", "ph_removal": "pacl_tank_ph",
    "pH_floc": "defluor_tank_ph", "ph_floc": "defluor_tank_ph",
}


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将中文列名映射为英文 canonical schema

    - 所有列名先 strip
    - 英文列名有空格也修正
    - 中文列映射后出现重复 canonical 字段时抛 ValueError
    """
    df = df.copy()
    # 先清理所有列名空格，避免 “入水PH值 ” 这类表头因为隐藏空格匹配失败。
    df.columns = [str(c).strip() for c in df.columns]
    # 英文列有前后空格的，直接修正（例如 " influent_flow " → "influent_flow"）
    known_en = set(cfg.MODEL_INPUT_COLS) | {cfg.TARGET_COL} | set(cfg.OPTIONAL_INPUT_COLS)
    rename_map = {}
    for col in df.columns:
        if col in known_en:
            continue
        if col in CN_TO_EN:
            en = CN_TO_EN[col]
            # 检查是否已有同一 canonical 列冲突
            existing = [c for c in df.columns if c in CN_TO_EN and CN_TO_EN[c] == en and c != col]
            if en in df.columns or existing:
                all_cn = [col] + existing + ([en] if en in df.columns else [])
                raise ValueError(f"重复列映射到 {en}: {all_cn}")
            rename_map[col] = en
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_required_columns(df: pd.DataFrame, for_training: bool = True):
    """校验必需列是否存在，缺列抛出 ValueError"""
    required = list(cfg.MODEL_INPUT_COLS)
    if for_training:
        required.append(cfg.TARGET_COL)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"缺少必需列: {missing}\n"
            f"  现有列: {list(df.columns)}\n"
            f"  请在 CSV 中使用中文列名或英文字段名"
        )
    return True


def load_data(filepath: str) -> pd.DataFrame:
    """加载 CSV/Excel，自动映射中文列名"""
    p = Path(filepath)
    if p.suffix == ".csv":
        df = pd.read_csv(p, encoding="utf-8-sig")
    elif p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        raise ValueError(f"不支持: {p.suffix}")
    return canonicalize_columns(df)


def clean_data(df: pd.DataFrame, for_training: bool = True) -> pd.DataFrame:
    """清洗：
    - 只用 ffill（避免未来信息泄露），后用列中位数兜底
    - 训练时 target 缺失的行直接 drop

    注意：更严格的生产口径可以把缺失值填充也做成“只 fit 训练段”的 transformer。
    当前特征工程层已经对模型特征做了训练段统计兜底。
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 训练目标缺失 → drop
    if for_training and cfg.TARGET_COL in df.columns:
        before = len(df)
        df = df.dropna(subset=[cfg.TARGET_COL])
        if len(df) < before:
            print(f"  Dropped {before - len(df)} rows with missing effluent_f")

    # 数值列：ffill → 中位数兜底
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].ffill()
            df[col] = df[col].fillna(df[col].median())

    return df


def align_target_by_delay(df: pd.DataFrame, delay_steps: int = 0) -> pd.DataFrame:
    """按水力停留时间对齐 target
    X(t) → effluent_f(t+delay_steps)
    delay=0 表示不对齐

    当前项目主要使用 config.FEATURE_DELAY_STEPS 做“按变量独立延迟”。
    只有在不使用按变量延迟时，才考虑打开这个全局 target 平移。
    """
    if delay_steps <= 0:
        return df
    target = cfg.TARGET_COL
    if target in df.columns:
        df[target] = df[target].shift(-delay_steps)
        df = df.dropna(subset=[target])
    return df


def split_train_test(df, target_col=None, test_ratio=0.2, shuffle=False):
    """划分训练/测试集（时序默认不打乱）"""
    if target_col is None:
        target_col = cfg.TARGET_COL
    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_test = max(1, int(len(df) * test_ratio))
    train = df.iloc[:-n_test].reset_index(drop=True)
    test = df.iloc[-n_test:].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c != target_col]
    return train[feat_cols], test[feat_cols], train[target_col], test[target_col]
