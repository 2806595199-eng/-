"""时序特征工程 — 按上位机字段重构

关键约束:
1. transform() 返回已标准化的 DataFrame，调用方禁止再次 scaler.transform。
2. _build() 只做历史方向的 ffill，不做 bfill，也不在全量数据上用 median 兜底。
3. 缺失兜底值只由训练段统计得到，避免测试段分布泄漏到训练/评估。
4. 添加派生特征 recycle_ratio, waste_ratio, pacl_mass_rate, defluor_mass_rate。
5. effluent_f 绝不能进入特征列。

人工 review 重点:
- 先确认 config.FEATURE_DELAY_STEPS 是否符合现场采样点/投加点到出水端的 HRT。
- 再确认缺失值、标准化、特征列对齐都只使用训练段统计量。
- 最后确认任何包含 effluent_f 的列都只作为标签，不进入模型输入。
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict, Any
import json
from core import config as cfg


class FeatureEngineer:

    def __init__(self, lag_steps=(1, 2, 3, 5), rolling_windows=(3, 5),
                 min_history=10, feature_delay_steps: Optional[Dict[str, int]] = None):
        self.lag_steps = sorted(lag_steps)
        self.rolling_windows = sorted(rolling_windows)
        self.min_history = min_history
        if feature_delay_steps is None:
            feature_delay_steps = getattr(cfg, "FEATURE_DELAY_STEPS", {})
        # 每个变量可以有不同 HRT 延迟；例如进水指标、PACl、除氟剂不一定同一时刻影响出水。
        self.feature_delay_steps = {
            str(k): max(0, int(v)) for k, v in dict(feature_delay_steps).items()
        }
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names_: List[str] = []
        self.fill_values_: Dict[str, float] = {}  # 训练段各列中位数，推理和头部 lag 缺失兜底

    # ── 入口 ──

    def fit_transform(self, df: pd.DataFrame, target_col="effluent_f",
                      feat_cols=None, train_idx=None) -> pd.DataFrame:
        """构建训练特征并拟合标准化器。

        df 可以同时包含训练段和测试段，但 scaler、缺失值兜底统计量只能从 train_idx 指向的训练段学习。
        返回值已经标准化；如果后续代码再次 scaler.transform，会造成重复标准化。
        """
        if target_col in df.columns:
            df_feat = df.drop(columns=[target_col])
            y = df[target_col]
        else:
            df_feat = df
            y = None

        # _build 只生成原始特征，不学习任何统计量；真正的“学习”只发生在下面的训练段。
        X = self._build(df_feat, feat_cols)
        self.feature_names_ = list(X.columns)
        # 只从训练段计算填充值。这里不要用全量 X.median()，否则测试段极端值会影响评估。
        if train_idx is not None:
            fill_values = X.iloc[train_idx].median()
        else:
            fill_values = X.median()
        self.fill_values_ = fill_values.fillna(0.0).to_dict()
        X = X.fillna(self.fill_values_).fillna(0.0)

        if train_idx is not None:
            self.scaler.fit(X.iloc[train_idx])
        else:
            self.scaler.fit(X)

        self.fitted = True
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        if y is not None:
            Xs[target_col] = y.values
        return Xs

    def transform(self, df: pd.DataFrame, feat_cols=None, target_col=None) -> pd.DataFrame:
        """对在线新数据构建特征并标准化。

        推理时最好传入最近一段 history；如果只传单行，lag/rolling 特征会更多依赖训练段兜底值。
        返回值已经标准化，调用方禁止再次 scaler.transform()。
        """
        if not self.fitted:
            raise RuntimeError("请先调用 fit_transform()")
        if target_col and target_col in df.columns:
            df_feat = df.drop(columns=[target_col])
        else:
            df_feat = df

        X = self._build(df_feat, feat_cols)
        # 对齐训练时保存的列顺序。线上缺列时不能临时新增口径，只能用训练段兜底值补齐。
        X = X.reindex(columns=self.feature_names_, fill_value=None)
        for col in self.feature_names_:
            if X[col].isna().any():
                X[col] = X[col].fillna(self.fill_values_.get(col, 0.0))

        Xs = self.scaler.transform(X.values)
        return pd.DataFrame(Xs, columns=self.feature_names_, index=X.index)

    # ── 特征构建 ──

    def _build(self, df: pd.DataFrame, feat_cols=None) -> pd.DataFrame:
        """构建时序特征和工程派生特征。

        输入 df 的每一行是一条时间记录；输出仍保持同样行数，但每行会增加历史滞后、
        滚动统计、变化率等字段，用来描述“当前出水对应的历史运行状态”。
        """
        df = df.copy()
        columns: Dict[str, pd.Series] = {}
        series_by_col: Dict[str, pd.Series] = {}

        if feat_cols is None:
            feat_cols = cfg.TS_FEAT_COLS
        ts_cols = [c for c in feat_cols if c in df.columns]

        for col in ts_cols:
            series_by_col[col] = self._delay_series(df[col], col)

        # 派生特征把现场机理融入模型：回流/排泥相对进水量，以及药剂投加量随流量变化的负荷。
        # FIXME: 以下比例在各自延迟后的序列上计算，分子分母时间点不一致。
        # 例如 recycle_ratio@t = recycle_flow@(t-3) / influent_flow@(t-1)，物理含义模糊。
        # 正确做法：先在原始时序上算出比例，再把比例按 max(component_delays) 延迟。
        # 但修正后需重新训练模型，当前模型已经用现有特征训练，暂不动。
        eps = 1e-6
        if "recycle_flow" in series_by_col and "influent_flow" in series_by_col:
            series_by_col["recycle_ratio"] = (
                series_by_col["recycle_flow"] /
                series_by_col["influent_flow"].clip(lower=eps)
            )
        if "waste_flow" in series_by_col and "influent_flow" in series_by_col:
            series_by_col["waste_ratio"] = (
                series_by_col["waste_flow"] /
                series_by_col["influent_flow"].clip(lower=eps)
            )
        if "pacl_dose" in series_by_col and "influent_flow" in series_by_col:
            series_by_col["pacl_mass_rate"] = (
                series_by_col["pacl_dose"] * series_by_col["influent_flow"]
            )
        if "defluor_dose" in series_by_col and "influent_flow" in series_by_col:
            series_by_col["defluor_mass_rate"] = (
                series_by_col["defluor_dose"] * series_by_col["influent_flow"]
            )

        # 派生特征也做时序展开，因为“药剂负荷的历史趋势”通常比单点值更有信息。
        derived_cols = [c for c in series_by_col.keys() if c not in ts_cols]

        # 时序展开:
        # - lag 表示前 1/2/3/5 个采样点的值；
        # - rolling mean/std/slope 表示近期水平、波动和趋势；
        # - diff/pct_change 表示短期变化速度。
        for col in ts_cols + derived_cols:
            if col not in series_by_col:
                continue
            # 用三元而非 .get()，避免 df[col] 被提前求值触发 KeyError
            series = series_by_col[col]
            columns[col] = series
            for lag in self.lag_steps:
                columns[f"{col}_lag{lag}"] = series.shift(lag)
            for w in self.rolling_windows:
                roll = series.rolling(window=w, min_periods=max(2, w // 2))
                columns[f"{col}_rmean{w}"] = roll.mean()
                columns[f"{col}_rstd{w}"] = roll.std()
                columns[f"{col}_slope{w}"] = self._rolling_slope(series, w)
            columns[f"{col}_pct_change"] = series.pct_change().replace([np.inf, -np.inf], np.nan)
            columns[f"{col}_diff1"] = series.diff(1)
            columns[f"{col}_diff2"] = series.diff(2)

        result = pd.DataFrame(columns, index=df.index)
        # 只允许使用过去值 ffill；剩余头部缺失交给 fit/transform 的训练段统计量处理。
        result = result.ffill()
        result = result.clip(lower=-1e6, upper=1e6)
        return result

    def _delay_series(self, series: pd.Series, col: str) -> pd.Series:
        """按变量自己的 HRT 延迟取历史值，保证特征方向是“过去影响未来出水”。"""
        delay = self.feature_delay_steps.get(col, 0)
        if delay <= 0:
            return series
        return series.shift(delay)

    def prediction_horizon_steps(self) -> int:
        return max(self.feature_delay_steps.values(), default=0)

    @staticmethod
    def _rolling_slope(series, window):
        """计算窗口内趋势斜率；正值表示近期上升，负值表示近期下降。"""
        x = np.arange(window)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return pd.Series(np.nan, index=series.index)
        def _slope(yw):
            if len(yw) < window or np.any(np.isnan(yw)):
                return np.nan
            ym = yw.mean()
            return ((yw - ym) * (x - x_mean)).sum() / denom
        return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)

    # ── 配置序列化 ──

    def get_config(self):
        return {
            "lag_steps": list(self.lag_steps),
            "rolling_windows": list(self.rolling_windows),
            "min_history": self.min_history,
            "feature_delay_steps": self.feature_delay_steps,
            "feature_names": self.feature_names_,
            "fill_values": {k: float(v) if not np.isnan(v) else 0.0
                           for k, v in self.fill_values_.items()},
        }

    def save_config(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.get_config(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_config(cls, config_dict):
        obj = cls(
            lag_steps=tuple(config_dict.get("lag_steps", [1, 2, 3, 5])),
            rolling_windows=tuple(config_dict.get("rolling_windows", [3, 5])),
            min_history=config_dict.get("min_history", 10),
            feature_delay_steps=config_dict.get("feature_delay_steps", {}),
        )
        obj.feature_names_ = config_dict.get("feature_names", [])
        obj.fill_values_ = config_dict.get("fill_values", {})
        obj.fitted = True
        return obj
