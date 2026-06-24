"""推理引擎 — 预测沉后出水氟化物浓度

无在线漂移检测、自适应校准、动态集成。
"""

import os, pickle, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from training import model_exporter as me
from training.model_registry import read_active_model, resolve_model_dir
from core.tabpfn_runtime import configure_tabpfn_cache, resolve_tabpfn_device

# TabPFN 延迟导入，避免单元测试强依赖
TabPFNRegressor = None

def _get_tabpfn():
    global TabPFNRegressor
    if TabPFNRegressor is None:
        configure_tabpfn_cache()
        from tabpfn import TabPFNRegressor as T
        TabPFNRegressor = T
    return TabPFNRegressor
from core import feature_engineer as fe
from core import config as cfg
from core.data_types import RiskLevel

def _force_backup() -> bool:
    return os.environ.get("USE_BACKUP", "").lower() in {"1", "true", "yes", "on"}


class InferenceEngine:

    def __init__(self, models_dir="models"):
        self.models_root = Path(models_dir)
        # models_root 是模型根目录；models_dir 会解析到 active_model.json 指向的具体版本目录。
        self.models_dir = resolve_model_dir(self.models_root)
        self.scaler = None        # 仅存储，不用于 predict（engineer.transform 已标准化）
        self.model = None
        self.backup_model = None
        self.engineer = None
        self.feature_names = []
        self.version = "0.0.0"
        self.residual_std = 0.12
        self.xgb_residual_std = 0.12
        self.main_model_fitted = False
        self.start_time = None

    def load(self):
        self.start_time = time.time()

        # 线上推理至少需要特征配置、标准化器、TabPFN 训练样本。
        # TabPFN 在这里用 train_data.pkl 重新 fit，而不是直接加载一个 pickle 后的主模型。
        required = ["feature_config.json", "scaler.pkl", "train_data.pkl"]
        missing = [name for name in required if not (self.models_dir / name).exists()]
        if missing:
            raise RuntimeError(f"缺少模型产物: {missing}，请先运行 train.py")

        cfg_path = self.models_dir / "feature_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                c = json.load(f)
            self.engineer = fe.FeatureEngineer.from_config(c)
            self.feature_names = self.engineer.feature_names_

        scaler_path = self.models_dir / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = me.load_scaler(str(scaler_path))
            if self.engineer is not None:
                self.engineer.scaler = self.scaler

        meta_path = self.models_dir / "model_metadata.json"
        if meta_path.exists():
            meta = me.load_metadata(str(meta_path))
            self.version = str(meta.get("r2_test", "0.0.0"))
            self.residual_std = meta.get("residual_std", 0.12) or 0.12
            self.xgb_residual_std = meta.get("xgb_residual_std", 0.12) or 0.12
        active_model = read_active_model(self.models_root)
        if active_model and active_model.get("active_version"):
            self.version = str(active_model["active_version"])

        # 先加载 XGBoost 备用模型。它主要用于优化器快速扫网格，也可在 TabPFN 不可用时兜底。
        backup_path = self.models_dir / "backup_model.pkl"
        if backup_path.exists():
            with open(backup_path, "rb") as f:
                self.backup_model = pickle.load(f)
            print("[Engine] backup (XGBoost) loaded")

        self.model = None
        if _force_backup():
            self.main_model_fitted = False
            print("[Engine] USE_BACKUP=true; skipping TabPFN fit")
        else:
            try:
                # TabPFN 是主模型。启动服务时重新 fit 训练样本，保证与当前 active 版本一致。
                device = resolve_tabpfn_device()
                self.model = _get_tabpfn()(device=device, ignore_pretraining_limits=True)
                train_path = self.models_dir / "train_data.pkl"
                with open(train_path, "rb") as f:
                    data = pickle.load(f)
                self.model.fit(data["X_train"], data["y_train"])
                self.main_model_fitted = True
                print(f"[Engine] TabPFN device={device}; fitted on {len(data['y_train'])} samples")
            except Exception as e:
                self.model = None
                self.main_model_fitted = False
                if self.backup_model is None and not cfg.ALLOW_FALLBACK_PREDICTION:
                    raise
                print(f"[Warning] TabPFN unavailable ({e}); using backup/fallback model")

        print(f"[Engine] features={len(self.feature_names)} v{self.version} "
              f"std={self.residual_std}")

    # ── 预测 ──

    def _build_prediction_result(self, predicted: float, model_used="tabpfn",
                                 warnings=None, low_confidence=False,
                                 residual_std_override=None) -> dict:
        """统一构造预测结果。

        q95 是偏保守的上置信界；推荐逻辑主要看 q95 是否超过安全线/红线。
        residual_std_override: 若指定则使用该值（XGBoost 使用自己的误差分布）
        """
        std = residual_std_override if residual_std_override is not None else self.residual_std
        raw_q95 = predicted + 1.645 * std
        if raw_q95 < 0.8:
            risk = RiskLevel.SAFE
        elif raw_q95 < cfg.LIMIT_F:
            risk = RiskLevel.WARNING
        else:
            risk = RiskLevel.DANGER
        std_final = std * 2 if risk == RiskLevel.DANGER else std
        result = {
            "predicted_f": round(predicted, 4),
            "q05": max(0, predicted - 1.645 * std_final),
            "q95": predicted + 1.645 * std_final,
            "risk_level": risk,
            "model_used": model_used,
            "warnings": warnings or [],
        }
        if low_confidence:
            result["low_confidence"] = True
            result["warnings"].append("历史数据不足，滞后/滚动特征部分由训练集中位数填充，预测置信度偏低")
        return result

    def predict(self, water_quality: dict, history=None) -> dict:
        """单次推理。

        engineer.transform() 已经完成标准化，不重复 scaler。
        传入 history 时，模型能计算真实 lag/rolling 特征；只传单行时，时序信息会变弱。
        """
        if _force_backup() or self.model is None or not self.main_model_fitted:
            if self.backup_model is not None:
                return self._xgb_predict_result(water_quality)
            if cfg.ALLOW_FALLBACK_PREDICTION:
                return self._fallback_rule(water_quality)
            raise RuntimeError("模型未加载或预测失败，请先训练模型并检查 models 目录")

        try:
            new_row = pd.DataFrame([water_quality])
            df = pd.concat([history, new_row], ignore_index=True) if history is not None else new_row
            horizon = self.engineer.prediction_horizon_steps()
            if horizon > 0:
                future = pd.DataFrame([water_quality] * horizon)
                df = pd.concat([df, future], ignore_index=True)
            window_rows = self.engineer.min_history * 2 + horizon
            # 历史不足时标记低置信度：lag 特征超过可用历史的部分由训练集中位数填充
            min_reliable = max(cfg.LAG_STEPS) + max(cfg.ROLLING_WINDOWS) + horizon
            low_conf = len(df) < min_reliable
            df_feat = self.engineer.transform(df.iloc[-window_rows:])
            X = df_feat.iloc[[-1]].reindex(columns=self.feature_names, fill_value=0)
            y_pred = self.model.predict(X.values.astype(np.float32))
            return self._build_prediction_result(float(y_pred[0]), low_confidence=low_conf)
        except Exception as e:
            # 异常时先试用 XGBoost（无条件，与主路径一致），再尝试规则兜底（需开关）
            if self.backup_model is not None:
                print(f"[Warning] TabPFN failed: {e}, using XGBoost")
                return self._xgb_predict_result(
                    water_quality,
                    warnings=[f"TabPFN failed, using XGBoost: {e}"],
                )
            if cfg.ALLOW_FALLBACK_PREDICTION:
                return self._fallback_rule(water_quality)
            raise RuntimeError(f"预测失败: {e}") from e

    def predict_batch(self, water_samples: list, prefer_model=None, history=None) -> list:
        """批量预测。默认与 predict() 一致，prefer_model="xgboost" 可切换。

        加药优化会产生大量候选组合，逐个跑 TabPFN 会很慢；
        因此默认用 XGBoost 快速粗筛，最终推荐前再由优化器调用主模型复核。
        prefer_model 为 None 时使用 FAST_OPTIMIZER_MODEL 配置。
        history 为历史水质 DataFrame，用于构建与训练一致的 HRT 延迟特征。
        """
        if prefer_model is None:
            prefer_model = cfg.FAST_OPTIMIZER_MODEL

        if prefer_model == "xgboost" and self.backup_model is not None:
            feats = np.array([self._build_simple_features(w, history) for w in water_samples])
            preds = self.backup_model.predict(feats)
            return [self._build_prediction_result(float(p), model_used="xgboost",
                     residual_std_override=self.xgb_residual_std)
                    for p in preds]

        # 主模型批量：历史特征一次计算，候选剂量只替换最后一行特征的差异列
        if len(water_samples) > 1 and self.model is not None and self.engineer is not None:
            try:
                # 用 history 构建一次完整特征矩阵（包含所有滞后/滚动/派生特征）
                if history is not None and len(history) > 0:
                    horizon = self.engineer.prediction_horizon_steps()
                    window_rows = self.engineer.min_history * 2 + horizon
                    # 先用最后一条 history 填 horizon，构建基础特征矩阵
                    last_wq = history.iloc[-1].to_dict()
                    future = pd.DataFrame([last_wq] * (horizon + 1))
                    df_base = pd.concat([history, future], ignore_index=True)
                    feat_base = self.engineer.transform(df_base.iloc[-window_rows:])
                    base_vec = feat_base.iloc[-1].values  # 基础特征行（用历史最后一条的剂量）
                else:
                    return [self.predict(w, history=history) for w in water_samples]

                # 识别候选剂量相关列：pacl_dose / defluor_dose 及其派生特征
                dose_cols = [c for c in feat_base.columns
                             if 'pacl_dose' in c or 'defluor_dose' in c
                             or 'pacl_mass' in c or 'defluor_mass' in c]

                # 为每个候选构建特征行：复制基础特征，替换剂量相关列为候选值
                feat_rows = []
                for wq in water_samples:
                    vec = base_vec.copy()
                    # 用候选剂量重建一行的特征 → 只改剂量相关列
                    tmp_df = pd.concat([history, pd.DataFrame([wq])], ignore_index=True)
                    tmp_horizon = pd.DataFrame([wq] * (horizon + 1))
                    tmp_full = pd.concat([history, tmp_horizon], ignore_index=True)
                    tmp_feat = self.engineer.transform(tmp_full.iloc[-window_rows:])
                    candidate_vec = tmp_feat.iloc[-1].values
                    # 只取候选诊断相关列，其余保留基础特征
                    for j, col in enumerate(feat_base.columns):
                        if col in dose_cols:
                            vec[j] = candidate_vec[j]
                    feat_rows.append(vec)

                X = np.vstack(feat_rows).astype(np.float32)
                y_preds = self.model.predict(X)
                return [self._build_prediction_result(float(p), model_used="tabpfn")
                        for p in y_preds]
            except Exception as e:
                print(f"[Warning] batch TabPFN failed: {e}, falling back to per-sample")
        return [self.predict(w, history=history) for w in water_samples]

    # ── 备用 ──

    def _xgb_predict_result(self, wq: dict, warnings=None) -> dict:
        pred = float(self.backup_model.predict([self._build_simple_features(wq)])[0])
        return self._build_prediction_result(pred, model_used="xgboost",
                                             warnings=warnings,
                                             residual_std_override=self.xgb_residual_std)

    def _fallback_rule(self, wq: dict) -> dict:
        f_in = wq.get("influent_f", 18)
        rough = f_in * 0.3
        return self._build_prediction_result(rough, model_used="fallback_rule",
            warnings=["当前结果来自规则估算，不应用作正式推荐"])

    def _build_simple_features(self, wq: dict, history=None) -> np.ndarray:
        """XGBoost 特征 — 顺序必须与 cfg.XGB_BASE_COLS 和训练阶段完全一致。

        若传入 history，使用 HRT 延迟后的历史值构建特征，与训练口径对齐。
        若没有 history，回退到当前值（优化器单候选调用场景）。
        """
        eps = 1e-6
        # 用 HRT 延迟后的值：从 history 中取 delay_steps 前的那一行
        def _hrt_val(col, default):
            delay = self.engineer.feature_delay_steps.get(col, 0) if self.engineer else 0
            if history is not None and len(history) > delay:
                val = history[col].iloc[-(delay + 1)]
                if pd.notna(val):
                    return float(val)
            return wq.get(col, default)

        flow = _hrt_val("influent_flow", 100.0)
        inf_flow = _hrt_val("influent_flow", 100.0)
        return np.array([
            flow,
            _hrt_val("influent_ph", 7.0),
            _hrt_val("conductivity", 6500.0),
            _hrt_val("influent_f", 18.0),
            _hrt_val("pacl_dose", 0.0),
            _hrt_val("defluor_dose", 0.0),
            _hrt_val("pacl_tank_ph", 7.0),
            _hrt_val("defluor_tank_ph", 6.0),
            _hrt_val("recycle_flow", 0.0),
            _hrt_val("waste_flow", 0.0),
            _hrt_val("pam_dose", 0.0),
            _hrt_val("recycle_flow", 0.0) / max(inf_flow, eps),
            _hrt_val("waste_flow", 0.0) / max(inf_flow, eps),
            _hrt_val("pacl_dose", 0.0) * inf_flow,
            _hrt_val("defluor_dose", 0.0) * inf_flow,
        ], dtype=np.float32)
