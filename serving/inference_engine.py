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
                                 warnings=None) -> dict:
        """统一构造预测结果。

        q95 是偏保守的上置信界；推荐逻辑主要看 q95 是否超过安全线/红线。
        """
        raw_q95 = predicted + 1.645 * self.residual_std
        if raw_q95 < 0.8:
            risk = RiskLevel.SAFE
        elif raw_q95 < cfg.LIMIT_F:
            risk = RiskLevel.WARNING
        else:
            risk = RiskLevel.DANGER
        std_final = self.residual_std * 2 if risk == RiskLevel.DANGER else self.residual_std
        return {
            "predicted_f": round(predicted, 4),
            "q05": max(0, predicted - 1.645 * std_final),
            "q95": predicted + 1.645 * std_final,
            "risk_level": risk,
            "model_used": model_used,
            "warnings": warnings or [],
        }

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
                # 为了预测“当前运行状态经过 HRT 后的出水”，这里把当前水质复制到未来 horizon 行。
                # 经过 _delay_series shift 后，最后一行特征会对应当前投加/进水对未来出水的影响。
                future = pd.DataFrame([water_quality] * horizon)
                df = pd.concat([df, future], ignore_index=True)
            window_rows = self.engineer.min_history * 2 + horizon
            df_feat = self.engineer.transform(df.iloc[-window_rows:])
            X = df_feat.iloc[[-1]].reindex(columns=self.feature_names, fill_value=0)
            y_pred = self.model.predict(X.values.astype(np.float32))
            return self._build_prediction_result(float(y_pred[0]))
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

    def predict_batch(self, water_samples: list, prefer_model=None) -> list:
        """批量预测。默认与 predict() 一致，prefer_model="xgboost" 可切换。

        加药优化会产生大量候选组合，逐个跑 TabPFN 会很慢；
        因此默认用 XGBoost 快速粗筛，最终推荐前再由优化器调用主模型复核。
        prefer_model 为 None 时使用 FAST_OPTIMIZER_MODEL 配置。
        """
        if prefer_model is None:
            prefer_model = cfg.FAST_OPTIMIZER_MODEL

        if prefer_model == "xgboost" and self.backup_model is not None:
            feats = np.array([self._build_simple_features(w) for w in water_samples])
            preds = self.backup_model.predict(feats)
            return [self._build_prediction_result(float(p), model_used="xgboost")
                    for p in preds]

        # 主模型批量：XGBoost 特征 + TabPFN 预测，避免逐条重复特征工程
        if len(water_samples) > 1 and self.model is not None:
            try:
                feats = np.array([self._build_simple_features(w) for w in water_samples])
                y_preds = self.model.predict(feats.astype(np.float32))
                return [self._build_prediction_result(float(p), model_used="tabpfn")
                        for p in y_preds]
            except Exception:
                pass
        return [self.predict(w) for w in water_samples]

    # ── 备用 ──

    def _xgb_predict_result(self, wq: dict, warnings=None) -> dict:
        pred = float(self.backup_model.predict([self._build_simple_features(wq)])[0])
        return self._build_prediction_result(pred, model_used="xgboost",
                                             warnings=warnings)

    def _fallback_rule(self, wq: dict) -> dict:
        f_in = wq.get("influent_f", 18)
        rough = f_in * 0.3
        return self._build_prediction_result(rough, model_used="fallback_rule",
            warnings=["当前结果来自规则估算，不应用作正式推荐"])

    def _build_simple_features(self, wq: dict) -> np.ndarray:
        """XGBoost 简单特征 — 顺序必须与 cfg.XGB_BASE_COLS 和训练阶段完全一致。"""
        eps = 1e-6
        flow = wq.get("influent_flow", 100.0)
        return np.array([
            flow,
            wq.get("influent_ph", 7.0),
            wq.get("conductivity", 6500.0),
            wq.get("influent_f", 18.0),
            wq.get("pacl_dose", 0.0),
            wq.get("defluor_dose", 0.0),
            wq.get("pacl_tank_ph", 7.0),
            wq.get("defluor_tank_ph", 6.0),
            wq.get("recycle_flow", 0.0),
            wq.get("waste_flow", 0.0),
            wq.get("pam_dose", 0.0),
            wq.get("recycle_flow", 0.0) / max(flow, eps),
            wq.get("waste_flow", 0.0) / max(flow, eps),
            wq.get("pacl_dose", 0.0) * flow,
            wq.get("defluor_dose", 0.0) * flow,
        ], dtype=np.float32)
