"""GridSearchDosingOptimizer — 成本-水质均衡优化

三种推荐模式:
  economic: q95≤LIMIT_F 中选成本最低
  safe:     q95≤TARGET_F 中选成本最低
  balanced: 综合评分(cost_norm + quality_score + penalties)最低

每个候选方案包含: recipe, prediction, cost, pump_flows, balanced_score

Pareto front: cost_per_ton 与 q95 的 trade-off 前沿
"""

import numpy as np
import inspect
from core import config as cfg
from core.data_types import DosingRecipe
from serving.cost_calculator import recipe_cost, cost_per_hour
from serving.pump_converter import compute_pump_flows


class GridSearchDosingOptimizer:

    def __init__(self, pacl_points=None, defluor_points=None):
        self.pacl_points = pacl_points or cfg.PACL_GRID_POINTS
        self.defluor_points = defluor_points or cfg.DEFLUOR_GRID_POINTS

    # ═══════════════════════════════════════════════════
    # 构造候选
    # ═══════════════════════════════════════════════════

    def _build_candidates(self, water_quality: dict) -> list:
        """生成所有 (pacl, defluor) 组合的候选样本。

        这里是穷举网格搜索：点数越多，推荐越细，但计算越慢。
        """
        pacl_vals = np.linspace(cfg.PACL_RANGE[0], cfg.PACL_RANGE[1], self.pacl_points)
        defluor_vals = np.linspace(cfg.DEFLUOR_RANGE[0], cfg.DEFLUOR_RANGE[1],
                                   self.defluor_points)
        samples = []
        for p in pacl_vals:
            for d in defluor_vals:
                wq = {**water_quality, "pacl_dose": float(p), "defluor_dose": float(d)}
                samples.append(wq)
        return samples

    # ═══════════════════════════════════════════════════
    # 评估
    # ═══════════════════════════════════════════════════

    def _evaluate(self, water_quality: dict, engine) -> list:
        """对每个候选批量预测 + 成本计算。

        默认先用 XGBoost 快速评估全部候选；这一步用于找方向，不作为最终精度保证。
        """
        samples = self._build_candidates(water_quality)
        flow = water_quality.get("influent_flow", 0)

        if hasattr(engine, 'predict_batch'):
            predictions = engine.predict_batch(samples, prefer_model=cfg.FAST_OPTIMIZER_MODEL)
        else:
            predictions = [engine.predict(w) for w in samples]

        candidates = []
        for s, pred in zip(samples, predictions):
            recipe = DosingRecipe(
                pacl_dose_setpoint=round(s["pacl_dose"], 1),
                defluor_dose_setpoint=round(s["defluor_dose"], 4),
            )
            cost = recipe_cost(recipe)
            pump = compute_pump_flows(recipe, flow) if flow > 0 else {}
            cp = cost_per_hour(cost["total_yuan_per_ton"], flow)

            q95 = pred.get("q95", 999)
            predicted_f = pred.get("predicted_f", 999)

            # 评分逻辑:
            # q95 超过安全线会被惩罚；超过排放红线会被重罚。
            # 这样 balanced 模式不会只追求低成本，而会优先避开超标风险。
            target_violation = max(q95 - cfg.TARGET_F, 0)
            limit_violation = max(q95 - cfg.LIMIT_F, 0)
            quality_score = target_violation / max(cfg.LIMIT_F - cfg.TARGET_F, 1e-6)
            target_penalty = cfg.TARGET_VIOLATION_PENALTY * target_violation ** 2
            limit_penalty = cfg.LIMIT_VIOLATION_PENALTY * limit_violation ** 2
            dose_score = (s["pacl_dose"] / cfg.PACL_RANGE[1]
                          + s["defluor_dose"] / cfg.DEFLUOR_RANGE[1])

            candidates.append({
                "recipe": recipe,
                "prediction": {
                    "predicted_f": predicted_f,
                    "q05": pred.get("q05", 0),
                    "q95": q95,
                    "risk_level": pred.get("risk_level", "danger"),
                    "model_used": pred.get("model_used", cfg.FAST_OPTIMIZER_MODEL),
                },
                "cost": cost,
                "cost_per_hour": cp.get("cost_per_hour_yuan"),
                "pump_flows": pump,
                "dose_score": round(dose_score, 4),
                "quality_score": round(quality_score, 4),
                "target_penalty": round(target_penalty, 4),
                "limit_penalty": round(limit_penalty, 4),
            })

        # 计算归一化成本（用于 balanced 评分）。
        # 归一化后，成本项和水质风险项才能在同一套评分里比较。
        costs = [c["cost"]["total_yuan_per_ton"] for c in candidates]
        min_c, max_c = min(costs), max(costs)
        cost_range = max_c - min_c + 1e-9
        for c in candidates:
            c["cost_norm"] = (c["cost"]["total_yuan_per_ton"] - min_c) / cost_range
            c["balanced_score"] = (
                cfg.COST_WEIGHT * c["cost_norm"]
                + cfg.QUALITY_WEIGHT * c["quality_score"]
                + c["target_penalty"]
                + c["limit_penalty"]
            )

        return candidates, min_c, max_c

    @staticmethod
    def _predict_validated(engine, water_quality: dict, history=None) -> dict:
        """调用主模型验证候选；兼容简单测试引擎的旧 predict(wq) 签名。"""
        try:
            params = inspect.signature(engine.predict).parameters
        except (TypeError, ValueError):
            params = {}
        if "history" in params:
            return engine.predict(water_quality, history=history)
        return engine.predict(water_quality)

    # ═══════════════════════════════════════════════════
    # 三种策略选择
    # ═══════════════════════════════════════════════════

    def _select_economic(self, candidates: list) -> dict:
        """q95≤LIMIT_F 中选成本最低"""
        within = [c for c in candidates if c["prediction"]["q95"] <= cfg.LIMIT_F]
        if within:
            best = min(within, key=lambda c: c["cost"]["total_yuan_per_ton"])
            best["selection_reason"] = "economic_min_cost_within_limit"
            best["warnings"] = []
        else:
            best = min(candidates, key=lambda c: c["prediction"]["q95"])
            best["selection_reason"] = "economic_min_q95_no_candidate_meets_limit"
            best["warnings"] = [
                "所有候选方案均存在超标风险，当前结果为最低风险方案",
                "不建议直接作为运行设定，请开启保障单元"
            ]
        return best

    def _select_safe(self, candidates: list) -> dict:
        """q95≤TARGET_F 中选成本最低"""
        under_target = [c for c in candidates if c["prediction"]["q95"] <= cfg.TARGET_F]
        if under_target:
            best = min(under_target, key=lambda c: c["cost"]["total_yuan_per_ton"])
            best["selection_reason"] = "safe_min_cost_under_target"
            best["warnings"] = []
            return best
        within = [c for c in candidates if c["prediction"]["q95"] <= cfg.LIMIT_F]
        if within:
            best = min(within, key=lambda c: c["prediction"]["q95"])
            best["selection_reason"] = "safe_min_q95_no_candidate_under_target"
            best["warnings"] = [f"无方案达安全线({cfg.TARGET_F})，已选排放限值内最低风险方案"]
        else:
            best = min(candidates, key=lambda c: c["prediction"]["q95"])
            best["selection_reason"] = "safe_min_q95_all_exceed"
            best["warnings"] = ["所有候选均超标，已选最低风险方案"]
        return best

    def _select_balanced(self, candidates: list) -> dict:
        """综合评分最低，但不超过 LIMIT_F（如果存在达标候选）"""
        within = [c for c in candidates if c["prediction"]["q95"] <= cfg.LIMIT_F]
        search = within if within else candidates
        best = min(search, key=lambda c: c["balanced_score"])
        if not within:
            best["selection_reason"] = "balanced_min_score_no_candidate_meets_limit"
            best["warnings"] = ["所有候选均超标，已选综合评分最低方案"]
        else:
            best["selection_reason"] = "balanced_min_score"
            best["warnings"] = []
        return best

    # ═══════════════════════════════════════════════════
    # Pareto front
    # ═══════════════════════════════════════════════════

    def _pareto_front(self, candidates: list, max_points=20) -> list:
        """选出 cost-q95 的 Pareto 前沿。

        前沿上的点表示：想进一步降低 q95，通常就要接受更高成本。
        """
        # 按成本排序
        sorted_c = sorted(candidates, key=lambda c: c["cost"]["total_yuan_per_ton"])
        front = []
        best_q95 = float("inf")
        for c in sorted_c:
            q95 = c["prediction"]["q95"]
            if q95 < best_q95:
                best_q95 = q95
                front.append({
                    "pacl_dose_setpoint": c["recipe"].pacl_dose_setpoint,
                    "defluor_dose_setpoint": c["recipe"].defluor_dose_setpoint,
                    "predicted_f": c["prediction"]["predicted_f"],
                    "q95": q95,
                    "cost_per_ton": c["cost"]["total_yuan_per_ton"],
                    "balanced_score": c["balanced_score"],
                    "risk_level": c["prediction"]["risk_level"],
                })
        return front[:max_points]

    # ═══════════════════════════════════════════════════
    # 主入口
    # ═══════════════════════════════════════════════════

    def optimize(self, water_quality: dict, engine, mode="balanced", history=None) -> dict:
        """主优化入口

        Args:
            water_quality: dict
            engine: InferenceEngine
            mode: "economic" / "balanced" / "safe"

        Returns:
            dict with recommended, alternatives, pareto_front, warnings, assumptions
        """
        candidates, min_c, max_c = self._evaluate(water_quality, engine)
        flow = water_quality.get("influent_flow", 0)

        risk_order = {"safe": 0, "warning": 1, "danger": 2}

        # 三种模式 XGBoost 初选
        eco = self._select_economic(candidates)
        bal = self._select_balanced(candidates)
        saf = self._select_safe(candidates)

        # 收集 top N 候选：三种模式最优 + 成本最低 + q95 最低，去重
        top_pool = {id(c): c for c in (eco, bal, saf)}
        by_cost = sorted(candidates, key=lambda c: c["cost"]["total_yuan_per_ton"])
        by_q95 = sorted(candidates, key=lambda c: c["prediction"]["q95"])
        for c in by_cost[:cfg.TABPFN_VALIDATE_TOP_N // 2]:
            top_pool[id(c)] = c
        for c in by_q95[:cfg.TABPFN_VALIDATE_TOP_N // 2]:
            top_pool[id(c)] = c
        for c in top_pool.values():
            c.setdefault("selection_reason", "top_n_candidate")
            c.setdefault("warnings", [])
        validated_pool = list(top_pool.values())

        # 用主模型（TabPFN）验证 top N 候选，确保最终输出经过了主模型复核。
        for sel in validated_pool:
            wq_sel = {**water_quality,
                      "pacl_dose": sel["recipe"].pacl_dose_setpoint,
                      "defluor_dose": sel["recipe"].defluor_dose_setpoint}
            tabpfn_pred = self._predict_validated(engine, wq_sel, history=history)
            validated_model = tabpfn_pred.get("model_used", "unknown")
            sel["prediction"] = {
                "predicted_f": tabpfn_pred["predicted_f"],
                "q05": tabpfn_pred["q05"],
                "q95": tabpfn_pred["q95"],
                "risk_level": tabpfn_pred["risk_level"],
                "model_used": validated_model,
            }
            sel["model_used"] = validated_model

        # TabPFN 验完后从验证池按模式约束重新选择——不再信赖 XGBoost 初选
        def _q95_ok(c, m):
            q = c["prediction"]["q95"]
            limit = cfg.TARGET_F if m == "safe" else cfg.LIMIT_F
            return q <= limit

        mode_picks = {"economic": eco, "balanced": bal, "safe": saf}
        recommended = mode_picks[mode]

        # 优先保留模式初选；仅当其不满足约束时从验证池中替换
        if not _q95_ok(recommended, mode):
            constrained = [c for c in validated_pool if _q95_ok(c, mode)]
            if constrained:
                recommended = min(constrained, key=lambda c: c["cost"]["total_yuan_per_ton"])
                recommended["selection_reason"] += " (re-ranked, q95 ok)"
            else:
                recommended = min(validated_pool, key=lambda c: c["prediction"]["q95"])
                recommended["selection_reason"] += " (all exceed, min q95)"
        elif "selection_reason" not in recommended or not recommended.get("selection_reason"):
            recommended["selection_reason"] = f"{mode}_selected"

        # 风险优化：从完整验证池中找风险更低且更便宜的
        rec_risk = risk_order.get(recommended["prediction"]["risk_level"], 9)
        if rec_risk > 0:
            for alt in validated_pool:
                if alt is recommended:
                    continue
                alt_risk = risk_order.get(alt["prediction"]["risk_level"], 9)
                if alt_risk < rec_risk or (alt_risk == rec_risk and
                   alt["cost"]["total_yuan_per_ton"] < recommended["cost"]["total_yuan_per_ton"]):
                    rec_risk = alt_risk
                    alt["selection_reason"] += " (re-ranked)"
                    recommended = alt

        # 构造统一输出
        def _format(c, label):
            r = c["recipe"]
            pred = c["prediction"]
            cost = c["cost"]
            pump = c.get("pump_flows", {})
            cp = cost_per_hour(cost["total_yuan_per_ton"], flow)
            return {
                "scheme_label": label,
                "pacl_dose_setpoint": r.pacl_dose_setpoint,
                "defluor_dose_setpoint": r.defluor_dose_setpoint,
                "pacl_pump_flow_l_h": pump.get("pacl_pump_flow_l_h"),
                "defluor_pump_flow_l_h": pump.get("defluor_pump_flow_l_h"),
                "pump_flow_unit": pump.get("pump_flow_unit", "L/h"),
                "formula_basis": pump.get("formula_basis", {}),
                "predicted_f": pred["predicted_f"],
                "q05": pred.get("q05", 0),
                "q95": pred["q95"],
                "risk_level": pred["risk_level"],
                "model_used": pred.get("model_used", "unknown"),
                "cost_per_ton": cost["total_yuan_per_ton"],
                "cost_per_hour_yuan": cp.get("cost_per_hour_yuan"),
                "cost_breakdown": {
                    "pacl_yuan_per_ton": cost["pacl_yuan_per_ton"],
                    "defluor_yuan_per_ton": cost["defluor_yuan_per_ton"],
                    "pam_yuan_per_ton": cost["pam_yuan_per_ton"],
                    "magnetic_yuan_per_ton": cost["magnetic_yuan_per_ton"],
                },
                "dose_score": c["dose_score"],
                "quality_score": c["quality_score"],
                "balanced_score": c["balanced_score"],
                "selection_reason": c.get("selection_reason", ""),
                "warnings": c.get("warnings", []),
            }

        return {
            "recommended_mode": mode,
            "recommended": _format(recommended, mode),
            "alternatives": {
                "economic": _format(eco, "economic"),
                "balanced": _format(bal, "balanced"),
                "safe": _format(saf, "safe"),
            },
            "pareto_front": self._pareto_front(candidates),
            "warnings": recommended.get("warnings", []),
            "assumptions": {
                "target_f": cfg.TARGET_F,
                "limit_f": cfg.LIMIT_F,
                "pacl_price_yuan_t": cfg.PACL_PRICE_YUAN_T,
                "defluor_price_yuan_t": cfg.DEFLUOR_PRICE_YUAN_T,
                "defluor_density_kg_l": cfg.DEFLUOR_DENSITY_KG_L,
                "pacl_dose_basis": cfg.PACL_DOSE_BASIS,
                "defluor_dose_basis": cfg.DEFLUOR_DOSE_BASIS,
                "cost_prices_need_confirmation": True,
                "optimizer_model": cfg.FAST_OPTIMIZER_MODEL,
                "n_candidates": len(candidates),
            },
        }
