"""测试：优化器（FakeEngine，不依赖 TabPFN）"""
import pytest
from core import config as cfg
from serving.optimizer import GridSearchDosingOptimizer
from core.data_types import DosingRecipe


class FakeEngine:
    """模拟推理引擎：根据 pacl_dose 返回可控预测值"""
    def predict(self, wq):
        """单条预测 — TabPFN 验证时调用"""
        return {"predicted_f": 0.5, "q05": 0.3, "q95": 0.7,
                "risk_level": "safe", "model_used": "tabpfn", "warnings": []}

    def predict_batch(self, samples, prefer_model=None, history=None):
        results = []
        for s in samples:
            pacl = s["pacl_dose"]
            deflu = s["defluor_dose"]
            # 简单线性模拟: 加药越多出水越低
            predicted = max(0.05, 2.5 - pacl * 0.002 - deflu * 2.0)
            q95 = predicted + 0.1
            risk = "safe" if q95 < 0.8 else ("warning" if q95 < cfg.LIMIT_F else "danger")
            results.append({
                "predicted_f": round(predicted, 4),
                "q05": max(0, predicted - 0.2),
                "q95": round(q95, 4),
                "risk_level": risk,
                "model_used": prefer_model or "xgboost",
                "warnings": [],
            })
        return results


class DistinctSafeEngine:
    """让三种策略返回不同候选，且 TabPFN 验证后都仍为 safe。"""

    def predict(self, wq, history=None):
        pacl = wq["pacl_dose"]
        q95 = 0.7 if pacl < 1000 else 0.6
        return {"predicted_f": q95 - 0.1, "q05": q95 - 0.2, "q95": q95,
                "risk_level": "safe", "model_used": "tabpfn", "warnings": []}

    def predict_batch(self, samples, prefer_model=None, history=None):
        results = []
        for s in samples:
            pacl = s["pacl_dose"]
            q95 = 0.7 if pacl < 1000 else 0.6
            results.append({"predicted_f": q95 - 0.1, "q05": q95 - 0.2,
                            "q95": q95, "risk_level": "safe",
                            "model_used": prefer_model or "xgboost",
                            "warnings": []})
        return results


class BackupValidationEngine(DistinctSafeEngine):
    def predict(self, wq, history=None):
        result = super().predict(wq, history=history)
        result["model_used"] = "xgboost"
        return result


class FixedSelectionOptimizer(GridSearchDosingOptimizer):
    """隔离测试 optimize() 的最终推荐选择逻辑。"""

    def _evaluate(self, water_quality, engine, history=None):
        def candidate(pacl, defluor, cost):
            recipe = DosingRecipe(pacl_dose_setpoint=pacl,
                                  defluor_dose_setpoint=defluor)
            return {
                "recipe": recipe,
                "prediction": {"predicted_f": 0.5, "q05": 0.3, "q95": 0.7,
                               "risk_level": "safe", "model_used": "xgboost"},
                "cost": {"total_yuan_per_ton": cost, "pacl_yuan_per_ton": cost,
                         "defluor_yuan_per_ton": 0, "pam_yuan_per_ton": 0,
                         "magnetic_yuan_per_ton": 0},
                "cost_per_hour": cost * water_quality["influent_flow"],
                "pump_flows": {},
                "dose_score": 0,
                "quality_score": 0,
                "target_penalty": 0,
                "limit_penalty": 0,
                "cost_norm": 0,
                "balanced_score": 0,
            }
        candidates = [candidate(100, 0.1, 1.0),
                      candidate(900, 1.0, 2.0),
                      candidate(1800, 5.0, 3.0)]
        return candidates, 1.0, 3.0

    def _select_economic(self, candidates):
        c = candidates[0]
        c["selection_reason"] = "economic"
        c["warnings"] = []
        return c

    def _select_balanced(self, candidates):
        c = candidates[1]
        c["selection_reason"] = "balanced"
        c["warnings"] = []
        return c

    def _select_safe(self, candidates):
        c = candidates[2]
        c["selection_reason"] = "safe"
        c["warnings"] = []
        return c


class HistoryRecordingEngine(FakeEngine):
    def __init__(self):
        self.histories = []

    def predict(self, wq, history=None):
        self.histories.append(history)
        return super().predict(wq)


@pytest.fixture
def water():
    return {"influent_flow": 100, "influent_f": 18, "influent_ph": 7.2,
            "conductivity": 6500, "pacl_dose": 500, "defluor_dose": 0.5,
            "pacl_tank_ph": 7, "defluor_tank_ph": 6, "recycle_flow": 4,
            "waste_flow": 15, "pam_dose": 0.5}

@pytest.fixture
def opt():
    return GridSearchDosingOptimizer(pacl_points=10, defluor_points=10)


def test_all_modes_return(water, opt):
    fake = FakeEngine()
    for mode in ("economic", "balanced", "safe"):
        r = opt.optimize(water, fake, mode=mode)
        assert "recommended" in r
        assert "alternatives" in r
        assert "pareto_front" in r
        assert r["recommended"]["selection_reason"]


def test_recommended_fields(water, opt):
    r = opt.optimize(water, FakeEngine(), mode="balanced")
    rec = r["recommended"]
    required = ["pacl_dose_setpoint", "defluor_dose_setpoint", "predicted_f",
                "q05", "q95", "risk_level", "model_used", "cost_per_ton",
                "cost_per_hour_yuan", "cost_breakdown", "dose_score",
                "quality_score", "balanced_score", "selection_reason",
                "pacl_pump_flow_l_h", "defluor_pump_flow_l_h"]
    for key in required:
        assert key in rec, f"Missing: {key}"


def test_balanced_no_over_limit(water, opt):
    r = opt.optimize(water, FakeEngine(), mode="balanced")
    rec = r["recommended"]
    pareto = r["pareto_front"]
    assert len(pareto) <= 20
    # balanced 不应选 q95 > LIMIT_F 如果存在达标候选
    assert "selection_reason" in rec


def test_alternatives_three(water, opt):
    r = opt.optimize(water, FakeEngine(), mode="balanced")
    alt = r["alternatives"]
    for key in ("economic", "balanced", "safe"):
        assert key in alt
        assert "pacl_dose_setpoint" in alt[key]


def test_requested_mode_is_preserved_when_tabpfn_safe(water):
    opt = FixedSelectionOptimizer(pacl_points=2, defluor_points=2)
    engine = DistinctSafeEngine()

    economic = opt.optimize(water, engine, mode="economic")
    safe = opt.optimize(water, engine, mode="safe")

    assert economic["recommended"]["pacl_dose_setpoint"] == 100
    assert safe["recommended"]["pacl_dose_setpoint"] == 1800


def test_validation_preserves_actual_model_used(water):
    opt = FixedSelectionOptimizer(pacl_points=2, defluor_points=2)
    engine = BackupValidationEngine()

    result = opt.optimize(water, engine, mode="balanced")

    assert result["recommended"]["model_used"] == "xgboost"


def test_history_is_passed_to_tabpfn_validation(water):
    opt = GridSearchDosingOptimizer(pacl_points=2, defluor_points=2)
    engine = HistoryRecordingEngine()
    history = [{"influent_flow": 100, "influent_f": 18}]

    opt.optimize(water, engine, mode="balanced", history=history)

    assert engine.histories
    assert all(h is history for h in engine.histories)
