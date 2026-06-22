"""测试：成本计算"""
import pytest
from core import config as cfg
from serving.cost_calculator import (normalize_fraction, pacl_cost_per_ton,
    defluor_cost_per_ton, cost_per_hour)


def test_normalize_fraction():
    assert normalize_fraction(11.7) == pytest.approx(0.117)
    assert normalize_fraction(0.117) == pytest.approx(0.117)
    assert normalize_fraction(1.0) == 1.0
    with pytest.raises(ValueError):
        normalize_fraction(0)
    with pytest.raises(ValueError):
        normalize_fraction(-1)


def test_pacl_cost_ton():
    # mg_L_product: 1000 mg/L, 700 元/吨 → 0.7 元/吨水
    assert pacl_cost_per_ton(1000) == pytest.approx(0.7)


def test_defluor_cost_3():
    # 3.0 mL/L, density=1.4, price=2600 → 3 × 1.4 / 1000 × 2600 = 10.92
    assert defluor_cost_per_ton(3.0) == pytest.approx(10.92)


def test_defluor_cost_2_2():
    # 2.2 × 1.4 / 1000 × 2600 = 8.008
    assert defluor_cost_per_ton(2.2) == pytest.approx(8.008)


def test_cost_per_hour_ok():
    r = cost_per_hour(4.43, 100)
    assert r["cost_per_hour_yuan"] == pytest.approx(443)

def test_cost_per_hour_bad_flow():
    r = cost_per_hour(4.43, 0)
    assert r["cost_per_hour_yuan"] is None
    assert "warning" in r


def test_pacl_unknown_basis(monkeypatch):
    monkeypatch.setattr(cfg, "PACL_DOSE_BASIS", "unknown_mode")
    with pytest.raises(ValueError, match="PACL_DOSE_BASIS"):
        pacl_cost_per_ton(1000)


def test_defluor_unknown_basis(monkeypatch):
    monkeypatch.setattr(cfg, "DEFLUOR_DOSE_BASIS", "unknown_mode")
    with pytest.raises(ValueError, match="DEFLUOR_DOSE_BASIS"):
        defluor_cost_per_ton(3.0)
