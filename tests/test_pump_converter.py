"""测试：泵流量换算"""
import pytest
from core import config as cfg
from serving.pump_converter import (pump_flow_from_mg_l, pacl_pump_flow_l_h,
    defluor_pump_flow_l_h, compute_pump_flows)
from core.data_types import DosingRecipe


def test_pump_flow_basic():
    assert pump_flow_from_mg_l(500, 100, 100, 1.0) == pytest.approx(500)

def test_pump_flow_zero_flow():
    with pytest.raises(ValueError, match="influent_flow"):
        pump_flow_from_mg_l(500, 0, 100, 1.0)

def test_pump_flow_zero_conc():
    with pytest.raises(ValueError, match="药液浓度"):
        pump_flow_from_mg_l(500, 100, 0, 1.0)

def test_defluor_pump_flow():
    # 0.5 mL/L × 100 m3/h = 50 L/h
    assert defluor_pump_flow_l_h(0.5, 100) == pytest.approx(50)

def test_defluor_pump_zero_flow():
    with pytest.raises(ValueError):
        defluor_pump_flow_l_h(0.5, 0)

def test_pacl_as_Al(monkeypatch):
    monkeypatch.setattr(cfg, "PACL_DOSE_BASIS", "mg_L_as_Al")
    monkeypatch.setattr(cfg, "PACL_AL_MASS_FRACTION", 0.117)
    monkeypatch.setattr(cfg, "PACL_STOCK_CONC_G_L", 100.0)
    # dose=117, fraction=0.117, stock=100 → q = 117*100/(100*0.117) = 1000
    assert pacl_pump_flow_l_h(117, 100) == pytest.approx(1000, rel=0.01)

def test_pacl_mM_as_Al(monkeypatch):
    monkeypatch.setattr(cfg, "PACL_DOSE_BASIS", "mM_as_Al")
    monkeypatch.setattr(cfg, "PACL_AL_MASS_FRACTION", 0.117)
    monkeypatch.setattr(cfg, "PACL_STOCK_CONC_G_L", 100.0)
    # dose=1 mM, w0=0.117, rho0=100 → q = 26.98*1*100/(0.117*100) ≈ 230.6
    q = pacl_pump_flow_l_h(1.0, 100)
    assert q == pytest.approx(230.6, rel=0.05)

def test_pacl_unknown_basis(monkeypatch):
    monkeypatch.setattr(cfg, "PACL_DOSE_BASIS", "bad")
    with pytest.raises(ValueError):
        pacl_pump_flow_l_h(500, 100)

def test_defluor_unknown_basis(monkeypatch):
    monkeypatch.setattr(cfg, "DEFLUOR_DOSE_BASIS", "bad")
    with pytest.raises(ValueError):
        defluor_pump_flow_l_h(0.5, 100)

def test_compute_pump_flows():
    recipe = DosingRecipe(pacl_dose_setpoint=500, defluor_dose_setpoint=0.5)
    r = compute_pump_flows(recipe, 100)
    assert "pacl_pump_flow_l_h" in r
    assert "defluor_pump_flow_l_h" in r
    assert r["pump_flow_unit"] == "L/h"
    assert "formula_basis" in r

def test_compute_pump_zero_flow():
    recipe = DosingRecipe(pacl_dose_setpoint=500, defluor_dose_setpoint=0.5)
    with pytest.raises(ValueError):
        compute_pump_flows(recipe, 0)
