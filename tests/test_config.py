import pytest
import importlib

from core import config as cfg


def test_site_hrt_minutes_are_mapped_to_feature_delay_steps():
    assert cfg.TANK_HRT_MIN["influent_tank"] == 4
    assert cfg.POINT_TO_EFFLUENT_HRT_MIN["influent"] == 65
    assert cfg.POINT_TO_EFFLUENT_HRT_MIN["pacl"] == 61
    assert cfg.POINT_TO_EFFLUENT_HRT_MIN["defluor"] == 49
    assert cfg.POINT_TO_EFFLUENT_HRT_MIN["pam"] == 15

    assert cfg.FEATURE_DELAY_STEPS["influent_f"] == 7
    assert cfg.FEATURE_DELAY_STEPS["pacl_dose"] == 7
    assert cfg.FEATURE_DELAY_STEPS["defluor_dose"] == 5
    assert cfg.FEATURE_DELAY_STEPS["pam_dose"] == 2


def test_defluor_stock_concentration_from_client_reply():
    assert cfg.DEFLUOR_STOCK_MASS_FRACTION == pytest.approx(0.15)
    assert cfg.DEFLUOR_STOCK_CONC_G_L == pytest.approx(210.0)


def test_chemical_prices_match_pilot_workbook():
    assert cfg.PACL_PRICE_YUAN_T == pytest.approx(700.0)
    assert cfg.DEFLUOR_PRICE_YUAN_T == pytest.approx(2600.0)
    assert cfg.PAM_PRICE_YUAN_T == pytest.approx(11000.0)
    assert cfg.MAGNETIC_PRICE_YUAN_T == pytest.approx(3000.0)


def test_runtime_config_can_come_from_environment(monkeypatch):
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    monkeypatch.setenv("TABPFN_MODEL_CACHE_DIR", "/tmp/tabpfn-cache")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_DIR", "/tmp/defluor-logs")

    reloaded = importlib.reload(cfg)

    assert reloaded.DEVICE == "cpu"
    assert reloaded.TABPFN_MODEL_CACHE_DIR == "/tmp/tabpfn-cache"
    assert reloaded.LOG_LEVEL == "DEBUG"
    assert reloaded.LOG_DIR == "/tmp/defluor-logs"

    monkeypatch.delenv("MODEL_DEVICE")
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR")
    monkeypatch.delenv("LOG_LEVEL")
    monkeypatch.delenv("LOG_DIR")
    importlib.reload(cfg)
