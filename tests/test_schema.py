"""测试：schema 一致性"""
from core import config as cfg
from serving.serve import WQInput


def test_target_not_in_input():
    assert cfg.TARGET_COL == "effluent_f"
    assert "effluent_f" not in cfg.MODEL_INPUT_COLS


def test_wqinput_no_effluent():
    fields = list(WQInput.model_fields.keys())
    assert "effluent_f" not in fields


def test_config_no_plc():
    assert not hasattr(cfg, "PLC_IP"), "PLC_IP should be removed"
    assert not hasattr(cfg, "PLC_READ_POINTS"), "PLC_READ_POINTS should be removed"
    assert not hasattr(cfg, "PLC_WRITE_POINTS"), "PLC_WRITE_POINTS should be removed"
