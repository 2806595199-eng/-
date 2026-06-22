"""测试：数据加载"""
import pytest
import pandas as pd
from training.data_loader import canonicalize_columns, validate_required_columns


def test_cn_to_en_mapping():
    df = pd.DataFrame({"入水口流量": [100], "入水PH值": [7.2]})
    result = canonicalize_columns(df)
    assert "influent_flow" in result.columns


def test_strip_columns():
    df = pd.DataFrame({" 入水口流量 ": [100], " 电导率 ": [6500]})
    result = canonicalize_columns(df)
    assert "influent_flow" in result.columns
    assert " 入水口流量 " not in result.columns


def test_duplicate_cn_mapping():
    df = pd.DataFrame({"混凝剂投加量": [500], "PAC投加量": [520]})
    with pytest.raises(ValueError, match="重复列映射"):
        canonicalize_columns(df)


def test_validate_training_missing():
    df = pd.DataFrame({"influent_f": [18]})
    with pytest.raises(ValueError, match="缺少必需列"):
        validate_required_columns(df, for_training=True)


def test_validate_inference_ok():
    from core import config as cfg
    df = pd.DataFrame({c: [1] for c in cfg.MODEL_INPUT_COLS})
    assert validate_required_columns(df, for_training=False) is True
