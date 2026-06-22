"""药剂成本计算 — 独立模块

所有成本参数来自 config.py，标注 TODO 待甲方确认。
"""

from core import config as cfg


def normalize_fraction(value: float) -> float:
    """归一化质量分数: >1 按百分数处理, 0~1 直接返回"""
    if value > 1:
        return value / 100.0
    if 0 < value <= 1:
        return value
    raise ValueError(f"无效的质量分数: {value}")


def pacl_cost_per_ton(pacl_dose: float) -> float:
    """PAC 吨水成本 元/吨水"""
    basis = cfg.PACL_DOSE_BASIS
    if basis == "mg_L_product":
        return pacl_dose / 1_000_000 * cfg.PACL_PRICE_YUAN_T
    elif basis == "mg_L_as_Al":
        product_dose = pacl_dose / normalize_fraction(cfg.PACL_AL_MASS_FRACTION)
        return product_dose / 1_000_000 * cfg.PACL_PRICE_YUAN_T
    elif basis == "mM_as_Al":
        as_Al_mg_L = pacl_dose * 26.98
        product_dose = as_Al_mg_L / normalize_fraction(cfg.PACL_AL_MASS_FRACTION)
        return product_dose / 1_000_000 * cfg.PACL_PRICE_YUAN_T
    raise ValueError(f"未知 PACL_DOSE_BASIS: {basis}")


def defluor_cost_per_ton(defluor_dose: float) -> float:
    """除氟剂吨水成本 元/吨水"""
    basis = cfg.DEFLUOR_DOSE_BASIS
    if basis == "mL_L_stock":
        return defluor_dose * cfg.DEFLUOR_DENSITY_KG_L / 1000 * cfg.DEFLUOR_PRICE_YUAN_T
    elif basis == "mg_L_active":
        return defluor_dose / 1_000_000 * cfg.DEFLUOR_PRICE_YUAN_T
    raise ValueError(f"未知 DEFLUOR_DOSE_BASIS: {basis}")


def pam_cost_per_ton(pam_dose_mg_L: float) -> float:
    return pam_dose_mg_L / 1_000_000 * cfg.PAM_PRICE_YUAN_T


def magnetic_cost_per_ton(magnetic_dose_g_L: float) -> float:
    makeup = magnetic_dose_g_L * cfg.MAGNETIC_LOSS_RATE
    return makeup / 1000 * cfg.MAGNETIC_PRICE_YUAN_T


def recipe_cost(recipe, include_fixed=None) -> dict:
    if hasattr(recipe, 'pacl_dose_setpoint'):
        pacl = recipe.pacl_dose_setpoint
        deflu = recipe.defluor_dose_setpoint
    else:
        pacl = recipe.get("pacl_dose_setpoint", 0)
        deflu = recipe.get("defluor_dose_setpoint", 0)

    if include_fixed is None:
        include_fixed = cfg.INCLUDE_FIXED_CHEMICAL_COSTS

    p_c = round(pacl_cost_per_ton(pacl), 6)
    d_c = round(defluor_cost_per_ton(deflu), 6)
    pa_c = round(pam_cost_per_ton(cfg.DEFAULT_PAM_DOSE_MG_L), 6) if include_fixed else 0.0
    m_c = round(magnetic_cost_per_ton(cfg.DEFAULT_MAGNETIC_DOSE_G_L), 6) if include_fixed else 0.0

    return {
        "total_yuan_per_ton": round(p_c + d_c + pa_c + m_c, 6),
        "pacl_yuan_per_ton": p_c,
        "defluor_yuan_per_ton": d_c,
        "pam_yuan_per_ton": pa_c,
        "magnetic_yuan_per_ton": m_c,
        "include_fixed_chemicals": include_fixed,
        "assumptions": {
            "pacl_price_yuan_t": cfg.PACL_PRICE_YUAN_T,
            "defluor_price_yuan_t": cfg.DEFLUOR_PRICE_YUAN_T,
            "defluor_density_kg_l": cfg.DEFLUOR_DENSITY_KG_L,
            "pacl_dose_basis": cfg.PACL_DOSE_BASIS,
            "defluor_dose_basis": cfg.DEFLUOR_DOSE_BASIS,
            "need_confirmation": True,
        },
    }


def cost_per_hour(cost_per_ton: float, influent_flow_m3_h: float) -> dict:
    if influent_flow_m3_h <= 0:
        return {"cost_per_hour_yuan": None,
                "warning": "influent_flow 缺失或<=0，无法计算每小时成本"}
    return {"cost_per_hour_yuan": round(cost_per_ton * influent_flow_m3_h, 2)}
