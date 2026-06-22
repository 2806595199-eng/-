"""计量泵流量换算 — 支持全部投加量单位模式"""

from core import config as cfg
from serving.cost_calculator import normalize_fraction


def pump_flow_from_mg_l(dose_mg_l: float, water_flow_m3_h: float,
                         stock_conc_g_l: float, active_fraction: float = 1.0) -> float:
    """通用泵流量公式: q = dose × Q / (stock_conc × active_fraction)

    Args:
        dose_mg_l: 有效成分投加量 mg/L
        water_flow_m3_h: 进水流量 m3/h
        stock_conc_g_l: 药液有效浓度 g/L
        active_fraction: 药液中有效成分质量分数
    """
    if water_flow_m3_h <= 0:
        raise ValueError("influent_flow 必须 > 0 才能计算泵流量")
    if stock_conc_g_l <= 0:
        raise ValueError(f"药液浓度必须 > 0, got {stock_conc_g_l}")
    if active_fraction <= 0:
        raise ValueError(f"有效成分分数必须 > 0, got {active_fraction}")
    return dose_mg_l * water_flow_m3_h / (stock_conc_g_l * active_fraction)


def pacl_pump_flow_l_h(pacl_dose: float, water_flow_m3_h: float) -> float:
    """PAC 计量泵流量 L/h"""
    basis = cfg.PACL_DOSE_BASIS
    if basis == "mg_L_product":
        return pump_flow_from_mg_l(pacl_dose, water_flow_m3_h, cfg.PACL_STOCK_CONC_G_L, 1.0)
    elif basis == "mg_L_as_Al":
        fraction = normalize_fraction(cfg.PACL_AL_MASS_FRACTION)
        return pump_flow_from_mg_l(pacl_dose, water_flow_m3_h,
                                    cfg.PACL_STOCK_CONC_G_L, fraction)
    elif basis == "mM_as_Al":
        # dose(mM as Al) × 26.98 → mg/L as Al
        # q = 26.98 × dose × Q / (w0 × rho0)
        w0 = normalize_fraction(cfg.PACL_AL_MASS_FRACTION)
        rho0 = cfg.PACL_STOCK_CONC_G_L
        return 26.98 * pacl_dose * water_flow_m3_h / (w0 * rho0)
    raise ValueError(f"未知 PACL_DOSE_BASIS: {basis}")


def defluor_pump_flow_l_h(defluor_dose: float, water_flow_m3_h: float) -> float:
    """除氟剂计量泵流量 L/h"""
    if water_flow_m3_h <= 0:
        raise ValueError("influent_flow 必须 > 0 才能计算泵流量")

    basis = cfg.DEFLUOR_DOSE_BASIS
    if basis == "mL_L_stock":
        # mL/L × m3/h = L/h（1 m3 = 1000 L, dose mL/L = dose L/m3）
        return defluor_dose * water_flow_m3_h
    elif basis == "mg_L_active":
        return pump_flow_from_mg_l(defluor_dose, water_flow_m3_h,
                                    cfg.DEFLUOR_STOCK_CONC_G_L, 1.0)
    raise ValueError(f"未知 DEFLUOR_DOSE_BASIS: {basis}")


def compute_pump_flows(recipe, influent_flow_m3_h: float) -> dict:
    if hasattr(recipe, 'pacl_dose_setpoint'):
        pacl = recipe.pacl_dose_setpoint
        deflu = recipe.defluor_dose_setpoint
    else:
        pacl = recipe.get("pacl_dose_setpoint", 0)
        deflu = recipe.get("defluor_dose_setpoint", 0)

    if influent_flow_m3_h <= 0:
        raise ValueError("缺少有效 influent_flow，无法换算计量泵流量")

    return {
        "pacl_pump_flow_l_h": round(pacl_pump_flow_l_h(pacl, influent_flow_m3_h), 2),
        "defluor_pump_flow_l_h": round(defluor_pump_flow_l_h(deflu, influent_flow_m3_h), 2),
        "pump_flow_unit": "L/h",
        "formula_basis": {
            "pacl": cfg.PACL_DOSE_BASIS,
            "defluor": cfg.DEFLUOR_DOSE_BASIS,
        },
        "assumptions": {
            "pacl_stock_conc_g_l": cfg.PACL_STOCK_CONC_G_L,
            "pacl_al_mass_fraction": cfg.PACL_AL_MASS_FRACTION,
            "defluor_stock_conc_g_l": cfg.DEFLUOR_STOCK_CONC_G_L,
            "defluor_density_kg_l": cfg.DEFLUOR_DENSITY_KG_L,
        },
    }
