"""全局配置 — 深度除氟预测与加药推荐服务

来源：中试原始数据表（净氟生态舱中试试验数据记录表）、上位机PLC交互数据。
已确认项以中试原始数据为准，仍标 TODO 的待甲方最终确认。
"""

import math
import os

# ═══════════════════════════════════════════════════════
# 一、模型字段
# ═══════════════════════════════════════════════════════

# MODEL_INPUT_COLS 是模型预测时允许使用的“输入变量”。
# 任何未来才知道的量、出水结果、人工判定结果，都不能放进输入列。
MODEL_INPUT_COLS = [
    # PLC DB19 寄存器 → 模型字段
    "influent_flow",     # DBD0  入水口流量
    "influent_ph",       # DBD4  入水PH值
    "conductivity",      # DBD8  入水电导率
    "influent_f",        # DBD12 入水氟化物浓度
    "pacl_dose",         # DBD16 混凝剂投加量
    "defluor_dose",      # DBD20 除氟剂投加量
    "pacl_tank_ph",      # DBD24 混凝剂投加池PH值
    "defluor_tank_ph",   # DBD28 除氟剂投加池PH值
    "recycle_flow",      # DBD32 回流流量
    "waste_flow",        # DBD36 排泥流量 (PLC称"剩余流量")
    "pam_dose",          # DBD40 PAM投加量
]

TARGET_COL = "effluent_f"

# 目标列是模型要预测的沉后出水氟，必须和输入列隔离，否则会形成“答案泄漏”。
assert TARGET_COL not in MODEL_INPUT_COLS, "effluent_f 不能作为模型输入"

OPTIONAL_INPUT_COLS = ["orp", "turbidity", "cod", "magnetic_dose"]

# TS_FEAT_COLS 会进入时序展开：滞后、滚动均值、变化率等都围绕这些字段生成。
TS_FEAT_COLS = MODEL_INPUT_COLS

# XGBoost 备用模型只用较少的基础特征，主要服务优化器快速筛选方案。
# 主预测仍以 TabPFN + 完整时序特征为准。
XGB_BASE_COLS = MODEL_INPUT_COLS + [
    "recycle_ratio", "waste_ratio", "pacl_mass_rate", "defluor_mass_rate",
]

# ═══════════════════════════════════════════════════════
# 二、控制线
# ═══════════════════════════════════════════════════════

TARGET_F = 0.9        # 安全控制线 mg/L
LIMIT_F = 1.0         # 排放红线 mg/L

PACL_RANGE = (50.0, 2000.0)      # mg/L 商品溶液，中试实际70-240，留足余量
DEFLUOR_RANGE = (0.1, 5.0)       # mL/L  中试实际 0.26-4.63 mL/L

# ═══════════════════════════════════════════════════════
# 三、成本 — 来自中试原始数据表
# ═══════════════════════════════════════════════════════

# 药剂价格 — 来源：中试原始数据表
PACL_PRICE_YUAN_T = 700.0          # 10% PAC 溶液 元/吨
DEFLUOR_A_PRICE_YUAN_T = 2600.0    # 除氟剂A 元/吨
DEFLUOR_B_PRICE_YUAN_T = 2500.0    # 除氟剂B 元/吨
DEFLUOR_PRICE_YUAN_T = DEFLUOR_A_PRICE_YUAN_T  # 默认用A，兼容旧引用
PAM_PRICE_YUAN_T = 11000.0         # PAM 元/吨
MAGNETIC_PRICE_YUAN_T = 3000.0     # 磁粉 元/吨
NAOH_PRICE_YUAN_T = 1200.0         # 30% 氢氧化钠 元/吨

DEFLUOR_DENSITY_KG_L = 1.4       # TODO: 推定值，待甲方确认
DEFLUOR_STOCK_MASS_FRACTION = 0.15  # 中试原始数据：除氟剂（15%溶液）
MAGNETIC_LOSS_RATE = 0.02        # TODO: 磁粉回收损耗率 待确认
INCLUDE_FIXED_CHEMICAL_COSTS = False
DEFAULT_PAM_DOSE_MG_L = 4.0  # 中试实测 3.77-4.0 mg/L（0.1-0.15‰溶液）
DEFAULT_MAGNETIC_DOSE_G_L = 0.15

# PAC 投加量单位（中试原始数据：PAC(10%溶液））
PACL_DOSE_BASIS = "mg_L_product"    # mg_L_product | mg_L_as_Al | mM_as_Al
PACL_STOCK_CONC_G_L = 100.0         # 10% PAC 药液浓度 g/L
PACL_AL_MASS_FRACTION = 0.117       # TODO: PAC 中 Al 质量分数 待确认

# 除氟剂投加量单位（中试原始数据：除氟剂（15%溶液））
DEFLUOR_DOSE_BASIS = "mL_L_stock"   # mL_L_stock | mg_L_active
DEFLUOR_STOCK_CONC_G_L = (
    DEFLUOR_DENSITY_KG_L * 1000 * DEFLUOR_STOCK_MASS_FRACTION
)                                      # 15% 除氟剂配药液，按密度折算为 g/L

# ═══════════════════════════════════════════════════════
# 四、优化参数
# ═══════════════════════════════════════════════════════

COST_WEIGHT = 1.0
QUALITY_WEIGHT = 3.0
TARGET_VIOLATION_PENALTY = 100.0
LIMIT_VIOLATION_PENALTY = 10000.0

PACL_GRID_POINTS = 40    # 生产建议 40，调试可降到 10
DEFLUOR_GRID_POINTS = 40
TABPFN_VALIDATE_TOP_N = 30  # XGBoost 粗筛后 TabPFN 复核的候选数

ALLOW_FALLBACK_PREDICTION = False   # True 时允许无 TabPFN 时用 XGBoost 或规则
FAST_OPTIMIZER_MODEL = "xgboost"
# "main"：优化用主模型，与 /predict 一致（40×40小时级延迟）
# "xgboost"：优化用 XGBoost 批量预测，毫秒级（推荐默认）

# ═══════════════════════════════════════════════════════
# 五、模型参数
# ═══════════════════════════════════════════════════════

LAG_STEPS = (1, 2, 3, 5)
ROLLING_WINDOWS = (3, 5)
MIN_HISTORY = 10
DEVICE = os.environ.get("MODEL_DEVICE", "auto")
RANDOM_SEED = 42
TABPFN_MODEL_CACHE_DIR = os.environ.get("TABPFN_MODEL_CACHE_DIR", "models/tabpfn_cache")
OUTPUT_DELAY_STEPS = 0   
# 当前使用 FEATURE_DELAY_STEPS 按变量处理 HRT；
# 池体停留时间，单位 min。
# 这些时间目前用于近似“某个采样点/投加点到沉后出水”的影响延迟。
# 后续如果甲方给出更准确的采样间隔、池容、流量变化，应优先修正这里。
TANK_HRT_MIN = {
    "influent_tank": 4,
    "defluor_reactor_1": 12,
    "defluor_reactor_2": 10,
    "defluor_reactor_3": 20,
    "magnetic_reactor": 4,
    "flocculation_tank": 5,
    "sedimentation_tank": 10,
}

MODEL_SAMPLE_INTERVAL_MIN = 10


def hrt_minutes_to_steps(minutes: float,
                         sample_interval_min: float = MODEL_SAMPLE_INTERVAL_MIN) -> int:
    """把分钟级 HRT 换算成数据表中的行数延迟。

    例如采样间隔 10 分钟，HRT=42 分钟时需要向前取 ceil(42/10)=5 行。
    这里用向上取整是偏保守的做法，避免把尚未真正到达出水端的影响提前使用。
    """
    if minutes <= 0:
        return 0
    if sample_interval_min <= 0:
        raise ValueError("sample_interval_min must be > 0")
    return max(1, math.ceil(minutes / sample_interval_min))


POINT_TO_EFFLUENT_HRT_MIN = {
    # 不同变量所在位置不同，所以不能把所有字段统一延迟。
    # 例如进水指标要经过全流程才影响沉后出水；除氟剂从其投加池之后才开始累计。
    "influent": sum(TANK_HRT_MIN.values()),
    "pacl": (
        TANK_HRT_MIN["defluor_reactor_1"] +
        TANK_HRT_MIN["defluor_reactor_2"] +
        TANK_HRT_MIN["defluor_reactor_3"] +
        TANK_HRT_MIN["magnetic_reactor"] +
        TANK_HRT_MIN["flocculation_tank"] +
        TANK_HRT_MIN["sedimentation_tank"]
    ),
    "defluor": (
        TANK_HRT_MIN["defluor_reactor_2"] +
        TANK_HRT_MIN["defluor_reactor_3"] +
        TANK_HRT_MIN["magnetic_reactor"] +
        TANK_HRT_MIN["flocculation_tank"] +
        TANK_HRT_MIN["sedimentation_tank"]
    ),
    "magnetic": (
        TANK_HRT_MIN["magnetic_reactor"] +
        TANK_HRT_MIN["flocculation_tank"] +
        TANK_HRT_MIN["sedimentation_tank"]
    ),
    "pam": (
        TANK_HRT_MIN["flocculation_tank"] +
        TANK_HRT_MIN["sedimentation_tank"]
    ),
    "sedimentation": TANK_HRT_MIN["sedimentation_tank"],
}

FEATURE_DELAY_MINUTES = {
    # 每个字段映射到它所在位置到最终沉后出水的近似停留时间。
    # 人工 review 时重点确认：字段对应的现场测点/投加点是否放在了正确的位置。
    "influent_flow": POINT_TO_EFFLUENT_HRT_MIN["influent"],
    "influent_ph": POINT_TO_EFFLUENT_HRT_MIN["influent"],
    "conductivity": POINT_TO_EFFLUENT_HRT_MIN["influent"],
    "influent_f": POINT_TO_EFFLUENT_HRT_MIN["influent"],
    "pacl_dose": POINT_TO_EFFLUENT_HRT_MIN["pacl"],
    "defluor_dose": POINT_TO_EFFLUENT_HRT_MIN["defluor"],
    "pacl_tank_ph": POINT_TO_EFFLUENT_HRT_MIN["pacl"],
    "defluor_tank_ph": POINT_TO_EFFLUENT_HRT_MIN["defluor"],
    "recycle_flow": POINT_TO_EFFLUENT_HRT_MIN["magnetic"],
    "waste_flow": POINT_TO_EFFLUENT_HRT_MIN["sedimentation"],
    "pam_dose": POINT_TO_EFFLUENT_HRT_MIN["pam"],
}

# 每个变量独立延迟到沉后出水，单位是“采样步数”。
# 已经使用 FEATURE_DELAY_STEPS 时，不要再打开全局 OUTPUT_DELAY_STEPS，
# 否则同一段 HRT 会被重复计算，时序样本会错位。
FEATURE_DELAY_STEPS = {
    name: hrt_minutes_to_steps(minutes)
    for name, minutes in FEATURE_DELAY_MINUTES.items()
}

# ═══════════════════════════════════════════════════════
# 六、日志
# ═══════════════════════════════════════════════════════

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_DIR = os.environ.get("LOG_DIR", "logs")
