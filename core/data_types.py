"""共享数据类型 — 除氟预测与加药推荐服务"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"


@dataclass
class WaterQuality:
    influent_flow: float = 0.0
    influent_ph: float = 7.0
    conductivity: float = 0.0
    influent_f: float = 0.0
    pacl_dose: float = 0.0
    defluor_dose: float = 0.0
    pacl_tank_ph: float = 7.0
    defluor_tank_ph: float = 6.0
    recycle_flow: float = 0.0
    waste_flow: float = 0.0
    pam_dose: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in [
            "influent_flow", "influent_ph", "conductivity", "influent_f",
            "pacl_dose", "defluor_dose", "pacl_tank_ph", "defluor_tank_ph",
            "recycle_flow", "waste_flow", "pam_dose", "timestamp"]}


@dataclass
class DosingRecipe:
    pacl_dose_setpoint: float = 0.0
    defluor_dose_setpoint: float = 0.0

    @property
    def pacl_dose(self):
        return self.pacl_dose_setpoint

    @property
    def defluor_dose(self):
        return self.defluor_dose_setpoint


@dataclass
class PredictionResult:
    predicted_f: float = 0.0
    q05: float = 0.0
    q95: float = 0.0
    risk_level: str = "safe"
    model_used: str = "tabpfn"
    warnings: list = field(default_factory=list)
