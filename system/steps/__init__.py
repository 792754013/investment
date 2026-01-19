"""步骤包导出：集中暴露流程步骤，便于统一调度。"""

from system.steps import (
    s01_demand_scan,
    s02_demand_quality,
    s03_match_constraints,
    s04_risk_gate,
    s05_scoring,
    s06_break_risk,
    s07_theme_rank,
    s08_stage_detect,
    s09_entry,
    s10_stoploss,
    s11_takeprofit,
    s12_portfolio,
    s13_killswitch,
)

# __all__ 用于控制“from package import *”时的导出列表
__all__ = [
    "s01_demand_scan",
    "s02_demand_quality",
    "s03_match_constraints",
    "s04_risk_gate",
    "s05_scoring",
    "s06_break_risk",
    "s07_theme_rank",
    "s08_stage_detect",
    "s09_entry",
    "s10_stoploss",
    "s11_takeprofit",
    "s12_portfolio",
    "s13_killswitch",
]
