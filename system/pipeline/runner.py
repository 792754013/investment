"""流程运行器：按顺序执行策略步骤并返回决策。"""

from __future__ import annotations

from datetime import date
from typing import Dict, List

from system.config_loader import load_thresholds
from system.models import DecisionResult, PipelineState
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


def run_pipeline(product: str, run_date: date, overrides: Dict[str, Dict[str, str]] | None = None) -> List[DecisionResult]:
    # 加载全局阈值配置并初始化流程状态
    thresholds = load_thresholds().get("thresholds", {})
    state = PipelineState(product=product, date=run_date, thresholds=thresholds, overrides=overrides or {})
    # 依次执行每个步骤，构成完整决策流水线
    s01_demand_scan.apply(state)
    s02_demand_quality.apply(state)
    s03_match_constraints.apply(state)
    s04_risk_gate.apply(state)
    s05_scoring.apply(state)
    s06_break_risk.apply(state)
    s07_theme_rank.apply(state)
    s08_stage_detect.apply(state)
    s09_entry.apply(state)
    s10_stoploss.apply(state)
    s11_takeprofit.apply(state)
    s12_portfolio.apply(state)
    s13_killswitch.apply(state)
    # 返回最终决策列表
    return state.decisions
