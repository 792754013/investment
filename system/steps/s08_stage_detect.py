from __future__ import annotations

from typing import List

from system.config_loader import load_macro
from system.models import PipelineState, StageResult


def apply(state: PipelineState) -> None:
    macro = load_macro()
    day_macro = macro[macro["date"].dt.date == state.date]
    if day_macro.empty:
        real_yield = 1.0
        inflation = 2.0
    else:
        real_yield = float(day_macro["REAL_YIELD"].iloc[0])
        inflation = float(day_macro["INFLATION"].iloc[0])
    stages: List[StageResult] = []
    override_for_day = state.overrides.get(state.date.isoformat(), {})
    for item in state.ranks:
        if item.theme in override_for_day:
            stage = override_for_day[item.theme]
            reason = "stage_override"
        elif real_yield > 1.5:
            stage = "late"
            reason = "实际利率偏高"
        elif inflation > 3.0:
            stage = "mid"
            reason = "通胀抬升"
        else:
            stage = "early"
            reason = "阶段偏早"
        stages.append(StageResult(theme=item.theme, stage=stage, reason=reason))
    state.stages = stages
