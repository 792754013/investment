"""周期阶段判定：结合宏观数据与覆盖配置识别市场阶段。"""

from __future__ import annotations

from typing import List

from system.config_loader import load_macro
from system.models import PipelineState, StageResult


def apply(state: PipelineState) -> None:
    # 加载宏观数据并筛选当日记录
    macro = load_macro()
    day_macro = macro[macro["date"].dt.date == state.date]
    if day_macro.empty:
        # 若当日没有宏观数据，用默认值保证流程可运行
        real_yield = 1.0
        inflation = 2.0
    else:
        real_yield = float(day_macro["REAL_YIELD"].iloc[0])
        inflation = float(day_macro["INFLATION"].iloc[0])
    stages: List[StageResult] = []
    # 先读取当天的人工覆盖配置，优先级最高
    override_for_day = state.overrides.get(state.date.isoformat(), {})
    for item in state.ranks:
        # 覆盖配置可强制指定阶段，用于教学与回测修正
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
    # 保存阶段判定结果
    state.stages = stages
