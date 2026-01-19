"""组合控制：限制同一约束下的仓位暴露数量。"""

from __future__ import annotations

from collections import Counter
from typing import List

from system.models import DecisionIntent, PipelineState


def apply(state: PipelineState) -> None:
    # 每个约束允许的最大持仓数量
    max_exposure = int(state.thresholds.get("max_constraint_exposure", 2))
    # 建立主题到约束 ID 的映射
    constraint_map = {item.theme: item.constraint_id for item in state.constraints}
    exposure_counter = Counter()
    for intent in state.intents:
        # 统计每个约束下的入场/加仓数量
        if intent.intent in {"ENTER", "ADD"}:
            constraint_id = constraint_map.get(intent.theme, "")
            exposure_counter[constraint_id] += 1
    adjusted: List[DecisionIntent] = []
    for intent in state.intents:
        constraint_id = constraint_map.get(intent.theme, "")
        # 超过暴露上限时，将入场意图改为减仓
        if exposure_counter.get(constraint_id, 0) > max_exposure and intent.intent in {"ENTER", "ADD"}:
            adjusted.append(
                DecisionIntent(
                    theme=intent.theme,
                    intent="REDUCE",
                    reason="组合约束暴露超限",
                    score=intent.score,
                )
            )
        else:
            adjusted.append(intent)
    # 保存组合调整后的意图
    state.intents = adjusted
