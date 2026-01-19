"""止损控制：根据破坏风险调整入场意图。"""

from __future__ import annotations

from typing import List

from system.models import DecisionIntent, PipelineState


def apply(state: PipelineState) -> None:
    # 破坏风险上限，超过则触发减仓
    limit = float(state.thresholds.get("break_risk_stop", 0.7))
    # 将主题与约束信息对应，便于取破坏风险
    constraint_map = {item.theme: item for item in state.constraints}
    adjusted: List[DecisionIntent] = []
    for intent in state.intents:
        constraint = constraint_map.get(intent.theme)
        break_risk = constraint.break_risk if constraint else 0.0
        # 满足入场且风险过高时，输出 REDUCE
        if intent.intent == "ENTER" and break_risk >= limit:
            adjusted.append(
                DecisionIntent(
                    theme=intent.theme,
                    intent="REDUCE",
                    reason="破坏风险过高",
                    score=intent.score,
                )
            )
        else:
            adjusted.append(intent)
    # 更新意图，后续止盈和组合步骤使用
    state.intents = adjusted
