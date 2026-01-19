from __future__ import annotations

from typing import List

from system.models import DecisionIntent, PipelineState


def apply(state: PipelineState) -> None:
    limit = float(state.thresholds.get("break_risk_stop", 0.7))
    constraint_map = {item.theme: item for item in state.constraints}
    adjusted: List[DecisionIntent] = []
    for intent in state.intents:
        constraint = constraint_map.get(intent.theme)
        break_risk = constraint.break_risk if constraint else 0.0
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
    state.intents = adjusted
