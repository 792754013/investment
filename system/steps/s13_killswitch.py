from __future__ import annotations

from typing import List

from system.config_loader import load_macro
from system.models import DecisionResult, PipelineState


def apply(state: PipelineState) -> None:
    killswitch_level = float(state.thresholds.get("killswitch_level", 0.85))
    macro = load_macro()
    day_macro = macro[macro["date"].dt.date == state.date]
    geo_risk = float(day_macro["GEO_RISK_INDEX"].iloc[0]) if not day_macro.empty else 0.0
    stage_map = {item.theme: item for item in state.stages}
    constraint_map = {item.theme: item for item in state.constraints}
    decisions: List[DecisionResult] = []
    for intent in state.intents:
        final_intent = intent.intent
        reason = intent.reason
        stage_info = stage_map.get(intent.theme)
        if stage_info and stage_info.reason == "stage_override":
            reason = f"{reason}; stage_override"
        if geo_risk >= killswitch_level:
            final_intent = "EXIT"
            reason = "地缘风险触发Kill Switch"
        stage = stage_info.stage if stage_info else "early"
        constraint = constraint_map.get(intent.theme)
        constraint_id = constraint.constraint_id if constraint else ""
        break_risk = constraint.break_risk if constraint else 0.0
        decisions.append(
            DecisionResult(
                date=state.date,
                product=state.product,
                theme=intent.theme,
                intent=final_intent,
                reason=reason,
                stage=stage,
                score=intent.score,
                constraint_id=constraint_id,
                break_risk=break_risk,
            )
        )
    state.decisions = decisions
