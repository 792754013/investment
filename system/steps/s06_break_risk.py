from __future__ import annotations

from typing import List

from system.models import OpportunityScore, PipelineState


def apply(state: PipelineState) -> None:
    penalty = float(state.thresholds.get("break_risk_penalty", 0.3))
    constraint_map = {item.theme: item for item in state.constraints}
    adjusted: List[OpportunityScore] = []
    for score_item in state.scores:
        constraint = constraint_map.get(score_item.theme)
        break_risk = constraint.break_risk if constraint else 0.0
        score = max(0.0, score_item.score - break_risk * penalty)
        reason = f"破坏风险惩罚={penalty:.2f}"
        adjusted.append(OpportunityScore(theme=score_item.theme, score=score, reason=reason))
    state.scores = adjusted
