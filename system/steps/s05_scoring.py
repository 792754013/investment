from __future__ import annotations

from typing import List

from system.models import OpportunityScore, PipelineState


def apply(state: PipelineState) -> None:
    demand_weight = float(state.thresholds.get("demand_weight", 0.6))
    constraint_weight = float(state.thresholds.get("constraint_weight", 0.4))
    quality_map = {item.theme: item for item in state.quality}
    scores: List[OpportunityScore] = []
    for snapshot in state.constraints:
        quality = quality_map.get(snapshot.theme)
        quality_score = quality.quality_score if quality else 0.0
        score = quality_score * demand_weight + snapshot.health_score * constraint_weight
        reason = f"需求权重={demand_weight:.2f} 约束权重={constraint_weight:.2f}"
        scores.append(OpportunityScore(theme=snapshot.theme, score=score, reason=reason))
    state.scores = scores
