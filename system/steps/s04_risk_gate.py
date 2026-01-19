from __future__ import annotations

from typing import List

from system.models import ConstraintSnapshot, PipelineState


def apply(state: PipelineState) -> None:
    min_health = float(state.thresholds.get("constraint_min_health", 0.4))
    filtered: List[ConstraintSnapshot] = []
    for snapshot in state.constraints:
        if snapshot.health_score >= min_health:
            filtered.append(snapshot)
    state.constraints = filtered
