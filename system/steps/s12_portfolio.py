from __future__ import annotations

from collections import Counter
from typing import List

from system.models import DecisionIntent, PipelineState


def apply(state: PipelineState) -> None:
    max_exposure = int(state.thresholds.get("max_constraint_exposure", 2))
    constraint_map = {item.theme: item.constraint_id for item in state.constraints}
    exposure_counter = Counter()
    for intent in state.intents:
        if intent.intent in {"ENTER", "ADD"}:
            constraint_id = constraint_map.get(intent.theme, "")
            exposure_counter[constraint_id] += 1
    adjusted: List[DecisionIntent] = []
    for intent in state.intents:
        constraint_id = constraint_map.get(intent.theme, "")
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
    state.intents = adjusted
