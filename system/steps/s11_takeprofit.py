from __future__ import annotations

from typing import List

from system.models import DecisionIntent, PipelineState


def apply(state: PipelineState) -> None:
    stage_map = {item.theme: item for item in state.stages}
    adjusted: List[DecisionIntent] = []
    for intent in state.intents:
        stage = stage_map.get(intent.theme).stage if intent.theme in stage_map else "early"
        if stage == "late" and intent.intent in {"ENTER", "ADD", "HOLD"}:
            adjusted.append(
                DecisionIntent(
                    theme=intent.theme,
                    intent="EXIT",
                    reason="阶段偏晚止盈",
                    score=intent.score,
                )
            )
        else:
            adjusted.append(intent)
    state.intents = adjusted
