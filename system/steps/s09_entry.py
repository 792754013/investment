from __future__ import annotations

from typing import List

from system.models import DecisionIntent, PipelineState


def apply(state: PipelineState) -> None:
    top_n = int(state.thresholds.get("top_theme_n", 3))
    min_score = float(state.thresholds.get("min_score", 0.4))
    stage_map = {item.theme: item for item in state.stages}
    intents: List[DecisionIntent] = []
    for rank in state.ranks:
        stage = stage_map.get(rank.theme).stage if rank.theme in stage_map else "early"
        if rank.rank <= top_n and rank.score >= min_score and stage != "late":
            intent = "ENTER"
            reason = "主题排名靠前"
        else:
            intent = "HOLD"
            reason = "排名或阶段不满足"
        intents.append(DecisionIntent(theme=rank.theme, intent=intent, reason=reason, score=rank.score))
    state.intents = intents
