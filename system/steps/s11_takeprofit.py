"""止盈控制：当阶段进入后期时倾向退出。"""

from __future__ import annotations

from typing import List

from system.models import DecisionIntent, PipelineState


def apply(state: PipelineState) -> None:
    # 先建立主题到阶段的映射，避免重复遍历
    stage_map = {item.theme: item for item in state.stages}
    adjusted: List[DecisionIntent] = []
    for intent in state.intents:
        stage = stage_map.get(intent.theme).stage if intent.theme in stage_map else "early"
        # 晚期阶段倾向获利了结
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
    # 保存调整后的意图
    state.intents = adjusted
