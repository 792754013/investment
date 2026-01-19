"""入场意图判定：结合排名、分数与阶段给出进入或观望。"""

from __future__ import annotations

from typing import List

from system.models import DecisionIntent, PipelineState


def apply(state: PipelineState) -> None:
    # 读取策略阈值：取前 N 名且分数达标
    top_n = int(state.thresholds.get("top_theme_n", 3))
    min_score = float(state.thresholds.get("min_score", 0.4))
    # 构造主题到阶段的映射，便于判断是否处于晚期阶段
    stage_map = {item.theme: item for item in state.stages}
    intents: List[DecisionIntent] = []
    for rank in state.ranks:
        stage = stage_map.get(rank.theme).stage if rank.theme in stage_map else "early"
        # 满足排名、分数与阶段条件则给出 ENTER
        if rank.rank <= top_n and rank.score >= min_score and stage != "late":
            intent = "ENTER"
            reason = "主题排名靠前"
        else:
            intent = "HOLD"
            reason = "排名或阶段不满足"
        intents.append(DecisionIntent(theme=rank.theme, intent=intent, reason=reason, score=rank.score))
    # 保存交易意图，供止损/止盈步骤复用
    state.intents = intents
