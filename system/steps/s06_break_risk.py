"""破坏风险调整：将约束破坏风险作为惩罚项降低机会分。"""

from __future__ import annotations

from typing import List

from system.models import OpportunityScore, PipelineState


def apply(state: PipelineState) -> None:
    # 惩罚系数来自阈值配置，系数越大惩罚越重
    penalty = float(state.thresholds.get("break_risk_penalty", 0.3))
    # 构造主题到约束的映射，便于查询破坏风险
    constraint_map = {item.theme: item for item in state.constraints}
    adjusted: List[OpportunityScore] = []
    for score_item in state.scores:
        constraint = constraint_map.get(score_item.theme)
        break_risk = constraint.break_risk if constraint else 0.0
        # 机会分扣减破坏风险惩罚，并且不允许小于 0
        score = max(0.0, score_item.score - break_risk * penalty)
        reason = f"破坏风险惩罚={penalty:.2f}"
        adjusted.append(OpportunityScore(theme=score_item.theme, score=score, reason=reason))
    # 更新为调整后的评分
    state.scores = adjusted
