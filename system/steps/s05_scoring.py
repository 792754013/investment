"""机会评分：综合需求质量与约束健康度，输出机会分。"""

from __future__ import annotations

from typing import List

from system.models import OpportunityScore, PipelineState


def apply(state: PipelineState) -> None:
    # 权重来自阈值配置，强调策略参数化
    demand_weight = float(state.thresholds.get("demand_weight", 0.6))
    constraint_weight = float(state.thresholds.get("constraint_weight", 0.4))
    # 先构造主题到质量的映射，加快后续查找
    quality_map = {item.theme: item for item in state.quality}
    scores: List[OpportunityScore] = []
    for snapshot in state.constraints:
        quality = quality_map.get(snapshot.theme)
        quality_score = quality.quality_score if quality else 0.0
        # 机会分 = 需求分 * 权重 + 约束健康度 * 权重
        score = quality_score * demand_weight + snapshot.health_score * constraint_weight
        reason = f"需求权重={demand_weight:.2f} 约束权重={constraint_weight:.2f}"
        scores.append(OpportunityScore(theme=snapshot.theme, score=score, reason=reason))
    # 写回评分结果
    state.scores = scores
