"""风险门控：根据约束健康度过滤掉高风险主题。"""

from __future__ import annotations

from typing import List

from system.models import ConstraintSnapshot, PipelineState


def apply(state: PipelineState) -> None:
    # 风险门槛来自阈值配置，教学上强调“阈值可调”
    min_health = float(state.thresholds.get("constraint_min_health", 0.4))
    filtered: List[ConstraintSnapshot] = []
    for snapshot in state.constraints:
        # 仅保留健康度足够的约束
        if snapshot.health_score >= min_health:
            filtered.append(snapshot)
    # 更新状态，后续评分步骤仅处理安全主题
    state.constraints = filtered
