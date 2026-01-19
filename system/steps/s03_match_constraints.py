"""约束匹配：将主题映射到约束配置，生成约束快照。"""

from __future__ import annotations

from typing import List

from system.config_loader import load_constraints, load_themes
from system.models import ConstraintSnapshot, PipelineState


def apply(state: PipelineState) -> None:
    # 加载主题与约束配置，说明“业务映射”通常在配置中维护
    themes = load_themes().get("themes", {})
    constraints = load_constraints().get("constraints", {})
    snapshots: List[ConstraintSnapshot] = []
    for item in state.quality:
        # 每个主题对应一个约束 ID，再取约束指标
        theme_info = themes.get(item.theme, {})
        constraint_id = theme_info.get("constraint_id", "")
        constraint_info = constraints.get(constraint_id, {})
        health = float(constraint_info.get("health_score", 0.5))
        break_risk = float(constraint_info.get("break_risk", 0.5))
        reason = f"约束={constraint_id} 健康度={health:.2f}"
        snapshots.append(
            ConstraintSnapshot(
                theme=item.theme,
                constraint_id=constraint_id,
                health_score=health,
                break_risk=break_risk,
                reason=reason,
            )
        )
    # 将约束快照写入流程状态，便于风险门控使用
    state.constraints = snapshots
