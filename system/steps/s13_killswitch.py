"""终局决策：加入宏观 Kill Switch 并生成最终决策。"""

from __future__ import annotations

from typing import List

from system.config_loader import load_macro
from system.models import DecisionResult, PipelineState


def apply(state: PipelineState) -> None:
    # Kill Switch 阈值用于在极端风险时强制退出
    killswitch_level = float(state.thresholds.get("killswitch_level", 0.85))
    macro = load_macro()
    day_macro = macro[macro["date"].dt.date == state.date]
    # 读取地缘风险指数，缺失则默认 0
    geo_risk = float(day_macro["GEO_RISK_INDEX"].iloc[0]) if not day_macro.empty else 0.0
    # 构造常用映射，减少重复遍历
    stage_map = {item.theme: item for item in state.stages}
    constraint_map = {item.theme: item for item in state.constraints}
    decisions: List[DecisionResult] = []
    for intent in state.intents:
        # 默认继承前面步骤的意图与理由
        final_intent = intent.intent
        reason = intent.reason
        stage_info = stage_map.get(intent.theme)
        # 若被覆盖配置影响，说明原因中补充提示
        if stage_info and stage_info.reason == "stage_override":
            reason = f"{reason}; stage_override"
        # 触发 Kill Switch 则统一退出
        if geo_risk >= killswitch_level:
            final_intent = "EXIT"
            reason = "地缘风险触发Kill Switch"
        stage = stage_info.stage if stage_info else "early"
        constraint = constraint_map.get(intent.theme)
        constraint_id = constraint.constraint_id if constraint else ""
        break_risk = constraint.break_risk if constraint else 0.0
        # 汇总为最终决策输出
        decisions.append(
            DecisionResult(
                date=state.date,
                product=state.product,
                theme=intent.theme,
                intent=final_intent,
                reason=reason,
                stage=stage,
                score=intent.score,
                constraint_id=constraint_id,
                break_risk=break_risk,
            )
        )
    # 保存决策结果
    state.decisions = decisions
