"""需求质量评估：将需求事件信号与阈值比对，输出是否达标。"""

from __future__ import annotations

from typing import List

from system.models import DemandQuality, PipelineState


def apply(state: PipelineState) -> None:
    # 从阈值配置中取最小需求信号，默认值用于教学示例
    min_signal = float(state.thresholds.get("demand_signal_min", 0.3))
    quality: List[DemandQuality] = []
    for event in state.events:
        # 这里用信号强度本身作为质量分数，便于理解
        score = event.signal_strength
        passed = score >= min_signal
        reason = "需求信号达标" if passed else "需求信号不足"
        quality.append(DemandQuality(theme=event.theme, quality_score=score, passed=passed, reason=reason))
    # 保存需求质量结果，供后续约束匹配使用
    state.quality = quality
