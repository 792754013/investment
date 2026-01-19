from __future__ import annotations

from typing import List

from system.models import DemandQuality, PipelineState


def apply(state: PipelineState) -> None:
    min_signal = float(state.thresholds.get("demand_signal_min", 0.3))
    quality: List[DemandQuality] = []
    for event in state.events:
        score = event.signal_strength
        passed = score >= min_signal
        reason = "需求信号达标" if passed else "需求信号不足"
        quality.append(DemandQuality(theme=event.theme, quality_score=score, passed=passed, reason=reason))
    state.quality = quality
