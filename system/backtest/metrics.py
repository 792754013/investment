from __future__ import annotations

from typing import List

from system.models import EquityPoint


def max_drawdown(points: List[EquityPoint]) -> float:
    if not points:
        return 0.0
    peak = points[0].equity
    max_dd = 0.0
    for point in points:
        if point.equity > peak:
            peak = point.equity
        drawdown = (peak - point.equity) / peak if peak else 0.0
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd
