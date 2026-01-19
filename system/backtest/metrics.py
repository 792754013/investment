"""回测指标计算：最大回撤与收益统计。"""

from __future__ import annotations

import math
from typing import Dict, List

from system.models import EquityPoint


def max_drawdown(points: List[EquityPoint]) -> float:
    # 最大回撤：从峰值到谷底的最大跌幅
    if not points:
        return 0.0
    peak = points[0].equity
    max_dd = 0.0
    for point in points:
        # 更新峰值并计算当前回撤
        if point.equity > peak:
            peak = point.equity
        drawdown = (peak - point.equity) / peak if peak else 0.0
        if drawdown > max_dd:
            max_dd = drawdown
    return max_dd


def equity_stats(points: List[EquityPoint]) -> Dict[str, float | int]:
    # 计算日收益序列与常见绩效指标
    trading_days = len(points)
    if trading_days < 2:
        # 数据不足时返回零值指标，避免除零错误
        return {
            "trading_days": trading_days,
            "avg_daily_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "positive_days": 0,
            "negative_days": 0,
            "flat_days": 0,
            "best_day": 0.0,
            "worst_day": 0.0,
            "profit_factor": 0.0,
        }

    returns: List[float] = []
    for idx in range(1, trading_days):
        prev_equity = points[idx - 1].equity
        curr_equity = points[idx].equity
        if prev_equity == 0:
            returns.append(0.0)
        else:
            returns.append((curr_equity - prev_equity) / prev_equity)

    avg_daily_return = sum(returns) / len(returns) if returns else 0.0
    # 方差与波动率计算：使用样本方差
    variance = 0.0
    if len(returns) > 1:
        variance = sum((value - avg_daily_return) ** 2 for value in returns) / (len(returns) - 1)
    daily_vol = math.sqrt(variance)
    annualized_vol = daily_vol * math.sqrt(252)

    downside_returns = [value for value in returns if value < 0.0]
    # 下行波动用于 Sortino 计算
    downside_variance = 0.0
    if len(downside_returns) > 1:
        downside_mean = sum(downside_returns) / len(downside_returns)
        downside_variance = sum((value - downside_mean) ** 2 for value in downside_returns) / (
            len(downside_returns) - 1
        )
    downside_vol = math.sqrt(downside_variance)

    sharpe = avg_daily_return / daily_vol * math.sqrt(252) if daily_vol > 0 else 0.0
    sortino = avg_daily_return / downside_vol * math.sqrt(252) if downside_vol > 0 else 0.0

    starting_equity = points[0].equity
    ending_equity = points[-1].equity
    annualized_return = 0.0
    if starting_equity > 0 and returns:
        # 年化收益率按 252 交易日换算
        annualized_return = (ending_equity / starting_equity) ** (252 / len(returns)) - 1

    positive_days = sum(1 for value in returns if value > 0.0)
    negative_days = sum(1 for value in returns if value < 0.0)
    flat_days = len(returns) - positive_days - negative_days

    positive_sum = sum(value for value in returns if value > 0.0)
    negative_sum = sum(value for value in returns if value < 0.0)
    # 收益因子 = 正收益总和 / 负收益绝对值总和
    profit_factor = positive_sum / abs(negative_sum) if negative_sum < 0 else 0.0

    return {
        "trading_days": trading_days,
        "avg_daily_return": avg_daily_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "win_rate": positive_days / len(returns) if returns else 0.0,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "flat_days": flat_days,
        "best_day": max(returns) if returns else 0.0,
        "worst_day": min(returns) if returns else 0.0,
        "profit_factor": profit_factor,
    }
