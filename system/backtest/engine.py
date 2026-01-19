from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

import pandas as pd

from system.backtest.metrics import equity_stats, max_drawdown
from system.config_loader import (
    load_assets,
    load_prices,
    load_stage_overrides,
    list_trading_dates,
    load_thresholds,
)
from system.models import BacktestSummary, BacktestTrade, DecisionResult, EquityPoint
from system.pipeline.runner import run_pipeline
from system.utils import ensure_dir, save_json


def _price_lookup(prices: pd.DataFrame, run_date: date) -> pd.DataFrame:
    return prices[prices["date"].dt.date == run_date].copy()


def _action_from_intent(intent: str) -> str:
    return intent


def run_backtest(
    product: str,
    start: date,
    end: date,
    stage_overrides_path: str | None,
    output_dir: str,
) -> Tuple[List[BacktestTrade], List[EquityPoint], BacktestSummary]:
    thresholds = load_thresholds().get("thresholds", {})
    initial_cash = float(thresholds.get("initial_cash", 1_000_000.0))
    max_positions = int(thresholds.get("max_positions", 3))
    slippage = float(thresholds.get("slippage", 0.0005))
    fee = float(thresholds.get("fee", 0.0003))

    overrides = load_stage_overrides(stage_overrides_path)
    assets = load_assets()
    prices = load_prices()

    cash = initial_cash
    positions: Dict[str, float] = {}
    trades: List[BacktestTrade] = []
    equity_points: List[EquityPoint] = []

    dates = list_trading_dates(start, end)
    for run_date in dates:
        decisions = run_pipeline(product, run_date, overrides=overrides)
        day_prices = _price_lookup(prices, run_date)
        theme_asset_map = assets[assets["product"] == product].set_index("theme")["asset_id"].to_dict()

        for decision in decisions:
            asset_id = theme_asset_map.get(decision.theme)
            if asset_id is None:
                continue
            price_row = day_prices[day_prices["asset_id"] == asset_id]
            if price_row.empty:
                continue
            price = float(price_row["close"].iloc[0])
            intent = decision.intent
            current_qty = positions.get(asset_id, 0.0)
            action = _action_from_intent(intent)
            if action == "ENTER" and current_qty == 0.0:
                slots = max_positions - len([qty for qty in positions.values() if qty > 0])
                if slots <= 0:
                    continue
                allocation = cash / slots
                quantity = allocation / price
                trade_cost = quantity * price * (1 + slippage + fee)
                cash -= trade_cost
                positions[asset_id] = current_qty + quantity
                trades.append(
                    BacktestTrade(
                        date=run_date,
                        asset_id=asset_id,
                        action="BUY",
                        price=price,
                        quantity=quantity,
                        cost=trade_cost,
                        reason=decision.reason,
                    )
                )
            elif action == "ADD" and current_qty > 0.0:
                allocation = cash * 0.5
                quantity = allocation / price
                trade_cost = quantity * price * (1 + slippage + fee)
                cash -= trade_cost
                positions[asset_id] = current_qty + quantity
                trades.append(
                    BacktestTrade(
                        date=run_date,
                        asset_id=asset_id,
                        action="BUY",
                        price=price,
                        quantity=quantity,
                        cost=trade_cost,
                        reason=decision.reason,
                    )
                )
            elif action == "REDUCE" and current_qty > 0.0:
                quantity = current_qty * 0.5
                trade_value = quantity * price * (1 - slippage - fee)
                cash += trade_value
                positions[asset_id] = current_qty - quantity
                trades.append(
                    BacktestTrade(
                        date=run_date,
                        asset_id=asset_id,
                        action="SELL",
                        price=price,
                        quantity=quantity,
                        cost=trade_value,
                        reason=decision.reason,
                    )
                )
            elif action == "EXIT" and current_qty > 0.0:
                quantity = current_qty
                trade_value = quantity * price * (1 - slippage - fee)
                cash += trade_value
                positions[asset_id] = 0.0
                trades.append(
                    BacktestTrade(
                        date=run_date,
                        asset_id=asset_id,
                        action="SELL",
                        price=price,
                        quantity=quantity,
                        cost=trade_value,
                        reason=decision.reason,
                    )
                )

        day_value = 0.0
        for asset_id, qty in positions.items():
            if qty <= 0:
                continue
            price_row = day_prices[day_prices["asset_id"] == asset_id]
            if price_row.empty:
                continue
            price = float(price_row["close"].iloc[0])
            day_value += qty * price
        equity = cash + day_value
        equity_points.append(EquityPoint(date=run_date, equity=equity, cash=cash, positions_value=day_value))

    final_equity = equity_points[-1].equity if equity_points else initial_cash
    total_return = (final_equity - initial_cash) / initial_cash if initial_cash else 0.0
    max_dd = max_drawdown(equity_points)
    stats = equity_stats(equity_points)
    calmar = stats["annualized_return"] / max_dd if max_dd > 0 else 0.0

    summary = BacktestSummary(
        start=start,
        end=end,
        initial_cash=initial_cash,
        final_equity=final_equity,
        total_return=total_return,
        annualized_return=float(stats["annualized_return"]),
        annualized_volatility=float(stats["annualized_volatility"]),
        sharpe=float(stats["sharpe"]),
        sortino=float(stats["sortino"]),
        calmar=calmar,
        max_drawdown=max_dd,
        trade_count=len(trades),
        trading_days=int(stats["trading_days"]),
        win_rate=float(stats["win_rate"]),
        positive_days=int(stats["positive_days"]),
        negative_days=int(stats["negative_days"]),
        flat_days=int(stats["flat_days"]),
        best_day=float(stats["best_day"]),
        worst_day=float(stats["worst_day"]),
        profit_factor=float(stats["profit_factor"]),
        avg_daily_return=float(stats["avg_daily_return"]),
    )

    ensure_dir(output_dir)
    trades_df = pd.DataFrame([t.dict() for t in trades])
    equity_df = pd.DataFrame([p.dict() for p in equity_points])
    trades_df.to_csv(f"{output_dir}/trades.csv", index=False)
    equity_df.to_csv(f"{output_dir}/equity.csv", index=False)
    save_json(f"{output_dir}/summary.json", summary.dict())

    return trades, equity_points, summary
