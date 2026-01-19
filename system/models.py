from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DemandEvent(BaseModel):
    theme: str
    signal_strength: float
    reason: str


class DemandQuality(BaseModel):
    theme: str
    quality_score: float
    passed: bool
    reason: str


class ConstraintSnapshot(BaseModel):
    theme: str
    constraint_id: str
    health_score: float
    break_risk: float
    reason: str


class OpportunityScore(BaseModel):
    theme: str
    score: float
    reason: str


class ThemeRank(BaseModel):
    theme: str
    rank: int
    score: float


class StageResult(BaseModel):
    theme: str
    stage: str
    reason: str


class DecisionIntent(BaseModel):
    theme: str
    intent: str
    reason: str
    score: float


class DecisionResult(BaseModel):
    date: date
    product: str
    theme: str
    intent: str
    reason: str
    stage: str
    score: float
    constraint_id: str
    break_risk: float


class RunResult(BaseModel):
    run_id: str
    date: date
    product: str
    results: List[DecisionResult]


class MonitorItem(BaseModel):
    theme: str
    constraint_id: str
    assets: List[str]


class MonitorPlan(BaseModel):
    product: str
    items: List[MonitorItem]


class BacktestTrade(BaseModel):
    date: date
    asset_id: str
    action: str
    price: float
    quantity: float
    cost: float
    reason: str


class EquityPoint(BaseModel):
    date: date
    equity: float
    cash: float
    positions_value: float


class BacktestSummary(BaseModel):
    start: date
    end: date
    initial_cash: float = 0.0
    final_equity: float = 0.0
    total_return: float
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float
    trade_count: int
    trading_days: int = 0
    win_rate: float = 0.0
    positive_days: int = 0
    negative_days: int = 0
    flat_days: int = 0
    best_day: float = 0.0
    worst_day: float = 0.0
    profit_factor: float = 0.0
    avg_daily_return: float = 0.0


class PipelineState(BaseModel):
    product: str
    date: date
    thresholds: Dict[str, float]
    events: List[DemandEvent] = Field(default_factory=list)
    quality: List[DemandQuality] = Field(default_factory=list)
    constraints: List[ConstraintSnapshot] = Field(default_factory=list)
    scores: List[OpportunityScore] = Field(default_factory=list)
    ranks: List[ThemeRank] = Field(default_factory=list)
    stages: List[StageResult] = Field(default_factory=list)
    intents: List[DecisionIntent] = Field(default_factory=list)
    decisions: List[DecisionResult] = Field(default_factory=list)
    overrides: Dict[str, Dict[str, str]] = Field(default_factory=dict)

    def theme_map(self) -> Dict[str, Dict[str, Optional[BaseModel]]]:
        mapping: Dict[str, Dict[str, Optional[BaseModel]]] = {}
        for item in self.events:
            mapping.setdefault(item.theme, {})["event"] = item
        for item in self.quality:
            mapping.setdefault(item.theme, {})["quality"] = item
        for item in self.constraints:
            mapping.setdefault(item.theme, {})["constraint"] = item
        for item in self.scores:
            mapping.setdefault(item.theme, {})["score"] = item
        for item in self.ranks:
            mapping.setdefault(item.theme, {})["rank"] = item
        for item in self.stages:
            mapping.setdefault(item.theme, {})["stage"] = item
        for item in self.intents:
            mapping.setdefault(item.theme, {})["intent"] = item
        return mapping
