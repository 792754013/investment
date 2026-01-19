"""数据模型：用 Pydantic 描述流程中的结构化数据。"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DemandEvent(BaseModel):
    """需求事件：记录主题与需求信号。"""

    theme: str
    signal_strength: float
    reason: str


class DemandQuality(BaseModel):
    """需求质量评估结果。"""

    theme: str
    quality_score: float
    passed: bool
    reason: str


class ConstraintSnapshot(BaseModel):
    """约束快照：主题对应的风险约束指标。"""

    theme: str
    constraint_id: str
    health_score: float
    break_risk: float
    reason: str


class OpportunityScore(BaseModel):
    """机会评分结果。"""

    theme: str
    score: float
    reason: str


class ThemeRank(BaseModel):
    """主题排序结果。"""

    theme: str
    rank: int
    score: float


class StageResult(BaseModel):
    """市场阶段判定结果。"""

    theme: str
    stage: str
    reason: str


class DecisionIntent(BaseModel):
    """阶段性交易意图（未经过终局开关）。"""

    theme: str
    intent: str
    reason: str
    score: float


class DecisionResult(BaseModel):
    """最终决策输出，包含完整上下文信息。"""

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
    """单次运行输出，包含决策列表。"""

    run_id: str
    date: date
    product: str
    results: List[DecisionResult]


class MonitorItem(BaseModel):
    """监控计划中的单个主题条目。"""

    theme: str
    constraint_id: str
    assets: List[str]


class MonitorPlan(BaseModel):
    """产品级监控计划。"""

    product: str
    items: List[MonitorItem]


class BacktestTrade(BaseModel):
    """回测交易记录。"""

    date: date
    asset_id: str
    action: str
    price: float
    quantity: float
    cost: float
    reason: str


class EquityPoint(BaseModel):
    """净值曲线采样点。"""

    date: date
    equity: float
    cash: float
    positions_value: float


class BacktestSummary(BaseModel):
    """回测汇总指标。"""

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
    """流程状态：贯穿整条策略流水线的共享上下文。"""

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
        # 将流程中分散的结果按主题聚合，便于调试或报表展示
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
