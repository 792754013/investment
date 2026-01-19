# DECISION_FLOW.md — 约束 × 需求 投资决策流水线（正式规范）

---

## 统一目标

本文件定义投资系统的**唯一决策流程**，所有交易必须按此顺序执行，禁止跳步。

核心原则：
- 结构优先于价格
- 风险优先于机会
- 退出先于进入
- 所有决策可回放、可审计、可解释

---

## 统一决策输出格式（强制）

每一步必须返回：

DecisionResult {
  step: string
  decision: PASS | FAIL | KEEP | REJECT | ENTER | ADD | REDUCE | EXIT | HOLD
  confidence: float (0–1)
  reason: string[]
  metrics: dict
  input_ref: dict
  timestamp: datetime
}

---

# 决策流水线（13 步）

---

## STEP 0 — 维护约束池（慢变量）

输入：
- ConstraintPool（人工维护）

输出：
- ConstraintPoolVersion
- DecisionResult: KEEP

规则：
- 约束必须是不可快速加速的
- 变更必须记录日志
- 冷却期 ≥ 7 天

失败模式：
- 约束模糊
- 频繁修改

---

## STEP 1 — 扫描需求事件

输入：
- 原始信号（订单、Capex、招标、政策、技术突破）

输出：
- DemandEvent[]
- DecisionResult: PASS / FAIL

规则：
- 必须结构化
- 必须可追溯来源

---

## STEP 2 — 需求真实性过滤（Demand Quality）

输入：
- DemandEvent

输出：
- quality_score (0–100)
- DecisionResult: PASS / REJECT

通过条件：
- quality_score ≥ 60

评分维度：
- 是否真实付费
- 是否可延迟
- 是否持续
- 是否规模足够

---

## STEP 3 — 匹配约束（Constraint Matching）

输入：
- DemandEvent
- ImpactMap

输出：
- ThemeCandidate[]
- DecisionResult: PASS / FAIL

通过条件：
- 至少命中 1 个 ACTIVE 约束
- health_score ≥ 60

---

## STEP 4 — 风险门槛过滤（Go / No-Go）

输入：
- ThemeCandidate
- RiskState
- AssetUniverse

输出：
- GateResult
- DecisionResult: PASS / REJECT

一票否决条件：
- 政策风险 HIGH
- 流动性不足
- 价格受控
- 无定价权
- 市场风险 OFF

---

## STEP 5 — 机会评分（Opportunity Scoring）

输入：
- DemandEvent
- Constraint
- ConstraintSnapshot

输出：
- OpportunityScore
- DecisionResult: PASS / FAIL

推荐公式：
score = energy × hardness × profit_concentration × (1 - penalty)

---

## STEP 6 — 约束破坏概率预警（Break Risk）

输入：
- ConstraintSnapshot
- 扩产/替代/政策信号

输出：
- break_risk (0–1)
- DecisionResult: KEEP / DOWNWEIGHT / REJECT

规则：
- break_risk > 0.6 降权
- break_risk > 0.8 禁止新开仓

---

## STEP 7 — 主题排序（Theme Ranking）

输入：
- OpportunityScore
- MarketFit
- LiquidityScore
- CrowdingPenalty
- BreakRisk

输出：
- RankedThemes
- DecisionResult: PASS

规则：
- 同时最多持有 3–5 个主题

---

## STEP 8 — 阶段识别（Stage Detection）

输入：
- 因子
- 价格
- 拥挤度
- 订单

输出：
- Stage: EARLY / MID / LATE / EXIT
- DecisionResult: PASS

---

## STEP 9 — 入场（Entry Engine）

输入：
- RankedThemes
- Stage
- RiskState
- AssetUniverse

输出：
- PositionIntent: ENTER / ADD / HOLD
- DecisionResult

规则：
- 只允许 EARLY / MID 入场
- 分批建仓（1/3 + 1/3 + 1/3）

---

## STEP 10 — 动态止损（Stoploss）

输入：
- 持仓
- 价格
- 时间
- 结构信号
- break_risk

输出：
- ExitIntent: REDUCE / EXIT / HOLD
- DecisionResult

止损类型：
- 价格止损
- 时间止损
- 结构止损
- 概率止损

---

## STEP 11 — 止盈（Take Profit）

输入：
- Stage
- CrowdingState
- 趋势破坏信号

输出：
- ExitIntent: REDUCE / EXIT / HOLD
- DecisionResult

规则：
- LATE 减仓
- 拥挤高减仓
- 趋势破坏清仓

---

## STEP 12 — 组合风控（Portfolio Control）

输入：
- 所有持仓
- 主题相关性
- 约束暴露
- 波动目标

输出：
- 调仓建议
- DecisionResult: PASS / REDUCE / FAIL

限制：
- 单主题上限
- 单标的上限
- 同约束上限
- 组合波动上限

---

## STEP 13 — 系统级 Kill Switch

输入：
- 组合回撤
- 连续失败次数
- 市场系统性风险
- 数据异常

输出：
- SystemState: NORMAL / DEGRADED / HALT
- DecisionResult

触发示例：
- 回撤 > 12% → DEGRADED
- 回撤 > 20% → HALT
- 连续 5 次结构失败 → DEGRADED
- 数据异常 → HALT

---

# 文件结束
