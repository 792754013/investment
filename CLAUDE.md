# investment

方案的唯一来源是 `docs/plan.md`（v3，2026-11-03 确认）：底仓被动配置为主，研究改为服务底仓的执行质量（费率、再平衡、现金管理、税费、账本）；增量机会改为被动管道，只收带"专属门槛"的候选；以赚钱为主。需要背景时直接读它，不要扫描整个仓库。v2（规则红利增量管道）归档在 `docs/archive/plan-v2.md`，v1.1（结构性小容量平台、加密资金费 carry）归档在 `docs/archive/plan-v1.1.md`，只作历史参考。

## 当前阶段

- **v1.1 的两个方向都已结束**：加密资金费 carry（OPP-001）G1 不通过；可转债双低 + 事件（OPP-002）证据不足、前向记录已停止。两者都是 retired，封存段都没有解封。结论见 `docs/research/G1-conclusion.md`、`CB-G1-conclusion.md`、`C1v5-events-overlay.md`、`CB-forward.md`。
- **OPP-003（QDII / LOF 溢价申购）已在快速否证阶段放弃**（`docs/research/OPP003-quick-falsify.md`）：溢价高的时候多半暂停申购，其余时候限额只有每天 10～100 元。
- **OPP-004（封闭 / 定开基金折价收敛）已停止**（retired，`docs/research/OPP004-G1-conclusion.md`、`OPP004-round2-conclusion.md`）：两轮都没有通过 G1，超额回撤超限，找不到高原；没有解封。
- **OPP-005（中小盘指数调样）已停止**（retired，`docs/research/OPP005-quick-falsify.md`、`OPP005-G1-conclusion.md`）：用户把单机会上限调到 30 万元后重启，G1 不通过；正收益年数不足，收益依赖每期 1～2 只暴涨股；没有解封。
- **OPP-006（加密永续插针）已停止**（retired，`docs/research/OPP006-quick-falsify.md`、`OPP006-ml-conclusion.md`、`OPP006-round3-conclusion.md`）：第 1 轮规则版只有微弱正收益且对成交假设敏感，高杠杆和全仓滚动均爆仓；第 2 轮机器学习 4 个方向（含小资金高杠杆秒级接针）全部不通过；第 3 轮秒级放量顺势、缩量离场两个方向全部为负（放量峰值多是一波末端）；可按新的结构性理由重开；不建议买收费数据；没有解封。Binance 逐笔原始数据约 51 GB 在 `data/raw/binance/`。
- **OPP-007（加密货币日线趋势跟踪）快速否证通过**（`docs/research/OPP007-quick-falsify.md`）：主策略是 BTC/ETH、8 个信号等权、只做多、目标波动 40%。按永续口径年化 23%，最大回撤 34%（一直持有是 25%、81%）；用现货执行年化约 30%。第 2 轮试了 8 类 50 个方向（指标、风险叠加、山寨币截面、机器学习、新闻情绪、宏观、链上、组合），没有一个显著优于主策略（`docs/research/OPP007-round2-report.md`），保持主策略，不再寻找新信号。G1 条件和第 7 问破例已于 2026-11-05 由用户确认（`docs/research/OPP007-G1-criteria.md`，参数冻结在 `configs/strategies/opp007_b0.toml`）；下一步是征得用户同意后解封，并开始前向纸面记录（主策略为主；恐惧贪婪趋势过滤 E2 和闲置资金理财 H4 作为影子版本）。
- **A 股指数日线趋势（B0 移植，2026-11-05，未建卡）**（`docs/research/ATREND-quick-falsify.md`）：停止条件没有触发，但优势几乎全部来自 2008 年；2009 年以后不如一直持有，也略不如固定半仓。不建议作为增量机会；如果要用作底仓的极端风险开关，属于底仓政策，等用户写好投资政策说明后放进执行研究。
- **v3（2026-11-03）**：5 个机会 0 个通过，停止主动挖掘公开数据里的异象，也不转向价格预测（理由见 plan.md 1.4 节）。
  - 研究主线是 plan.md 4.2 节的执行研究 E1～E5，每项写出"每年能省多少元"，由用户决定是否采用；不需要 G1 和解封，但回测性质的计算照样记录试验次数。E3（现金管理，`docs/research/E03-cash-management.md`）、E4（税费与持有规则，`docs/research/E04-tax-fees.md`，2026-11-06）已完成；E4 的开户费率清单在该文第 6 节，开户拿到书面费率后改 `research/analysis/e04_tax_fees.py` 重算。为帮助用户写投资政策说明，已做配置的历史回看（`docs/research/ALLOC-history.md`，2026-11-06；交互页面 https://claude.ai/artifact/Fn47Z3Eyvx3pAVEpAX5EBX，模板在 `research/analysis/a01_page/`），只给示意组合的历史表现，不替用户选比例。E1、E2、E5 等用户写好投资政策说明再做。
  - 增量机会：准入第 7 问"专属门槛"是硬门槛，卡片写不出 `exclusive_barrier` 就不建卡；管道为空是正常状态。
  - 信息优势前向记录（plan.md 5.2 节）暂不启用，等用户确认有信息来源。
- **用户的决定**（plan.md 第 9 节）：以赚钱为主；A 股资金最多约 30 万元，前提是风险可控（plan.md 3.3 节限额，单机会上限 30 万元）；研究时间按 plan.md 3.4 节按需投入（执行研究前 2～3 个月每月约 5～10 小时，之后每季度约 2 小时）；1 个月内开户；加密方向不再以法律风险为前置条件（用户 2026-11-03 决定，由用户自行判断），和其他增量机会一样按准入七问、G0、G1 走。
- **底仓政策由用户自定**，不替用户选配置比例，也不替用户选产品；执行研究只给比较和结论。
- **一切在本地 Mac 上跑**：不买 VPS，也不要建议购买；不搭 Go 平台；没有通过 plan.md 第 7 节的条件之前，不写自动下单代码；不买付费数据。定时任务用 launchd，Mac 休眠或断网造成的缺口如实记录，不插值。
- `research/` 下的代码是一次性原型，够用即可，不要为将来做抽象。

## 研究纪律（硬性规定）

1. **封存样本外**：`research/seal.toml` 的 `start` 之后的数据封存。
   - 分析代码只能通过 `lab.lake.scan()` 读数据，不得直接读取 `data/raw/` 或 `data/sealed/` 下的文件。
   - 解封需要 `LAB_UNSEAL=1` 加 `reason`，只有在参数已冻结、该机会的 G1 条件文件已提交之后才允许，而且必须先征得用户同意。解封只有一次。
   - 不得修改 `seal.toml` 的 `start`，不得自行把数据集加入 `exempt`。
2. **试验日志**：每跑一次回测，每换一次参数或规则，都调用 `lab.trials.record()`。`research/trials.csv` 只追加，不删改。报告里的试验次数从这里统计，如实写出。
3. **G1 门槛先于结果**：每个机会的 G1 条件文件（如 `G1-criteria.md`、`CB-G1-criteria.md`）由用户确认并提交后，才能开始规则回测。提交后不得为了让结果通过而修改门槛。同一个机会不设轮数上限（用户 2026-11-04 决定）：每轮结论写明是否还有研究价值，有新的结构性理由就继续，只换参数不算（plan.md 第 6 节）。
4. **悲观成本**：手续费按实际可得的最差档，加密按 taker；最低收费要计入；滑点或冲击成本取悲观值。
5. **选高原，不选尖峰**：报告参数敏感性，不要只报告最优点。
6. **不要只找支持性证据**：结论对"做"不利时同样要写清楚。G1 结论出来后，需要由一个不带前文的 agent 独立复核。

## 数据约定

- 时间一律使用 UTC，列类型为 `Datetime(time_zone="UTC")`。`lab.lake.write()` 会拒绝不带时区的时间列。
- `data/raw/`：下载的原始文件，只给入库脚本读。`data/lake/`：封存起点之前的数据。`data/sealed/`：封存段。`data/` 不进 git。
- 入库时一律调用 `lab.lake.write(dataset, df, ts_col=..., part=...)`。同一个 part 重复写入会覆盖，结果是幂等的。
- 历史数据必须包含已下架、已退市、已到期的品种，否则会有幸存者偏差。

## 产出位置

- 研究代码：`research/analysis/`（每个研究问题一个模块，例如 `r01_exec_map.py`），一次性 notebook 放 `research/notebooks/`
- 报告和结论：`docs/research/`（每个研究问题写一份结论文件，后续会话读结论，不重新读原始数据）
- 执行研究：代码 `research/analysis/e0X_*.py`，结论 `docs/research/E0X-*.md`
- 机会卡片：`docs/opportunities/OPP-XXX.yaml`（模板见 plan.md 附录 A，必须回答准入七问，第 7 问专属门槛为硬门槛）
- 冻结的策略参数：`configs/strategies/`

## 命令

```bash
uv sync --all-packages      # 安装或更新环境
uv run pytest -q            # 跑测试
uv run jupyter lab          # notebook
```

如果出现 `ModuleNotFoundError: No module named 'lab'`：macOS 给 `.venv` 里的 `.pth` 文件加了 hidden 标记，Python 3.13 会跳过它。运行 `chflags nohidden .venv/lib/python3.13/site-packages/*.pth` 即可。这个标记会被系统反复加回来（疑似 iCloud 同步 Documents），所以每次运行脚本前都先执行一次。

## 安全

- API Key、token 只放在 `.env`（变量名见 `.env.example`），由用户本人填写。不要读取、打印或提交 `.env`。
- 研究数据基本走公开接口，不需要 Key。
