# investment

方案的唯一来源是 `docs/plan.md`（v2，2026-11-02 确认）：底仓被动配置 + 规则红利做增量 + 研究作为机会管道，以赚钱为主。需要背景时直接读它，不要扫描整个仓库。v1.1（结构性小容量平台、加密资金费 carry）已归档到 `docs/archive/plan-v1.1.md`，只作历史参考。

## 当前阶段

- **v1.1 的两个方向都已结束**：加密资金费 carry（OPP-001）G1 不通过；可转债双低 + 事件（OPP-002）证据不足、前向记录已停止。两者都是 retired，封存段都没有解封。结论见 `docs/research/G1-conclusion.md`、`CB-G1-conclusion.md`、`C1v5-events-overlay.md`、`CB-forward.md`。
- **OPP-003（QDII / LOF 溢价申购）已在快速否证阶段放弃**（`docs/research/OPP003-quick-falsify.md`）：溢价高的时候多半暂停申购，其余时候限额只有每天 10～100 元。
- **OPP-004（封闭 / 定开基金折价收敛）两轮都没有通过 G1**（`docs/research/OPP004-G1-conclusion.md`、`OPP004-round2-conclusion.md`）：超额回撤超过 3,000 元的上限；能压低回撤的限制方式，会同步压低超额，所以找不到高原。没有解封。等用户决定停止还是做第 3 轮。OPP-005（中小盘指数调样）还没开始。
- **用户的决定**（plan.md 第 9 节）：以赚钱为主；A 股资金最多约 30 万元，前提是风险可控（plan.md 3.3 节限额）；研究每月约 10 小时，同时最多推进 1～2 个机会；1 个月内开户；加密方向保留观察，确认法律风险之前不研究、不执行。
- **底仓政策由用户自定**，不替用户选配置比例。
- **一切在本地 Mac 上跑**：不买 VPS，也不要建议购买；不搭 Go 平台；没有通过 plan.md 第 7 节的条件之前，不写自动下单代码；不买付费数据。定时任务用 launchd，Mac 休眠或断网造成的缺口如实记录，不插值。
- `research/` 下的代码是一次性原型，够用即可，不要为将来做抽象。

## 研究纪律（硬性规定）

1. **封存样本外**：`research/seal.toml` 的 `start` 之后的数据封存。
   - 分析代码只能通过 `lab.lake.scan()` 读数据，不得直接读取 `data/raw/` 或 `data/sealed/` 下的文件。
   - 解封需要 `LAB_UNSEAL=1` 加 `reason`，只有在参数已冻结、该机会的 G1 条件文件已提交之后才允许，而且必须先征得用户同意。解封只有一次。
   - 不得修改 `seal.toml` 的 `start`，不得自行把数据集加入 `exempt`。
2. **试验日志**：每跑一次回测，每换一次参数或规则，都调用 `lab.trials.record()`。`research/trials.csv` 只追加，不删改。报告里的试验次数从这里统计，如实写出。
3. **G1 门槛先于结果**：每个机会的 G1 条件文件（如 `G1-criteria.md`、`CB-G1-criteria.md`）由用户确认并提交后，才能开始规则回测。提交后不得为了让结果通过而修改门槛。同一个机会的规则修改超过 3 轮仍未通过，默认停止（plan.md 第 6 节）。
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
- 机会卡片：`docs/opportunities/OPP-XXX.yaml`（模板见 plan.md 附录 A，必须回答准入七问，包括拥挤度）
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
