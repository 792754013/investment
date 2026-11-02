# investment

方案的唯一来源是 `docs/plan.md`（结构性小容量优势，第一个落地是加密资金费率 carry）。需要背景时直接读它，不要扫描整个仓库。

## 当前阶段

P0（经济性与准备）+ P1（快速研究）→ 闸门 G0、G1。目标只有一个：回答"做不做、主力用变体 A 还是 B"。

**2026-11-02 更新：G1 不通过，资金费 carry 已停止**（`docs/research/G1-conclusion.md`，OPP-001 已 retired）。封存段没有解封。不再为 carry 追加试验，也不写执行代码。用户已决定**提前启动二期 A 股可转债**（OPP-002），计划和免费数据可行性见 `docs/research/CB-P0-plan.md`。研究纪律、封存起点和"G1 前不写执行代码、不买付费数据"同样适用。

- **一期交易所：OKX**（理由见 `docs/research/P0-exchange-comparison.md`）。资金费历史用 `public/market-data-history` 的月度归档，最早到 2022-01；API 只给最近约 3 个月。
- **一切在本地 Mac 上跑**。本地验证通过之前不买 VPS，也不要建议购买。采集脚本用 launchd 在本机运行，Mac 休眠或断网造成的缺口要如实记录，不要插值掩盖。
- G1 之前**不写任何执行或下单代码**，不购买付费数据（plan.md 8.5.4）。
- `research/` 下的代码是一次性原型（plan.md 2.5 节第 10 条），够用即可，不要为将来做抽象。

## 研究纪律（硬性规定）

1. **封存样本外**：`research/seal.toml` 的 `start` 之后的数据封存。
   - 分析代码只能通过 `lab.lake.scan()` 读数据，不得直接读取 `data/raw/` 或 `data/sealed/` 下的文件。
   - 解封需要 `LAB_UNSEAL=1` 加 `reason`，只有在参数已定、`docs/research/G1-criteria.md` 已提交之后才允许，而且必须先征得用户同意。
   - 不得修改 `seal.toml` 的 `start`，不得自行把数据集加入 `exempt`。
2. **试验日志**：每跑一次回测，每换一次参数或规则，都调用 `lab.trials.record()`。`research/trials.csv` 只追加，不删改。报告里的试验次数从这里统计，如实写出。
3. **G1 门槛先于结果**：`docs/research/G1-criteria.md` 由用户确认并提交后，才能开始规则回测。提交后不得为了让结果通过而修改门槛。
4. **悲观成本**：手续费一律按 taker 费率，滑点取 R4 的 75 分位。
5. **选高原，不选尖峰**：报告参数敏感性，不要只报告最优点。
6. **不要只找支持性证据**：结论对"做"不利时同样要写清楚。G1 结论出来后，需要由一个不带前文的 agent 独立复核。

## 数据约定

- 时间一律使用 UTC，列类型为 `Datetime(time_zone="UTC")`。`lab.lake.write()` 会拒绝不带时区的时间列。
- `data/raw/`：下载的原始文件，只给入库脚本读。`data/lake/`：封存起点之前的数据。`data/sealed/`：封存段。`data/` 不进 git。
- 入库时一律调用 `lab.lake.write(dataset, df, ts_col=..., part=...)`。同一个 part 重复写入会覆盖，结果是幂等的。
- 资金费历史必须包含已下架合约，否则会有幸存者偏差。

## 产出位置

- 研究代码：`research/analysis/`（每个研究问题一个模块，例如 `r01_exec_map.py`），一次性 notebook 放 `research/notebooks/`
- 报告和结论：`docs/research/`（每个研究问题写一份结论文件，后续会话读结论，不重新读原始数据）
- 机会卡片：`docs/opportunities/OPP-XXX.yaml`（模板见 plan.md 附录 A）
- 策略参数草稿：`configs/strategies/funding_carry.yaml`

## 命令

```bash
uv sync --all-packages      # 安装或更新环境
uv run pytest -q            # 跑测试
uv run jupyter lab          # notebook
```

如果出现 `ModuleNotFoundError: No module named 'lab'`：macOS 给 `.venv` 里的 `.pth` 文件加了 hidden 标记，Python 3.13 会跳过它。运行 `chflags nohidden .venv/lib/python3.13/site-packages/*.pth` 即可。这个标记会被系统反复加回来（疑似 iCloud 同步 Documents），所以每次运行脚本前都先执行一次。

## 安全

- API Key、token 只放在 `.env`（变量名见 `.env.example`），由用户本人填写。不要读取、打印或提交 `.env`。
- P0、P1 只需要只读 Key，研究数据基本走公开接口。
