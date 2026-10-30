# P0 交易所对比：Binance / OKX / Bybit

- 日期：2026-10-30
- 对应：plan.md 19 节 P0 第 1 步、25 节待决问题 1
- 数据：公开接口快照（`research/analysis/p0_exchange_snapshot.py`，输出 `data/p0/exchange_snapshot_20261030.parquet`）+ 公开文档与第三方资料
- 局限：品种数和费率都只是**单日快照**，会随行情变化。手续费、开通门槛来自网页资料，开户前要在官网核实

## 结论

1. **变体 B（统一账户）决定了 Binance 是否可选。** 附录 B 的测算显示，变体 A 在主流币上按资本计只有约 3% 年化，所以一期大概率要靠变体 B。
   - Binance 的现货抵押要用 Portfolio Margin。网上资料说门槛是总资产 5 万或 10 万 USDT，两种说法都没有找到 2026 年的官方确认；抵押品折扣率也要 Key 才能查。
   - 如果计划资金低于这个门槛，Binance 只能做变体 A。
2. **OKX 和 Bybit 都能做变体 B，而且门槛低。**
   - Bybit 统一交易账户（UTA）对所有人开放。
   - OKX 多币种保证金模式要求账户权益 ≥ 1 万 USD，另需通过一次知识测试。
   - 两家的抵押品折扣率都能从公开接口查到。
3. **免费的现货盘口历史，OKX 最好。** R4 成本估算要同时知道现货和永续两条腿的滑点：
   - OKX：永续和现货都有 400 档 L2 盘口日文件，最早至少到 2023-01。
   - Bybit：永续有 500 档盘口（2023-03 起），现货只有 200 档，而且大约从 2025 年中才开始。
   - Binance：永续 `bookDepth` 从 2023-01 起，但**没有现货盘口历史**。
4. **建议：不急于现在定一家，把交易所选择并入 P1 的 R1。** 所有研究数据都是公开的，可以在 P1 用 OKX 和 Bybit 分别算一遍 R1 可执行费率地图，选净收益更好的一家作为 G1 主力，然后再去开户。多出的工作量大约 1～2 天，换来的是用数据选交易所。只有当你的资金能达到 Binance PM 门槛时，才把 Binance 加入比较。

**前提**：三家的服务条款都排除了中国大陆用户（Bybit 2026 年的条款还排除香港）。你需要先确认自己所在的司法辖区能合规开户、完成 KYC。这一点决定了哪家能进入候选，优先级高于上面所有比较。

## 1. 品种与可执行池（2026-10-30 快照）

品种池过滤采用 plan.md 9.6 节的初值：有同所 USDT 现货，上线 ≥ 30 天，现货 24h 成交额 ≥ 500 万 USD，永续 OI ≥ 1000 万 USD。盘口深度条件暂未计入。

| | Binance | OKX | Bybit |
|---|---|---|---|
| USDT 本位永续 | 522 | 517 | 815 |
| 其中有同所 USDT 现货 | 360 | 220 | 300 |
| **满足品种池初值（变体 A）** | **59** | **30** | **40** |
| 其中现货可作统一账户抵押（变体 B） | 未知（需 Key） | 30 | 40 |
| 池内抵押折扣率（一档） | — | 0.80～1.0，中位 0.95 | 0.60～1.0，中位 0.85 |
| 永续总 OI（十亿 USD） | 20.0 | 7.0 | 12.2 |
| 放宽到 现货 ≥ 100 万、OI ≥ 300 万 | 158 | 59 | 86 |
| 池内当期费率年化中位数 | 6.7% | 4.3% | −5.2% |
| 池内当期费率年化 > 10% 的品种数 | 26 | 11 | 10 |

- 三家池子高度重叠，三家都有的品种有 23 个。Binance 多出的主要是 DOT、OP、APT、INJ、TIA 等，这些在另外两家多半是因为成交额或 OI 没过线。
- 结算间隔不是 8h 的永续很多（三家池内都有一批 4h），回测必须按每个品种的实际间隔计算。
- 当期费率只是单日快照，**不能用来判断收益**，那是 R1、R2 的工作。

## 2. 研究数据可得性（全部免费，无需 Key）

| | Binance | OKX | Bybit |
|---|---|---|---|
| 资金费历史（API） | 2019-09 起 | **只有最近约 3 个月** | 2020-03 起 |
| 资金费历史（归档下载） | `data.binance.vision` 月包 | `market-data-history` 月包，最早约 2022-01 | 用 API 即可 |
| 已下架合约的资金费 | 有（已用 SRMUSDT 验证） | 有（已用 SRM-USDT 验证） | 有（已用 SRMUSDT 验证） |
| 永续盘口历史 | `bookDepth` 2023-01 起（30 秒，按百分比档位聚合） | 400 档 L2 日文件，2023-01 起 | 500 档日文件，2023-03 起 |
| **现货盘口历史** | **无** | **400 档 L2 日文件，2023-01 起** | 200 档日文件，约 2025 年中起 |
| 逐笔成交 | 有 | 有 | 有 |
| 持仓量历史 | `metrics` 5 分钟 | API 有限 | API 有限 |

需要注意：
- 盘口文件体积很大（OKX 的 BTC 永续单日约 500 MB）。R4 只做抽样（按品种、按行情分段选日期），不全量下载。
- 已下架合约需要先拿到合约列表。Binance 可以从 `data.binance.vision` 的目录列出来；OKX 和 Bybit 要另外想办法（接口参数或归档目录），留到 P1 第 1 步处理。

## 3. 账户模式与费用

| | Binance | OKX | Bybit |
|---|---|---|---|
| 现货抵押永续保证金 | Portfolio Margin；门槛据称为总资产 5 万或 10 万 USDT（**待核实**） | 多币种保证金模式；权益 ≥ 1 万 USD + 知识测试 | UTA；无资金门槛 |
| 抵押品折扣率查询 | 需 Key | 公开接口 | 公开接口 |
| VIP0 现货 maker / taker | 0.10% / 0.10%（BNB 抵扣 25%） | 0.08% / 0.10% | 0.10% / 0.10% |
| VIP0 永续 maker / taker | 0.02% / 0.05% | 0.02% / 0.05% | 0.02% / 0.055% |
| 一次开平仓的 taker 总成本（两腿×两次） | 0.30% | 0.30% | 0.31% |

三家的费用差别很小，不是决定因素。按悲观口径，一次完整的开平仓仅手续费就约 0.30%，这需要持有期内的累计资金费先覆盖掉。

## 4. 待核实（开户前）

- [ ] 你所在司法辖区能否合规使用这几家
- [ ] Binance Portfolio Margin 的当前门槛（只在资金接近 5 万 USD 时才有意义）
- [ ] OKX 多币种保证金模式的门槛是否仍为 1 万 USD
- [ ] 各家官网的当前费率表

## 来源

- 公开接口：`fapi.binance.com`、`api.binance.com`、`data.binance.vision`、`www.okx.com/api/v5`（含 `public/market-data-history`）、`api.bybit.com/v5`、`api2.bybit.com/quote/public/support/download/list-files`
- [OKX：Comparing futures mode and multi-currency margin mode](https://www.okx.com/help/comparing-futures-mode-and-multi-currency-margin-mode)
- [OKX：Multi-currency margin mode vs. Portfolio margin mode](https://www.okx.com/help/11951498951821)
- [Binance Portfolio Margin 交易规则公告](https://www.binance.com/en/support/announcement/binance-portfolio-margin-trading-rules-174861644e7049aaa64845f1db2e4376)
- [Binance：Portfolio Margin 抵押率调整公告](https://www.binance.com/en/support/articles/568fe6d4bcaf4fcea744075e230644bf)
- [Bitsgap：Bybit, Binance, OKX 费用对比 2026](https://bitsgap.com/blog/bybit-binance-okx-where-your-bot-pays-less-in-2026)
- [Trade Reclaim：Crypto exchange fees compared 2026](https://trade-reclaim.com/en/blog/crypto-exchange-fees-compared-2026)
- [Datawallet：Bybit restricted countries 2026](https://datawallet.com/crypto/bybit-restricted-countries)
- [CoinDesk：交易所地理封锁与 KYC](https://hotfix.coindesk.com/business/2024/11/19/bybit-bitget-okx-vpn-geofencing-kyc-binance)
