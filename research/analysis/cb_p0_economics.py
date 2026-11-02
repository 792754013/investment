"""二期可转债 P0 经济性测算（CB-P0-plan.md 第 4 节）。

只用假设参数，不读任何历史数据，不算回测，不计入试验次数。
重点看两件事：最低佣金在小资金、多持仓时把成本抬高多少；超额收益折成每年的绝对金额是多少。
用法：uv run python research/analysis/cb_p0_economics.py
"""

import itertools

import polars as pl

CAPITALS = [50_000, 75_000, 100_000]  # RMB，用户选定 5万～10万
N_HOLD = [5, 10, 20]  # 等权持仓只数
TURNOVER = [4, 12, 24]  # 年单边换手倍数（买入金额 / 资本），双低轮动按周、月调仓大约落在这里
# 佣金情形：(费率, 每笔最低元)。可转债没有印花税；经手费约十万分之一，并入费率
COMMISSION = {"万0.5无最低": (0.00005, 0.0), "万2无最低": (0.0002, 0.0), "万2最低5元": (0.0002, 5.0)}
EXCHANGE_FEE = 0.00001
SLIP = 0.001  # 每边冲击成本，悲观初值，C6 再用成交额校准
EXCESS = [0.03, 0.05, 0.08]  # 假设的成本前超额收益（相对中证转债指数）
DEV_HOURS = 160  # 二期研究平台约 4 人周（plan.md 10.5）
OPS_HOURS_PER_YEAR = 70  # 每日收盘后约 10 分钟 + 每周人工下单约 30 分钟


def side_cost(trade: float, rate: float, min_fee: float) -> float:
    return max(trade * rate, min_fee) / trade + EXCHANGE_FEE + SLIP


def main() -> None:
    rows = []
    for cap, n, turn, (cname, (rate, mn)) in itertools.product(CAPITALS, N_HOLD, TURNOVER, COMMISSION.items()):
        trade = cap / n  # 换仓时整只换掉，每笔约等于单只仓位
        drag = 2 * turn * side_cost(trade, rate, mn)
        rows.append(dict(capital=cap, n=n, turnover=turn, commission=cname,
                         per_trade=round(trade), side_cost_bp=round(side_cost(trade, rate, mn) * 1e4, 1),
                         annual_drag=round(drag, 4)))
    t = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(200)
    print("表 1 年化成本拖累（按资本）\n",
          t.filter(pl.col("capital") == 75_000).pivot("commission", index=["n", "turnover"], values="annual_drag"))
    print("\n表 1b 最低 5 元时的每边成本（bp）\n",
          t.filter(pl.col("commission") == "万2最低5元").pivot("n", index="capital", values="side_cost_bp",
                                                             aggregate_function="first"))

    hours = OPS_HOURS_PER_YEAR + DEV_HOURS / 2
    rows = []
    for cap, ex in itertools.product(CAPITALS, EXCESS):
        drag = 2 * 12 * side_cost(cap / 10, 0.0002, 0.0)  # 参考情形：10 只、年换手 12 倍、万2无最低
        net = ex - drag
        rows.append(dict(capital=cap, gross_excess=ex, net_excess=round(net, 4),
                         rmb_per_year=round(cap * net), rmb_per_hour=round(cap * net / hours, 1)))
    print("\n表 2 超额收益折成绝对金额（10 只、年换手 12 倍、万2无最低）\n", pl.DataFrame(rows))
    print(f"\n投入时间：开发 {DEV_HOURS}h 按 2 年摊销 + 运维 {OPS_HOURS_PER_YEAR}h/年 = 每年约 {hours:.0f}h")


if __name__ == "__main__":
    main()
