"""bt01 的有利口径敏感性：评估"是不是门槛太高"（只用样本内）。

在 bt01 规则不变的前提下，换成对策略最有利、但仍算现实的假设：
  - 滑点从每腿每次 0.10% 降到 0.03%（主流币的常见水平），单边成本 0.35% → 0.21%
  - 闲置资金放 USDT 活期，按同期出借利率计息
这不符合 G1 的悲观口径，只用来看"换一个更低的门槛，策略能不能过"。

用法：uv run python research/analysis/bt01b_favorable.py
"""

import itertools

import polars as pl

import bt01_rules as bt
from lab import trials

FAVORABLE_LEG_COST = 0.0021  # 现货 0.10% + 永续 0.05% + 两腿滑点各 0.03%
GRID = dict(entry=[0.30, 0.50], n_pos=[5, 10])


def main() -> None:
    lend = bt.benchmark().group_by(pl.col("ts").dt.truncate("1d").alias("day")).agg(
        pl.col("lending_rate").mean().alias("rate"))
    bench = float(lend["rate"].mean())
    f = bt.frame()
    groups = [(h, g.to_dicts()) for (h,), g in f.group_by(["hour"], maintain_order=True)]
    regime = bt.btc_regime()
    bt.LEG_COST = FAVORABLE_LEG_COST  # run() 在调用时读取模块级常量

    res = []
    for entry, n_pos in itertools.product(*GRID.values()):
        eq, tr = bt.run(groups, "B", entry, n_pos)
        m = bt.evaluate(eq, tr, bench, regime)
        daily = (
            eq.group_by(pl.col("hour").dt.truncate("1d").alias("day")).agg(pl.col("n_open").mean())
            .join(lend, on="day", how="left").with_columns(pl.col("rate").forward_fill().backward_fill())
            .with_columns((1 - pl.col("n_open") / n_pos).alias("idle"))
        )
        idle_apr = float((daily["idle"] * daily["rate"]).mean())
        yearly = daily.group_by(pl.col("day").dt.year().alias("year")).agg(
            (pl.col("idle") * pl.col("rate")).mean().alias("idle_apr"), pl.col("rate").mean().alias("bench"))
        years = {
            y: round(m["years"][str(y)] + i - b, 4)
            for y, i, b in zip(yearly["year"], yearly["idle_apr"], yearly["bench"]) if str(y) in m["years"]
        }
        out = dict(entry=entry, n_pos=n_pos, net_apr=m["net_apr"], idle_apr=idle_apr,
                   total_apr=m["net_apr"] + idle_apr, bench=bench, excess=m["net_apr"] + idle_apr - bench,
                   mdd=m["mdd"], trades=m["trades"], excess_by_year=years, regimes=m["regimes"])
        trials.record("bt01b_favorable", variant="B",
                      params=dict(entry=entry, n_pos=n_pos, leg_cost=FAVORABLE_LEG_COST, idle_in_savings=True),
                      metrics={k: out[k] for k in ("net_apr", "total_apr", "excess", "mdd")} | {"years": years},
                      sample="in_sample", note="有利口径敏感性，不符合 G1 悲观口径")
        res.append(out)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
