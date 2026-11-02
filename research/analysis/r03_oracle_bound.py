"""R3 之前的上界检查：假设能完美预知未来资金费，策略最多能赚多少（只用样本内）。

不是可实现的策略，只用来回答"引入预测模型能不能翻盘"：
完美预知都过不了门槛，任何模型都过不了；完美预知能过，才值得做 R3，
而现实模型通常只能拿到上界的一小部分。

规则：在池内、未来 H 天资金费扣除悲观成本后 > 0 的结算才入场，按扣成本后净收益从高到低填满 N 个仓位，
持有 H 天（下架则提前结束，滑点加倍，口径同 R1）。不计基差（bt01 中基差合计接近 0）。
闲置资金可选择放入 USDT 活期，按同期出借利率计息。

用法：uv run python research/analysis/r03_oracle_bound.py
"""

import itertools

import polars as pl

from bt01_rules import CAPITAL, benchmark
from lab import trials
from lab.paths import data_root
from r01_exec_map import CAPITAL_MULT, build

GRID = dict(variant=["A", "B"], hold=[7, 14, 30], n_pos=[5, 10], hurdle=[None])
# 第二轮：只在交易的按资本年化 ≥ 当天活期利率 + 5pp 时入场（否则不如把钱放活期）
GRID_HURDLE = dict(variant=["B"], hold=[7, 14, 30], n_pos=[5, 10], hurdle=[0.05])
# 第三轮：假设全部挂单成交（VIP0 maker：现货 0.08%、永续 0.02%，开平各一次 = 0.20%，无滑点）。
# 这不符合 G1 的悲观口径，只用来看"降成本"这条补救路线的天花板
GRID_MAKER = dict(variant=["B"], hold=[7, 14, 30], n_pos=[5, 10], hurdle=[0.05], cost=[0.0020])
OUT = data_root() / "research" / "r03_oracle"


def run(f: pl.DataFrame, variant: str, hold: int, n_pos: int, lend: pl.DataFrame, hurdle: float | None,
        cost: float | None = None) -> dict:
    pool = "pool_a" if variant == "A" else "pool_b"
    if cost is not None:
        f = f.with_columns((pl.col(f"fund_{hold}") - cost).alias(f"net_{hold}"))
    notional = CAPITAL / (n_pos * CAPITAL_MULT[variant])
    apr = pl.col(f"net_{hold}") / CAPITAL_MULT[variant] * 365 / hold
    min_apr = pl.lit(0.0) if hurdle is None else pl.col("rate") + hurdle
    c = (
        f.filter(pl.col(pool), pl.col(f"net_{hold}").is_not_null())
        .join(lend, left_on=pl.col("ts").dt.truncate("1d"), right_on="day", how="left")
        .filter(apr > min_apr)
        .with_columns(pl.min_horizontal(pl.col("ts") + pl.duration(days=hold), pl.col("ep_end")).alias("end"))
        .select("hour", "inst_id", "episode", "end", f"net_{hold}")
        .sort("hour", f"net_{hold}", descending=[False, True])
    )
    held: dict[tuple, object] = {}
    booked = []
    for (hour,), g in c.group_by(["hour"], maintain_order=True):
        held = {k: e for k, e in held.items() if e > hour}
        for inst, ep, end, net in g.select("inst_id", "episode", "end", f"net_{hold}").iter_rows():
            if len(held) >= n_pos:
                break
            if (inst, ep) in held:
                continue
            held[(inst, ep)] = end
            booked.append((hour, end, net * notional))
    b = pl.DataFrame(booked, schema=["start", "end", "pnl"], orient="row")

    # 资金占用：按天统计在仓数量，闲置比例用于活期利息
    d = f["ts"].dt.truncate("1d")
    days = pl.datetime_range(d.min(), d.max(), "1d", eager=True).alias("day")
    occ = (
        pl.DataFrame({"day": days})
        .join(b.select("start", "end"), how="cross")
        .filter((pl.col("start") <= pl.col("day")) & (pl.col("end") > pl.col("day")))
        .group_by("day").len()
    )
    daily = (
        pl.DataFrame({"day": days}).join(occ, on="day", how="left").fill_null(0)
        .join(lend, on="day", how="left").with_columns(pl.col("rate").forward_fill().backward_fill())
        .with_columns((1 - pl.col("len") / n_pos).alias("idle"))
    )
    years = (days.max() - days.min()).days / 365
    trade_apr = b["pnl"].sum() / CAPITAL / years
    idle_apr = float((daily["idle"] * daily["rate"]).sum() / 365 / years)
    bench = float(daily["rate"].mean())
    by_year = (
        b.group_by(pl.col("start").dt.year().alias("year")).agg((pl.col("pnl").sum() / CAPITAL).alias("ret"))
        .join(daily.group_by(pl.col("day").dt.year().alias("year")).agg(
            pl.len().alias("n_days"), pl.col("rate").mean().alias("bench"),
            (pl.col("idle") * pl.col("rate")).mean().alias("idle_apr"), pl.col("idle").mean().alias("idle_share")),
            on="year", how="right")
        .with_columns((pl.col("ret").fill_null(0) * 365 / pl.col("n_days")).alias("trade_apr"))
        .with_columns((pl.col("trade_apr") + pl.col("idle_apr") - pl.col("bench")).alias("excess"))
        .sort("year")
    )
    return dict(
        variant=variant, hold=hold, n_pos=n_pos, hurdle=hurdle, cost=cost, trades=b.height,
        trade_apr=trade_apr, idle_share=float(daily["idle"].mean()),
        total_apr_with_idle=trade_apr + idle_apr, bench=bench,
        excess=trade_apr + idle_apr - bench,
        years={str(y): round(e, 4) for y, e in zip(by_year["year"], by_year["excess"])},
    )


def main() -> None:
    lend = benchmark().group_by(pl.col("ts").dt.truncate("1d").alias("day")).agg(pl.col("lending_rate").mean().alias("rate"))
    f = build()
    res = []
    # 前两轮（GRID、GRID_HURDLE）已跑过并记录，见 trials.csv
    combos = [*itertools.product(*GRID_MAKER.values())]
    for variant, hold, n_pos, hurdle, cost in combos:
        m = run(f, variant, hold, n_pos, lend, hurdle, cost)
        trials.record("r03_oracle_bound", variant=variant,
                      params=dict(hold=hold, n_pos=n_pos, hurdle=hurdle, cost=cost, capital=CAPITAL),
                      metrics={k: m[k] for k in ("trade_apr", "total_apr_with_idle", "excess")} | {"years": m["years"]},
                      sample="in_sample", note="完美预知上界，不可实现，不是候选策略")
        res.append({k: v for k, v in m.items() if k != "years"} | {f"ex_{k}": v for k, v in m["years"].items()})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    t = pl.DataFrame(res)
    OUT.mkdir(parents=True, exist_ok=True)
    t.write_parquet(OUT / "grid.parquet")
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    print("\n完美预知上界（按资本；excess = 交易收益 + 闲置资金活期利息 − 同期基准）\n", t)


if __name__ == "__main__":
    main()
