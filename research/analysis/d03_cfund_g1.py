"""OPP-004 规则回测：按 docs/research/OPP004-G1-criteria.md 逐条核对样本内条件。

复用 d01 的口径（trades / simulate），补上快速否证没有算的几项：
  - 超额曲线：每个持仓逐日盯市 size × [(P_d + D)/P_0 − (N_d + D)/N_0]，买入成本记在买入日，卖出成本记在卖出日；
    停牌日沿用上一个价格。超额回撤 = 累计超额曲线相对历史高点的最大回落（元）。
  - 参数网格：θ × 距开放天数 × 退出日，主口径场内卖出、盯市修正、最低佣金 5 元。
  - 条件 5：前 5 只占比、去掉贡献最大的 1 只后重跑；条件 6：资金 20 万的利用率。
只读样本内。

用法：uv run python research/analysis/d03_cfund_g1.py
"""

import itertools

import polars as pl

import d01_cfund_discount as d1
from lab import trials
from lab.paths import data_root

THETAS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12]
WINDOWS = [(20, 800), (40, 600), (60, 400)]
LAGS = [1, 3, 5, 10]
MIN_FEE = 5.0
COL = "gross"
OUT = data_root() / "research" / "d03"


def excess_curve(p: pl.DataFrame, b: pl.DataFrame, min_fee: float = MIN_FEE) -> pl.DataFrame:
    days = p.select("ts").unique().sort("ts")
    px = p.select("code", "ts", "close", "nav", "cumdiv")
    pos = b.select("code", pl.col("ts").alias("t0"), "x_ts", "size").with_row_index("pid")
    pos = pos.join(px.rename({"ts": "t0", "close": "p0", "nav": "n0", "cumdiv": "c0"}), on=["code", "t0"])
    g = pos.join(days, how="cross").filter((pl.col("ts") >= pl.col("t0")) & (pl.col("ts") <= pl.col("x_ts")))
    g = g.join(px, on=["code", "ts"], how="left").sort("pid", "ts").with_columns(
        pl.col("close", "nav", "cumdiv").forward_fill().over("pid"))
    d = pl.col("cumdiv") - pl.col("c0")
    cost1 = pl.max_horizontal(pl.col("size") * d1.RATE, pl.lit(min_fee)) + pl.col("size") * d1.SLIP
    g = g.with_columns((pl.col("size") * ((pl.col("close") + d) / pl.col("p0") - (pl.col("nav") + d) / pl.col("n0"))
                        - cost1 - pl.when(pl.col("ts") == pl.col("x_ts")).then(cost1).otherwise(0.0)).alias("pnl"))
    final = g.filter(pl.col("ts") == pl.col("x_ts")).select("pid", "x_ts", pl.col("pnl").alias("fin"))
    open_ = g.filter(pl.col("ts") < pl.col("x_ts")).group_by("ts").agg(pl.col("pnl").sum().alias("u"))
    real = final.group_by("x_ts").agg(pl.col("fin").sum().alias("r")).rename({"x_ts": "ts"})
    c = days.join(open_, on="ts", how="left").join(real, on="ts", how="left").fill_null(0.0).sort("ts")
    c = c.with_columns((pl.col("r").cum_sum() + pl.col("u")).alias("cum"))
    return c.with_columns((pl.col("cum") - pl.col("cum").cum_max().clip(lower_bound=0.0)).alias("dd"))


def run(p, t, theta, lo, hi, lag, cap=d1.CAP_TOTAL, exclude=(), tag="grid"):
    m = d1.simulate(t, theta, lo, hi, MIN_FEE, COL, cap_total=cap, exclude=exclude)
    b = m.pop("book", None)
    if b is not None:
        c = excess_curve(p, b)
        m["excess_dd"] = float(c["dd"].min())
        m["curve_end"] = float(c["cum"][-1])
    years = {k: v for k, v in m.get("by_year", {}).items() if "2021" <= k <= "2025"}
    m["pos_years_2021_2025"] = sum(v > 0 for v in years.values())
    trials.record("d03_cfund_g1", params=dict(theta=theta, min_days=lo, max_days=hi, exit_lag=lag, exit=COL, mtm=True,
                                              min_fee=MIN_FEE, cap_total=cap, cap_one=d1.CAP_ONE, adv_frac=d1.ADV_FRAC,
                                              slip=d1.SLIP, exclude=list(exclude), tag=tag),
                  metrics=m, sample="in_sample", note="OPP-004 规则回测：按 G1 条件核对")
    return m, b


def main() -> None:
    pl.Config.set_tbl_rows(120)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(4)
    OUT.mkdir(parents=True, exist_ok=True)
    p = d1.panel()
    ts = {lag: d1.trades(p, lag, mtm=True) for lag in LAGS}
    rows = []
    for theta, (lo, hi), lag in itertools.product(THETAS, WINDOWS, LAGS):
        m, _ = run(p, ts[lag], theta, lo, hi, lag)
        rows.append(dict(theta=theta, min_days=lo, max_days=hi, lag=lag, trades=m["trades"], avg=m["rmb_per_year"],
                         pos_years=m["pos_years_2021_2025"], excess_dd=m["excess_dd"], util=m["util"],
                         top5=m["top5_share"]) | {f"y{k}": v for k, v in m["by_year"].items()})
    g = pl.DataFrame(rows).with_columns(
        ((pl.col("avg") >= 3000) & (pl.col("pos_years") >= 4)).alias("c1"),
        (pl.col("excess_dd") >= -3000).alias("c2"),
        (pl.col("top5") <= 0.7).alias("c5a"))
    g.write_parquet(OUT / "grid.parquet")
    print("表 1 参数网格（资金 10 万；c1 年均 ≥ 3,000 且 ≥ 4 年为正；c2 超额回撤 ≥ −3,000；c5a 前 5 只 ≤ 70%）\n", g)
    ok = g.filter("c1" & pl.col("c2") & pl.col("c5a"))
    print(f"\n满足 c1、c2、c5a 的配置：{ok.height} / {g.height}")
    for k in ["theta", "lag"]:
        print(g.group_by(k).agg(pl.col("avg").mean(), pl.col("c1").mean().alias("c1_share"), pl.col("c2").mean().alias("c2_share")).sort(k))
    print(g.group_by("min_days").agg(pl.col("avg").mean(), pl.col("c1").mean().alias("c1_share"), pl.col("c2").mean().alias("c2_share")).sort("min_days"))


if __name__ == "__main__":
    main()
