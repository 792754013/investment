"""C1 诊断：2025 年起双低为什么跑输中证转债指数（描述性，不是新策略）。

拆三块：
  1. 现金拖累：C1 代表参数下每年的现金占比；假设闲置现金跟随指数时，超额会变成多少
  2. 风格：可投池（与 C1 v3 相同的过滤）每 20 个交易日按双低值分 5 组，看各组之后 20 日的等权收益；
     另按价格、溢价率分组。看 2025 年是不是"双低因子"本身反转了
  3. 指数构成：可投池等权收益与中证转债指数（规模加权）每年的差，区分"双低跑输"和"小券跑输大券"
只读样本内。分组收益不扣成本；持有期内退出的券按最后收盘价计。

用法：uv run python research/analysis/cb02_diagnose_2025.py
"""

import polars as pl

import cb01_double_low as c1
from lab import trials
from lab.paths import data_root

STEP = 20
REPS = [(10, 30, 1.5), (20, 20, 1.5)]
OUT = data_root() / "research" / "cb02"


def cash_drag(days, by_day, ranks, info, idx) -> pl.DataFrame:
    cb = idx["000832"].with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("r_idx")).select("ts", "r_idx")
    rows = []
    for f, n, bf in REPS:
        eq, _, b = c1.run(days, by_day, ranks, info, f, n, bf)
        e = (
            eq.join(cb, on="ts", how="left")
            .with_columns(
                (pl.col("equity") / pl.col("equity").shift(1) - 1).alias("r"),
                (pl.col("cash").shift(1) / pl.col("equity").shift(1)).alias("w_cash"),
            )
            .with_columns((pl.col("r") + pl.col("w_cash") * pl.col("r_idx").fill_null(0)).alias("r_alt"))
        )
        m = c1.evaluate(eq, b, idx)
        trials.record("cb02_diagnose_2025", params=dict(freq=f, n=n, buffer=bf, px_floor=c1.PX_FLOOR),
                      metrics=dict(excess_b1=m["excess_b1"]), sample="in_sample",
                      note="诊断：C1 v3 规则不变，只多记录现金，用于拆现金拖累")
        y = e.group_by(pl.col("ts").dt.year().alias("year")).agg(
            pl.col("w_cash").mean().alias("cash_share"),
            ((1 + pl.col("r").fill_null(0)).product() - 1).alias("ret"),
            ((1 + pl.col("r_alt").fill_null(0)).product() - 1).alias("ret_cash_in_index"),
            ((1 + pl.col("r_idx").fill_null(0)).product() - 1).alias("index"),
        ).sort("year")
        rows.append(y.with_columns(pl.lit(f"f{f}_n{n}_b{int(bf * 10)}").alias("config")))
    return pl.concat(rows).with_columns(
        (pl.col("ret") - pl.col("index")).alias("excess"),
        (pl.col("ret_cash_in_index") - pl.col("index")).alias("excess_cash_in_index"),
    )


def forward_returns(p: pl.DataFrame, days: list) -> pl.DataFrame:
    """每 STEP 个交易日的可投池截面，以及每只券到下一个截面的收益。"""
    marks = days[::STEP]
    nxt = dict(zip(marks[:-1], marks[1:]))
    elig = p.filter(pl.col("ts").is_in(marks[:-1]) & pl.col("close").is_not_null() & pl.col("score").is_not_null()
                    & (pl.col("age") >= 5) & (pl.col("amt20") >= c1.MIN_AMT20)
                    & (pl.col("close") >= c1.PX_FLOOR) & (pl.col("stock_close") >= c1.STOCK_FLOOR))
    elig = elig.with_columns(pl.col("ts").replace_strict(nxt).alias("t_end"))
    px = p.select("bond_code", pl.col("ts").alias("t_px"), pl.col("close").alias("close_end")).sort("t_px")
    f = (
        elig.sort("t_end")
        .join_asof(px, left_on="t_end", right_on="t_px", by="bond_code", strategy="backward")
        .with_columns((pl.col("close_end") / pl.col("close") - 1).alias("fwd"))
    )
    return f.with_columns(
        (pl.col("score").rank("ordinal").over("ts") * 5 / pl.len().over("ts")).ceil().cast(pl.Int8).alias("q_score"),
        pl.col("close").cut([110, 130], labels=["<110", "110-130", ">130"]).alias("price_bucket"),
        pl.col("conv_premium").cut([10, 30], labels=["<10%", "10-30%", ">30%"]).alias("prem_bucket"),
    )


def by_year(f: pl.DataFrame, col: str) -> pl.DataFrame:
    """每期组内等权收益，再按年复合。"""
    per = f.group_by("ts", col).agg(pl.col("fwd").mean().alias("r"), pl.len().alias("k"))
    return (
        per.group_by(pl.col("ts").dt.year().alias("year"), col)
        .agg(((1 + pl.col("r")).product() - 1).alias("ret"), pl.col("k").mean().round().alias("avg_n"))
        .sort("year", col)
        .pivot(col, index="year", values="ret")
        .sort("year")
    )


def main() -> None:
    p = c1.panel()
    idx = c1.indexes()
    end = p["ts"].max()
    info = {r["bond_code"]: (r["last_ts"], r["last_close"], (end - r["last_ts"]).days > c1.ALIVE_DAYS)
            for r in p.select("bond_code", "last_ts", "last_close").unique().to_dicts()}
    p = p.filter(pl.col("ts") >= c1.START)
    days = p["ts"].unique().sort().to_list()
    by_day = {d: {r[0]: (r[1], r[2]) for r in g.select("bond_code", "close", "amt20").iter_rows()}
              for (d,), g in p.group_by(["ts"])}
    elig = p.filter(pl.col("close").is_not_null() & pl.col("score").is_not_null() & (pl.col("age") >= 5)
                    & (pl.col("amt20") >= c1.MIN_AMT20) & (pl.col("stock_close") >= c1.STOCK_FLOOR)
                    & (pl.col("close") >= c1.PX_FLOOR))
    r = {d: g.sort("score")["bond_code"].to_list() for (d,), g in elig.group_by(["ts"])}
    ranks = {c1.PX_FLOOR: {d: r.get(d, []) for d in days}}

    OUT.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_float_precision(3)

    cd = cash_drag(days, by_day, ranks, info, idx)
    print("表 1 现金拖累\n", cd.select("config", "year", "cash_share", "ret", "index", "excess", "excess_cash_in_index"))

    f = forward_returns(p, days)
    cb = idx["000832"]
    marks = days[::STEP]
    ix = (cb.filter(pl.col("ts").is_in(marks)).sort("ts")
          .with_columns((pl.col("close").shift(-1) / pl.col("close") - 1).alias("r")).drop_nulls("r")
          .group_by(pl.col("ts").dt.year().alias("year")).agg(((1 + pl.col("r")).product() - 1).alias("index")))
    ew = f.group_by("ts").agg(pl.col("fwd").mean().alias("r")).group_by(pl.col("ts").dt.year().alias("year")).agg(
        ((1 + pl.col("r")).product() - 1).alias("ew_pool"), pl.len().alias("periods"))
    t3 = ew.join(ix, on="year").sort("year").with_columns((pl.col("ew_pool") - pl.col("index")).alias("ew_minus_index"))
    t2 = by_year(f, "q_score").join(ix, on="year")
    t2b = by_year(f, "price_bucket").join(ix, on="year")
    t2c = by_year(f, "prem_bucket").join(ix, on="year")
    print("\n表 2 双低值五分组（1 = 最低）的年收益，不扣成本\n", t2)
    print("\n表 2b 按价格分组\n", t2b)
    print("\n表 2c 按转股溢价率分组\n", t2c)
    print("\n表 3 可投池等权 vs 中证转债指数\n", t3)
    for name, d in (("cash", cd), ("quintile", t2), ("price", t2b), ("premium", t2c), ("ew_vs_index", t3)):
        d.write_parquet(OUT / f"{name}.parquet")
    trials.record("cb02_diagnose_2025", params=dict(step=STEP, filters="v3"), metrics={}, sample="in_sample",
                  note="描述性分组统计，不扣成本，非规则回测")


if __name__ == "__main__":
    main()
