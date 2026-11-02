"""R2 持续性（plan.md 9.5 节，P1 粗略版）。

问题：某次结算费率很高时，之后 1～30 天的费率是多少？多快回落到基准？多大概率转负？
只读样本内。品种池沿用 R1 的变体 B 池。

用法：uv run python research/analysis/r02_persistence.py
"""

import polars as pl

from lab import trials
from lab.paths import data_root
from r01_exec_map import build

BASE_APR = 0.1095  # 0.01% / 8h
ENTRY = [0.15, 0.30, 0.50, 1.00]
DAYS = [1, 2, 3, 5, 7, 10, 14, 21, 30]
OUT = data_root() / "research" / "r02"


def main() -> None:
    # 只在 B 池入场，但之后的费率不管是否还在池内都要算（持仓不会因为成交额下降而消失）
    allrows = build().sort("inst_id", "ts").with_columns(
        pl.col("funding_rate").cum_sum().over("inst_id", "episode").alias("cum"),
    )
    entries = allrows.filter(pl.col("pool_b"))
    last = allrows["ts"].max()
    tgt = allrows.select("inst_id", "episode", "ts", "cum")

    rows = []
    for d in DAYS:
        e = (
            entries.with_columns((pl.col("ts") + pl.duration(days=d)).alias("t_end"))
            .filter(pl.col("t_end") <= last)
            .sort("t_end")
            .join_asof(tgt.rename({"ts": "ts_r", "cum": "cum_end"}).sort("ts_r"),
                       left_on="t_end", right_on="ts_r", by=["inst_id", "episode"], strategy="backward")
            .with_columns(((pl.col("cum_end") - pl.col("cum")) * 365 / d).alias("fwd_apr"))
        )
        for thr in ENTRY:
            s = e.filter(pl.col("apr_raw") >= thr)
            rows.append(dict(
                entry_apr_min=thr, days=d, n=s.height,
                fwd_apr_mean=s["fwd_apr"].mean(), fwd_apr_median=s["fwd_apr"].median(),
                share_fwd_negative=(s["fwd_apr"] < 0).mean(),
            ))
    t1 = pl.DataFrame(rows)

    # 半衰期：前 d 天平均年化中超出基准的部分，降到第 1 天的一半时的 d（用累计口径的边际值近似）
    marg = []
    for thr in ENTRY:
        sub = t1.filter(pl.col("entry_apr_min") == thr).sort("days")
        cum = (sub["fwd_apr_mean"] * sub["days"] / 365).to_list()
        prev_d, prev_c = 0, 0.0
        for d, c in zip(sub["days"].to_list(), cum):
            marg.append(dict(entry_apr_min=thr, from_day=prev_d, to_day=d,
                             marginal_apr=(c - prev_c) * 365 / (d - prev_d)))
            prev_d, prev_c = d, c
    t2 = pl.DataFrame(marg)
    hl = (
        t2.join(t2.filter(pl.col("from_day") == 0).select("entry_apr_min", pl.col("marginal_apr").alias("day1")),
                on="entry_apr_min")
        .filter((pl.col("marginal_apr") - BASE_APR) <= (pl.col("day1") - BASE_APR) / 2)
        .group_by("entry_apr_min").agg(pl.col("to_day").min().alias("half_life_days_upper"))
        .sort("entry_apr_min")
    )

    by_year = (
        entries.filter(pl.col("apr_raw") >= 0.30)
        .with_columns((pl.col("ts") + pl.duration(days=7)).alias("t_end"))
        .filter(pl.col("t_end") <= last)
        .sort("t_end")
        .join_asof(tgt.rename({"ts": "ts_r", "cum": "cum_end"}).sort("ts_r"),
                   left_on="t_end", right_on="ts_r", by=["inst_id", "episode"], strategy="backward")
        .group_by(pl.col("ts").dt.year().alias("year"))
        .agg(pl.len().alias("n"), ((pl.col("cum_end") - pl.col("cum")) * 365 / 7).median().alias("fwd7_apr_median"),
             ((pl.col("cum_end") - pl.col("cum")) * 365 / 7).mean().alias("fwd7_apr_mean"))
        .sort("year")
    )

    pl.Config.set_tbl_rows(80)
    pl.Config.set_float_precision(4)
    print("表 1 入场后 d 天内的平均年化（按名义，B 池入场）\n", t1)
    print("\n表 2 各区间的边际年化\n", t2)
    print("\n表 3 半衰期（超出基准部分减半所需天数，取网格上界）\n", hl)
    print("\n表 4 入场 ≥30% 后 7 天的年化，按年份\n", by_year)
    OUT.mkdir(parents=True, exist_ok=True)
    for n, d in (("forward", t1), ("marginal", t2), ("half_life", hl), ("by_year", by_year)):
        d.write_parquet(OUT / f"{n}.parquet")
    trials.record("r02_persistence", params=dict(entry=ENTRY, days=DAYS, pool="pool_b"),
                  metrics=dict(zip([str(x) for x in hl["entry_apr_min"]], hl["half_life_days_upper"].to_list())),
                  sample="in_sample", note="描述性统计，非规则回测")


if __name__ == "__main__":
    main()
