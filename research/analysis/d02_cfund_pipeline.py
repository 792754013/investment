"""OPP-004 快速否证（第二步）：样本内最后一天还处在封闭期的品种有多少、折价多少，用来判断以后的容量。

不是回测，不记 trials。
- 周期从公告标题里的全称解析：两年 / 2 年、三年 / 3 年、18 个月；"封闭运作…转为上市开放式"的基金到期后不再封闭。
- 下一次开放日 = 最近一次开放公告日 + 周期（近似，误差约 1 个月）。封存起点之后的公告不读。
- 另外按年统计"处在封闭期且折价 ≤ −3%"的基金数，看这个机会是在变大还是变小。

用法：uv run python research/analysis/d02_cfund_pipeline.py
"""

import re

import polars as pl

import d01_cfund_discount as d1
from lab import lake


def period_months(title: str) -> int | None:
    if re.search(r"封闭运作.*转(为|型)", title) or "转为上市开放式" in title:
        return None  # 封闭期届满后转为普通 LOF，不再封闭
    for pat, m in [(r"(两|2)年", 24), (r"(三|3)年", 36), (r"18个月", 18), (r"(一|1)年", 12)]:
        if re.search(pat, title):
            return m
    return None


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(4)
    e = lake.scan("cfund_events").collect().sort("code", "ts")
    per = e.with_columns(pl.col("title").map_elements(period_months, return_dtype=pl.Int64).alias("m"))
    conv = per.filter(pl.col("title").str.contains("转为上市开放式|转型为")).select("code").unique()
    last = (per.filter(~pl.col("code").is_in(conv["code"].to_list())).group_by("code")
            .agg(pl.col("ts").max().alias("last_open"), pl.col("m").drop_nulls().first().alias("m")))
    last = last.with_columns(pl.col("last_open").dt.offset_by(pl.format("{}mo", pl.col("m"))).alias("next_open_est"))

    p = d1.panel()
    end = p["ts"].max()
    snap = p.filter(pl.col("ts") == end).join(last, on="code").join(
        lake.scan("lof_list").select("code", "name").collect(), on="code")
    snap = snap.with_columns(((pl.col("next_open_est") - pl.col("ts")).dt.total_days()).alias("days_left")).filter(
        pl.col("days_left") > 0).with_columns((-pl.col("prem") / (pl.col("days_left") / 365)).alias("ann_disc"))
    print(f"样本内最后一天 {end:%Y-%m-%d} 仍在封闭期的定开基金（下一次开放日为估计值）\n",
          snap.select("code", "name", "prem", "next_open_est", "days_left", "ann_disc", "amt20").sort("ann_disc", descending=True))
    s = snap.filter(pl.col("ann_disc") >= 0.03)
    cap = s.with_columns(pl.min_horizontal(pl.lit(d1.CAP_ONE), pl.col("amt20") * d1.ADV_FRAC).alias("size"))
    print("年化折价 ≥ 3% 的只数", s.height, "；按单只 3 万 / 20 日均成交额 10% 计可投", round(cap["size"].sum()),
          "元；按当前折价持有到开放，年化超额约", round((cap["size"] * cap["ann_disc"]).sum()), "元")

    # 按年：处在封闭期（距下一次开放公告 20～800 天）且折价 ≤ −3% 的基金日
    t = d1.trades(p, 5).filter(pl.col("days").is_between(20, 800))
    y = t.with_columns(pl.col("ts").dt.year().alias("year")).group_by("year").agg(
        pl.col("code").n_unique().alias("closed_funds"),
        pl.col("code").filter(pl.col("prem") <= -0.03).n_unique().alias("disc3_funds"),
        pl.col("prem").median().alias("prem_med"),
        pl.col("amount").median().alias("amt_med")).sort("year")
    print("\n按年：能在样本内结算的封闭期基金（注意：越接近封存起点，能结算的越少）\n", y)
    n = (p.join(e.group_by("code").agg(pl.col("ts").min().alias("first_open")), on="code")
         .filter(pl.col("prem") <= -0.03).with_columns(pl.col("ts").dt.year().alias("year"))
         .group_by("year").agg(pl.col("code").n_unique().alias("funds_disc3_any")).sort("year"))
    print("\n按年：折价 ≤ −3% 出现过的基金数（不要求样本内结算）\n", n)


if __name__ == "__main__":
    main()
