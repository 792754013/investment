"""C3 下修、C4 强赎事件研究（描述性，不是规则回测）。

事件来自 cb_notices（巨潮公告标题分类）。同一只转债、同一类事件 20 个交易日内只取第一条。
事件日 d0：
  公告时间 00:00（开盘前可得）或盘中 → 公告日当天或之后的第一个交易日；
  15:00 以后发布 → 公告日之后的第一个交易日。
收益：转债收盘价，停牌或已退出按最近一个收盘价（asof）。超额 = 转债收益 − 中证转债指数同期收益。
窗口：
  pre   d0-21 → d0-1 收盘（公告前，事先不可知，只用于理解）
  react d0-1 → d0 收盘（公告当天的反应，来不及成交）
  post5 / post20 / post60  d0 → d0+k 收盘（公告当天收盘买入后，可以实际拿到的部分）
不扣成本。只读样本内。

用法：uv run python research/analysis/cb03_events.py
"""

import polars as pl

from lab import lake, trials
from lab.paths import data_root

START = pl.datetime(2018, 1, 1, time_zone="UTC")
WINDOWS = {"pre": (-21, -1), "react": (-1, 0), "post5": (0, 5), "post20": (0, 20), "post60": (0, 60)}
GAP = 20
OUT = data_root() / "research" / "cb03"


def calendar() -> pl.DataFrame:
    days = lake.scan("cb_daily").select("ts").unique().collect().filter(pl.col("ts") >= START).sort("ts")
    return days.with_row_index("pos").with_columns(pl.col("pos").cast(pl.Int64), pl.col("ts").dt.date().alias("day"))


def events(cal: pl.DataFrame) -> pl.DataFrame:
    n = lake.scan("cb_notices").collect().filter(pl.col("ts") >= START)
    t = n.with_columns((pl.col("ts") + pl.duration(hours=8)).dt.replace_time_zone(None).alias("t_cst"))
    t = t.with_columns(
        pl.col("t_cst").dt.date().alias("ann_date"),
        (~pl.col("pre_open") & (pl.col("t_cst").dt.hour() >= 15)).alias("after_close"),
    ).with_columns(
        pl.when(pl.col("after_close")).then(pl.col("ann_date") + pl.duration(days=1)).otherwise(pl.col("ann_date")).alias("from_day")
    )
    t = t.sort("from_day").join_asof(cal.select("day", "pos").sort("day"), left_on="from_day", right_on="day",
                                     strategy="forward").rename({"pos": "d0"}).drop_nulls("d0")
    # 同一转债同一类事件，20 个交易日内只取第一条
    t = t.sort("bond_code", "kind", "d0").with_columns(
        (pl.col("d0") - pl.col("d0").shift(1).over("bond_code", "kind")).alias("gap"))
    t = t.filter(pl.col("gap").is_null() | (pl.col("gap") > GAP))
    return t.select("bond_code", "bond_name", "kind", "title", "ann_time", "d0")


def prices(cal: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    b = lake.scan("cb_daily").select("bond_code", "ts", "close").collect().filter(pl.col("ts") >= START)
    b = b.join(cal.select("ts", "pos"), on="ts")
    i = lake.scan("cb_index").filter(pl.col("index_code") == "000832").select("ts", "close").collect()
    i = cal.select("ts", "pos").join(i, on="ts", how="left").sort("pos").with_columns(pl.col("close").forward_fill())
    return b.sort("pos"), i.select("pos", pl.col("close").alias("idx"))


def window_returns(ev: pl.DataFrame, b: pl.DataFrame, idx: pl.DataFrame, last_pos: int) -> pl.DataFrame:
    out = ev
    for name, (a, z) in WINDOWS.items():
        e = ev.select("bond_code", "kind", "d0").with_columns(
            (pl.col("d0") + a).alias("pa"), (pl.col("d0") + z).alias("pz"))
        e = e.filter((pl.col("pa") >= 0) & (pl.col("pz") <= last_pos))
        for end in ("pa", "pz"):
            e = e.sort(end).join_asof(b.select("bond_code", "pos", pl.col("close").alias(f"b_{end}")),
                                      left_on=end, right_on="pos", by="bond_code", strategy="backward").drop("pos")
            e = e.join(idx.rename({"pos": end, "idx": f"i_{end}"}), on=end, how="left")
        e = e.with_columns(
            (pl.col("b_pz") / pl.col("b_pa") - 1).alias(f"r_{name}"),
            (pl.col("b_pz") / pl.col("b_pa") - pl.col("i_pz") / pl.col("i_pa")).alias(f"ar_{name}"),
        ).select("bond_code", "kind", "d0", f"r_{name}", f"ar_{name}")
        out = out.join(e, on=["bond_code", "kind", "d0"], how="left")
    return out


def summary(r: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    aggs = [pl.len().alias("n")]
    for name in WINDOWS:
        c = f"ar_{name}"
        aggs += [pl.col(c).mean().alias(f"{name}_mean"), pl.col(c).median().alias(f"{name}_med"),
                 (pl.col(c) > 0).mean().alias(f"{name}_win")]
    return r.group_by(by).agg(aggs).sort(by)


def main() -> None:
    cal = calendar()
    ev = events(cal)
    b, idx = prices(cal)
    r = window_returns(ev, b, idx, int(cal["pos"].max()))

    # 董事会提议下修之后 60 个交易日内有没有下修结果公告（是否通过）
    done = ev.filter(pl.col("kind") == "down_done").select("bond_code", pl.col("d0").alias("d_done"))
    prop = r.filter(pl.col("kind") == "down_propose").join(done, on="bond_code", how="left").with_columns(
        ((pl.col("d_done") - pl.col("d0")).is_between(0, 60)).fill_null(False).alias("passed")
    ).group_by("bond_code", "d0").agg(pl.col("passed").any(), pl.exclude("passed", "d_done").first())
    # 下修预警之后 60 个交易日内是否出现提议
    pr = ev.filter(pl.col("kind") == "down_propose").select("bond_code", pl.col("d0").alias("d_prop"))
    warn = r.filter(pl.col("kind") == "down_warn").join(pr, on="bond_code", how="left").with_columns(
        ((pl.col("d_prop") - pl.col("d0")).is_between(0, 60)).fill_null(False).alias("proposed")
    ).group_by("bond_code", "d0").agg(pl.col("proposed").any(), pl.exclude("proposed", "d_prop").first())

    years = r.join(cal.select("pos", pl.col("ts").dt.year().alias("year")), left_on="d0", right_on="pos")
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(3)
    t1 = summary(r, ["kind"])
    t2 = summary(prop, ["passed"])
    t3 = summary(warn, ["proposed"])
    t4 = summary(years.filter(pl.col("kind").is_in(["down_propose", "down_warn", "call_yes", "call_no", "call_warn"])),
                 ["kind", "year"]).select("kind", "year", "n", "react_mean", "post20_mean", "post20_med", "post20_win",
                                         "post60_mean")
    print("表 1 各类事件的超额收益（相对中证转债指数，不扣成本）\n", t1)
    print("\n表 2 董事会提议下修：之后 60 日内是否有下修结果\n", t2)
    print("\n表 3 下修预警：之后 60 日内是否出现董事会提议\n", t3)
    print("\n表 4 按年份\n", t4)
    OUT.mkdir(parents=True, exist_ok=True)
    r.write_parquet(OUT / "events.parquet")
    for nme, d in (("by_kind", t1), ("propose_passed", t2), ("warn_proposed", t3), ("by_year", t4)):
        d.write_parquet(OUT / f"{nme}.parquet")
    trials.record("cb03_events", params=dict(windows=WINDOWS, gap=GAP), metrics={}, sample="in_sample",
                  note="描述性事件研究，不扣成本，非规则回测")


if __name__ == "__main__":
    main()
