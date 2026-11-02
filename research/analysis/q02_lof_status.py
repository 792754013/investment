"""OPP-003 快速否证（第二步）：加入申购状态后，每年还能赚多少元。

q01 假设每天都能申购，结果远超 G0，但溢价高的时候往往正是暂停或限额申购的时候。
本步用 lof_status_events（东方财富销售类公告标题）重建每只基金每天的申购状态：
  suspend → 当天不能申购；limited → 每日限额 LIMITED 元（金额在公告正文里拿不到，取情形 100 / 1,000）；
  open → 每日不超过 OPEN_CAP 元（plan.md 3.3 节风险限额，且不超过 20 日均成交额的 10%）。
状态从公告日之后的第一个交易日生效。第一条公告之前、以及没有下载公告的基金（q01 中贡献合计 <5%），按 open 处理（对策略有利）。
其余口径与 q01 相同。只读样本内。

用法：uv run python research/analysis/q02_lof_status.py
"""

import itertools

import polars as pl

import q01_lof_premium as q1
from lab import lake, trials
from lab.paths import data_root

LIMITED = [100, 1_000]
OPEN_CAP = 10_000
THETAS = [0.03, 0.05, 0.08]
SUB_FEES = [0.006, 0.012]
MIN_FEES = [0.0, 5.0]
OUT = data_root() / "research" / "q02"


def status(p: pl.DataFrame) -> pl.DataFrame:
    ev = lake.scan("lof_status_events").filter(pl.col("state") != "holiday").select("code", "ts", "state").collect()
    days = p.select("code", "ts").unique().sort("code", "ts")
    # 生效日 = 公告日之后该基金的第一个交易日
    ev = ev.with_columns((pl.col("ts") + pl.duration(days=1)).alias("eff")).sort("eff")
    ev = ev.join_asof(days.rename({"ts": "day"}).sort("day"), left_on="eff", right_on="day", by="code",
                      strategy="forward").drop_nulls("day").select("code", "day", "state")
    ev = ev.unique(["code", "day"], keep="last").sort("day")
    return p.sort("ts").join_asof(ev.rename({"day": "ts_ev"}), left_on="ts", right_on="ts_ev", by="code",
                                  strategy="backward").with_columns(pl.col("state").fill_null("open"))


def main() -> None:
    p = status(q1.panel())
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(4)

    sig = p.filter(pl.col("signal") >= 0.03)
    t1 = sig.group_by("is_qdii", "state").agg(pl.len().alias("signal_days"), pl.col("gross").mean().alias("gross_mean"),
                                              pl.col("gross").median().alias("gross_med")).sort("is_qdii", "state")
    print("表 1 信号溢价 ≥ 3% 的基金日，按当天申购状态\n", t1)

    rows = []
    for theta, fee, mn, lim in itertools.product(THETAS, SUB_FEES, MIN_FEES, LIMITED):
        s = p.filter((pl.col("signal") >= theta) & (pl.col("state") != "suspend")).with_columns(
            pl.when(pl.col("state") == "limited").then(pl.lit(float(lim))).otherwise(
                pl.min_horizontal(pl.lit(float(OPEN_CAP)), pl.col("sell_amt20") * 0.1)).alias("size"))
        s = s.filter(pl.col("size") >= 100).with_columns(
            (pl.col("gross") - fee - pl.max_horizontal(pl.col("size") * q1.RATE, pl.lit(mn)) / pl.col("size")
             - q1.EXCH - q1.SLIP).alias("net")).with_columns((pl.col("net") * pl.col("size")).alias("rmb"))
        yearly = s.group_by("year").agg(pl.col("rmb").sum(), pl.len().alias("n")).sort("year")
        full = yearly.filter(pl.col("year").is_between(2021, 2025))
        by_state = s.group_by("state").agg(pl.col("rmb").sum()).to_dicts()
        top = s.group_by("code").agg(pl.col("rmb").sum()).sort("rmb", descending=True)
        m = dict(trades=s.height, net_mean=s["net"].mean(), net_med=s["net"].median(),
                 rmb_per_year=float(full["rmb"].mean()), rmb_min_year=float(full["rmb"].min()),
                 top1_share=float(top["rmb"][0] / top["rmb"].sum()) if top.height else None,
                 top5_share=float(top["rmb"].head(5).sum() / top["rmb"].sum()) if top.height else None)
        trials.record("q02_lof_status", params=dict(theta=theta, sub_fee=fee, min_fee=mn, limited=lim, open_cap=OPEN_CAP),
                      metrics=m | {"years": {str(y): round(v) for y, v in zip(yearly["year"], yearly["rmb"])}},
                      sample="in_sample", note="快速否证第二步：按公告标题重建申购状态")
        rows.append(dict(theta=theta, sub_fee=fee, min_fee=mn, limited=lim) | m
                    | {f"rmb_{d['state']}": d["rmb"] for d in by_state}
                    | {f"y{y}": v for y, v in zip(yearly["year"], yearly["rmb"])})
    t = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.write_parquet(OUT / "scenarios.parquet")
    print("\n表 2 每年净额（元，2021～2025 年平均 / 最差年份），G0 = 3,000 元\n",
          t.select("theta", "sub_fee", "min_fee", "limited", "trades", "net_med", "rmb_per_year", "rmb_min_year",
                   "rmb_open", "rmb_limited", "top1_share", "top5_share"))
    print("\n表 3 分年份（元）\n", t.select("theta", "sub_fee", "min_fee", "limited", *[c for c in t.columns if c.startswith("y")]))


if __name__ == "__main__":
    main()
