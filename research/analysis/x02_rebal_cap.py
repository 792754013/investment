"""OPP-005 补充：单机会上限改为 30 万元（用户 2026-11-03 提出）后的 G0 重算，加上板块权限限制。

沿用 x01 的逐股收益（data/research/x01/book_*.parquet，只含样本内），只重算成本和资金：
  - 资金 C ∈ {7.5 万（爬坡首期 = 上限的 1/4）, 10 万, 30 万}，单只 ≤ 10%，每期买 10 只、每只 C/10；
  - 股票池：全部 / 只限沪深主板（新开户没有创业板、科创板、北交所权限）。
期望值 = C × 全部可买股票净收益的平均。冲击每边 0.2% 和 0.3%。

用法：uv run python research/analysis/x02_rebal_cap.py
"""

import itertools

import numpy as np
import polars as pl

from lab import trials
from lab.paths import data_root

RATE, MIN_FEE = 0.00025, 5.0
X01 = data_root() / "research" / "x01"
BOOKS = {"中证1000 调入 公告+0→生效+0": "book_中证1000_调入_公告+0-生效+0.parquet",
         "中证1000 调入 公告+2→生效+0": "book_中证1000_调入_公告+2-生效+0.parquet",
         "中证500+1000 调入 公告+2→生效+0": "book_中证500+1000_调入_公告+2-生效+0.parquet"}


def main_board(code: pl.Expr) -> pl.Expr:
    return code.str.slice(0, 2).is_in(["60", "00"])


def run(b: pl.DataFrame, cap: float, slip: float) -> dict:
    size = cap / 10
    stamp = pl.when(pl.col("x_ts").dt.date() < pl.date(2023, 8, 28)).then(0.001).otherwise(0.0005)
    b = b.with_columns((pl.col("ab") - (2 * max(size * RATE, MIN_FEE) / size + 2 * slip + stamp)).alias("net"))
    per = b.group_by("ts").agg(pl.col("net").mean(), pl.col("ab").mean())
    years = b["ts"].dt.year().n_unique()
    yy = per.group_by(pl.col("ts").dt.year().alias("y")).agg((pl.col("net").sum() * cap).alias("rmb")).sort("y")
    rng = np.random.default_rng(0)
    tot = np.zeros(2000)
    for _, g in b.group_by("ts"):
        tot += rng.choice(g["net"].to_numpy(), size=(2000, 10), replace=True).sum(axis=1) * size
    a = tot / years
    return dict(events=per.height, stocks_per_event=b.height / per.height, ab_event_mean=float(per["ab"].mean()),
                ex_per_year=float(per["net"].sum() * cap / years), p10=float(np.percentile(a, 10)), p50=float(np.percentile(a, 50)),
                share_ge_3000=float((a >= 3000).mean()), worst_event_rmb=float(per["net"].min() * cap),
                by_year={str(y): round(v) for y, v in zip(yy["y"], yy["rmb"])})


def main() -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    rows = []
    for (name, f), pool, cap, slip in itertools.product(BOOKS.items(), ["全部", "沪深主板"], [75_000, 100_000, 300_000], [0.002, 0.003]):
        b = pl.read_parquet(X01 / f)
        if pool == "沪深主板":
            b = b.filter(main_board(pl.col("code")))
        m = run(b, cap, slip)
        trials.record("x02_rebal_cap", params=dict(window=name, pool=pool, cap=cap, slip=slip, k=10, rate=RATE, min_fee=MIN_FEE),
                      metrics=m, sample="in_sample", note="OPP-005 补充：单机会上限 30 万 + 板块权限")
        rows.append(dict(window=name, pool=pool, cap=cap, slip=slip) | {k: v for k, v in m.items() if k != "by_year"}
                    | {f"y{k}": v for k, v in m["by_year"].items()})
    g = pl.DataFrame(rows)
    g.write_parquet(X01 / "x02_grid.parquet")
    print(g)


if __name__ == "__main__":
    main()
