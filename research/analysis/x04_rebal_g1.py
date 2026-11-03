"""OPP-005 G1 判定（样本内部分）。条件：docs/research/OPP005-G1-criteria.md；规则：configs/strategies/idx_rebal.yaml。

只读样本内。条件 1、2、5、6 在样本内计算；条件 1 不满足就不解封。
口径修正（复核前自查发现）：买入日收盘涨停时实际买不进，跳过（和"停牌不买"同类）。修正前后两版都报。
股票池 A（沪深主板）是本次判定对象；股票池 B（主板 + 创业板）没有权限，只作参考，不算判定。

用法：uv run python research/analysis/x04_rebal_g1.py
"""

import itertools

import polars as pl

from lab import lake, trials
from lab.paths import data_root

RATE, MIN_FEE, CAP, MAX_W = 0.00025, 5.0, 300_000, 0.10
X01 = data_root() / "research" / "x01"
BOOK = {0: "book_中证1000_调入_公告+0-生效+0.parquet", 2: "book_中证1000_调入_公告+2-生效+0.parquet"}  # +2 = 公告后第 3 个交易日
POOLS = {"A": ["60", "00"], "B": ["60", "00", "30"]}


def per_event(b: pl.DataFrame, slip: float, drop_top: bool = False) -> pl.DataFrame:
    if drop_top:  # 条件 6：每期去掉贡献最大的 1 只
        b = b.sort("ab", descending=True).with_columns(pl.int_range(pl.len()).over("ts").alias("_r")).filter(pl.col("_r") > 0)
    size = pl.min_horizontal(pl.lit(CAP) / pl.len().over("ts"), pl.lit(CAP * MAX_W))
    stamp = pl.when(pl.col("x_ts").dt.date() < pl.date(2023, 8, 28)).then(0.001).otherwise(0.0005)
    b = b.with_columns((size * pl.col("ab") - 2 * pl.max_horizontal(size * RATE, pl.lit(MIN_FEE)) - size * (2 * slip + stamp)).alias("rmb"))
    return b.group_by("ts").agg(pl.col("rmb").sum(), pl.len().alias("n")).sort("ts")


def metrics(p: pl.DataFrame) -> dict:
    cum = p["rmb"].cum_sum()
    yy = p.group_by(pl.col("ts").dt.year().alias("y")).agg(pl.col("rmb").sum()).sort("y")
    return dict(avg=float(p["rmb"].sum() / yy.height), pos_years=int((yy["rmb"] > 0).sum()),
                dd=float((cum - cum.cum_max().clip(lower_bound=0)).min()),
                by_year={str(y): round(v) for y, v in zip(yy["y"], yy["rmb"])}, by_event=[round(v) for v in p["rmb"]])


def limit_up_on_entry(b: pl.DataFrame) -> pl.DataFrame:
    """买入日收盘涨停（主板 ≥ 9.5%、创业板 ≥ 19.5%）的标记：封板时实际买不进。"""
    cal = (lake.scan("idx_daily").filter(pl.col("index") == "中证500").select("ts").collect().unique().sort("ts")
           .with_row_index("i").with_columns(pl.col("i").cast(pl.Int64)))
    e = lake.scan("eq_daily").filter(pl.col("code").is_in(b["code"].unique().to_list())).collect().join(cal, on="ts").sort("code", "i")
    e = e.with_columns((pl.col("close") / pl.col("close").shift(1).over("code") - 1).alias("r0"))
    b = b.join(e.select("code", pl.col("i").alias("i0"), "r0"), on=["code", "i0"], how="left")
    lim = pl.when(pl.col("code").str.slice(0, 2).is_in(["30", "68"])).then(0.195).otherwise(0.095)
    return b.with_columns((pl.col("r0") >= lim).fill_null(False).alias("limit_up"))


def main() -> None:
    books = {k: limit_up_on_entry(pl.read_parquet(X01 / f)) for k, f in BOOK.items()}
    for fix, pool, slip in itertools.product([False, True], POOLS, [0.003, 0.002]):
        sel = {k: b.filter(pl.col("code").str.slice(0, 2).is_in(POOLS[pool]) & (~pl.col("limit_up") if fix else pl.lit(True)))
               for k, b in books.items()}
        m0 = metrics(per_event(sel[0], slip))
        m5 = metrics(per_event(sel[2], slip))
        m6 = metrics(per_event(sel[0], slip, drop_top=True))
        res = dict(c1=m0["avg"] >= 3000 and m0["pos_years"] >= 4, c2=m0["dd"] >= -6000, c5=m5["avg"] >= 2000, c6=m6["avg"] >= 2500)
        m = dict(main=m0, entry_plus3=m5["avg"], drop_top=m6["avg"]) | res
        trials.record("x04_rebal_g1", params=dict(pool=pool, slip=slip, cap=CAP, rate=RATE, min_fee=MIN_FEE, skip_limit_up=fix,
                                                  rule="configs/strategies/idx_rebal.yaml"),
                      metrics=m, sample="in_sample", note="OPP-005 G1 样本内判定" + ("" if pool == "A" and slip == 0.003 else "（参考，不算判定）")
                      + ("；口径修正：买入日封涨停不买" if fix else ""))
        tag = "判定" if pool == "A" and slip == 0.003 else "参考"
        print(f"\n[{tag}] 股票池 {pool}，冲击每边 {slip:.1%}" + ("，买入日封涨停不买" if fix else ""))
        print(f"  条件 1：年均 {m0['avg']:,.0f} 元，正收益 {m0['pos_years']}/5 年 → {'满足' if res['c1'] else '不满足'}；逐年 {m0['by_year']}")
        print(f"  条件 2：超额回撤 {m0['dd']:,.0f} 元 → {'满足' if res['c2'] else '不满足'}；逐期 {m0['by_event']}")
        print(f"  条件 5：公告后第 3 个交易日买入，年均 {m5['avg']:,.0f} 元 → {'满足' if res['c5'] else '不满足'}")
        print(f"  条件 6：每期去掉贡献最大的 1 只，年均 {m6['avg']:,.0f} 元 → {'满足' if res['c6'] else '不满足'}")


if __name__ == "__main__":
    main()
