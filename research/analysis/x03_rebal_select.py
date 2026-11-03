"""OPP-005 补充：G1 起草前，固定选股规则（代替"随机挑 10 只"）和按期累计的超额回撤。

沿用 x01 的逐股收益（中证 1000 调入，公告+0 / +2 → 生效日收盘，只含样本内），只限沪深主板，资金 30 万元：
  - 全买：主板调入股等权（每期约 58 只，每只约 5,000 元）；
  - 前 10 / 前 20：按买入日 20 日均成交额从高到低取，等权。
冲击每边 0.2% 和 0.3%。超额回撤 = 按生效日顺序累计每期净超额后的最大回撤。

用法：uv run python research/analysis/x03_rebal_select.py
"""

import itertools

import polars as pl

from lab import trials
from lab.paths import data_root

RATE, MIN_FEE, CAP = 0.00025, 5.0, 300_000
X01 = data_root() / "research" / "x01"
BOOKS = {"公告+0→生效+0": "book_中证1000_调入_公告+0-生效+0.parquet", "公告+2→生效+0": "book_中证1000_调入_公告+2-生效+0.parquet"}


POOLS = {"沪深主板": ["60", "00"], "主板+创业板": ["60", "00", "30"], "全部": ["60", "00", "30", "68"]}


def run(b: pl.DataFrame, rule: str, slip: float, pool: str = "沪深主板") -> dict:
    b = b.filter(pl.col("code").str.slice(0, 2).is_in(POOLS[pool]))
    if rule != "全买":
        k = int(rule[1:])
        b = b.sort("amt20", descending=True).group_by("ts", maintain_order=True).head(k)
    n = pl.len().over("ts")
    size = pl.lit(CAP) / n
    stamp = pl.when(pl.col("x_ts").dt.date() < pl.date(2023, 8, 28)).then(0.001).otherwise(0.0005)
    b = b.with_columns((size * pl.col("ab") - 2 * pl.max_horizontal(size * RATE, pl.lit(MIN_FEE)) - size * (2 * slip + stamp)).alias("rmb"))
    per = b.group_by("ts").agg(pl.col("rmb").sum(), pl.col("ab").mean(), pl.len().alias("n")).sort("ts")
    cum = per["rmb"].cum_sum()
    dd = float((cum - cum.cum_max().clip(lower_bound=0)).min())
    years = per["ts"].dt.year().n_unique()
    yy = per.group_by(pl.col("ts").dt.year().alias("y")).agg(pl.col("rmb").sum()).sort("y")
    return dict(events=per.height, n_mean=float(per["n"].mean()), ab_event_mean=float(per["ab"].mean()),
                pos_events=int((per["rmb"] > 0).sum()), ex_per_year=float(per["rmb"].sum() / years),
                worst_event=float(per["rmb"].min()), excess_dd=dd, pos_years=int((yy["rmb"] > 0).sum()),
                by_year={str(y): round(v) for y, v in zip(yy["y"], yy["rmb"])}, by_event=[round(v) for v in per["rmb"]])


def main() -> None:
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    rows = []
    for (name, f), rule, slip in itertools.product(BOOKS.items(), ["全买", "前10", "前20"], [0.002, 0.003]):
        m = run(pl.read_parquet(X01 / f), rule, slip)
        trials.record("x03_rebal_select", params=dict(window=name, pool="沪深主板", rule=rule, cap=CAP, slip=slip, rate=RATE, min_fee=MIN_FEE),
                      metrics=m, sample="in_sample", note="OPP-005 G1 起草前：固定选股规则 + 超额回撤")
        rows.append(dict(window=name, rule=rule, slip=slip) | {k: v for k, v in m.items() if k not in ("by_year", "by_event")}
                    | {f"y{k}": v for k, v in m["by_year"].items()} | dict(by_event=str(m["by_event"])))
    # 补充：创业板权限（24 个月交易经验 + 10 万元资产）拿到之后的股票池，只看"全买"
    for (name, f), pool, slip in itertools.product(BOOKS.items(), ["主板+创业板", "全部"], [0.002, 0.003]):
        m = run(pl.read_parquet(X01 / f), "全买", slip, pool)
        trials.record("x03_rebal_select", params=dict(window=name, pool=pool, rule="全买", cap=CAP, slip=slip, rate=RATE, min_fee=MIN_FEE),
                      metrics=m, sample="in_sample", note="OPP-005 G1 起草前：权限扩大后的股票池")
        rows.append(dict(window=name, rule=f"全买·{pool}", slip=slip) | {k: v for k, v in m.items() if k not in ("by_year", "by_event")}
                    | {f"y{k}": v for k, v in m["by_year"].items()} | dict(by_event=str(m["by_event"])))
    print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
