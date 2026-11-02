"""OPP-003 快速否证：场内 LOF 溢价申购套利（plan.md 5.1 节）。

问题：按悲观口径，每天在溢价足够高的 LOF 上做一次"场内申购 → 份额到账后场内卖出"，
     在每日申购限额下一年能多赚多少元？低于 3,000 元就放弃。

口径（悲观）：
  信号：申购日 T 只能看到已公布的净值。境内 LOF 用 T-1 的溢价（T-1 收盘价 / T-1 净值），
        QDII 用 T-2 的溢价（境外净值晚一天公布）。
  成交：按 T 日净值申购（申购时未知），份额到账后按 T+H 日收盘价卖出：境内 H=2，QDII H=3。
  成本：申购费（情形：0.15% / 0.6% / 1.2%）+ 卖出佣金万 2（情形：无最低 / 每笔最低 5 元）
        + 经手费 0.001% + 卖出冲击 0.2%。
  容量：每只基金每天申购 LIMIT 元（情形 100 / 1,000 / 10,000），卖出额不超过该基金 20 日均成交额的 10%，超过就不做。
  可申购：**假设每天都能申购**。历史上溢价高的时候往往暂停或限额申购，所以这是对策略最有利的上限，
        只用于否证：上限都不够 3,000 元就放弃。
只读样本内（封存起点 2026-05-01 之前）。lof_list 的当前申购状态有前视，只作描述，不参与筛选。

用法：uv run python research/analysis/q01_lof_premium.py
"""

import itertools

import polars as pl

from lab import lake, trials
from lab.paths import data_root

THETAS = [0.02, 0.03, 0.05, 0.08]
SUB_FEES = [0.0015, 0.006, 0.012]
MIN_FEES = [0.0, 5.0]
LIMITS = [100, 1_000, 10_000]
RATE, EXCH, SLIP = 0.0002, 0.00001, 0.002
H = {False: 2, True: 3}
LAG = {False: 1, True: 2}
G0_RMB = 3_000
OUT = data_root() / "research" / "q01"


def panel() -> pl.DataFrame:
    price = lake.scan("lof_price").collect()
    nav = lake.scan("lof_nav").collect()
    lst = lake.scan("lof_list").select("code", "name", "is_qdii").collect()
    p = (
        price.join(nav, on=["code", "ts"], how="inner").join(lst, on="code", how="inner")
        .filter((pl.col("close") > 0) & (pl.col("nav") > 0))
        .sort("code", "ts")
        .with_columns((pl.col("close") / pl.col("nav") - 1).alias("prem"),
                      pl.col("amount").rolling_mean(20, min_samples=5).over("code").alias("amt20"),
                      pl.int_range(pl.len()).over("code").alias("i"))
    )
    # 信号：LAG 天前的溢价；卖出：H 天后的收盘价和成交额（按该基金自己的交易日序号）
    out = []
    for q in (False, True):
        g = p.filter(pl.col("is_qdii") == q)
        g = g.with_columns(
            pl.col("prem").shift(LAG[q]).over("code").alias("signal"),
            pl.col("close").shift(-H[q]).over("code").alias("sell_px"),
            pl.col("amt20").shift(-H[q]).over("code").alias("sell_amt20"),
        )
        out.append(g)
    return pl.concat(out).drop_nulls(["signal", "sell_px"]).with_columns(
        (pl.col("sell_px") / pl.col("nav") - 1).alias("gross"), pl.col("ts").dt.year().alias("year"))


def scenario(p: pl.DataFrame, theta: float, sub_fee: float, min_fee: float, limit: int) -> pl.DataFrame:
    t = p.filter((pl.col("signal") >= theta) & (pl.col("sell_amt20") * 0.1 >= limit))
    sell_cost = pl.max_horizontal(pl.lit(limit * RATE), pl.lit(min_fee)) / limit + EXCH + SLIP
    return t.with_columns((pl.col("gross") - sub_fee - sell_cost).alias("net")).with_columns(
        (pl.col("net") * limit).alias("rmb"))


def main() -> None:
    p = panel()
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(4)

    # 描述：溢价分布
    desc = p.group_by("year", "is_qdii").agg(
        pl.col("code").n_unique().alias("funds"),
        pl.len().alias("fund_days"),
        pl.col("prem").median().alias("prem_med"),
        (pl.col("prem") >= 0.03).mean().alias("share_ge3pct"),
        (pl.col("prem") >= 0.05).mean().alias("share_ge5pct"),
        (pl.col("prem") >= 0.03).sum().alias("days_ge3pct"),
    ).sort("is_qdii", "year")
    print("表 1 溢价分布（场内收盘价 / 同日净值 − 1）\n", desc)

    # 每笔毛收益：信号溢价 vs 实际拿到的（申购净值 → T+H 卖出）
    conv = p.with_columns(pl.col("signal").cut([0.02, 0.03, 0.05, 0.08, 0.15]).alias("bucket")).group_by(
        "is_qdii", "bucket").agg(pl.len().alias("n"), pl.col("gross").mean().alias("gross_mean"),
                                 pl.col("gross").median().alias("gross_med"),
                                 pl.col("gross").quantile(0.1).alias("gross_p10"),
                                 (pl.col("gross") > 0).mean().alias("win")).sort("is_qdii", "bucket")
    print("\n表 2 按信号溢价分组：申购净值到 T+H 收盘卖出的毛收益（未扣成本）\n", conv)

    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for theta, fee, mn in itertools.product(THETAS, SUB_FEES, MIN_FEES):
        per_limit = {}
        for limit in LIMITS:
            s = scenario(p, theta, fee, mn, limit)
            yearly = s.group_by("year").agg(pl.col("rmb").sum()).sort("year")
            full = yearly.filter(pl.col("year").is_between(2021, 2025))
            per_limit[limit] = dict(
                trades=s.height, net_mean=s["net"].mean(), net_med=s["net"].median(),
                win=(s["net"] > 0).mean() if s.height else None,
                rmb_per_year=float(full["rmb"].mean()) if full.height else 0.0,
                rmb_by_year={str(y): round(v) for y, v in zip(yearly["year"], yearly["rmb"])},
            )
            rows.append(dict(theta=theta, sub_fee=fee, min_fee=mn, limit=limit) | {
                k: v for k, v in per_limit[limit].items() if k != "rmb_by_year"} | {
                f"y{k}": v for k, v in per_limit[limit]["rmb_by_year"].items()})
        trials.record("q01_lof_premium", params=dict(theta=theta, sub_fee=fee, min_fee=mn, slip=SLIP, h=H, lag=LAG),
                      metrics={str(lim): {k: per_limit[lim][k] for k in ("trades", "net_mean", "rmb_per_year")}
                               for lim in LIMITS}, sample="in_sample",
                      note="快速否证，假设每天都能申购（对策略最有利的上限）")
    t = pl.DataFrame(rows)
    t.write_parquet(OUT / "scenarios.parquet")
    p.write_parquet(OUT / "panel.parquet")
    print("\n表 3 各情形：每年净额（元，2021～2025 年平均），G0 = 3,000 元\n",
          t.select("theta", "sub_fee", "min_fee", "limit", "trades", "net_mean", "net_med", "win", "rmb_per_year"))
    print("\n表 4 分年份（元）\n", t.filter(pl.col("limit") == 1_000).drop("trades", "net_mean", "net_med", "win"))


if __name__ == "__main__":
    main()
