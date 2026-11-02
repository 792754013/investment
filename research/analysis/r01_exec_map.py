"""R1 可执行费率地图（plan.md 9.5 节，P1 粗略版）。

只读样本内（lab.lake.scan），不读封存段。
层层套上可执行条件，看原始资金费还剩多少；再扣除悲观成本，看按资本计的净年化。
这不是规则回测：入场点是"每一次满足条件的结算"，不考虑同时持仓数、资金分配和出场规则。

用法：uv run python research/analysis/r01_exec_map.py
输出：data/research/r01/*.parquet，并在终端打印各表
"""

import httpx
import polars as pl

from lab import lake, trials
from lab.paths import data_root

# ---- 参数（G1-criteria.md 第 2 节的悲观口径；品种池初值见 plan.md 9.6）----
MIN_LISTED_DAYS = 30
MIN_SPOT_TURNOVER = 5e6  # 现货 24h 成交额，USD
OI_PROXY_TURNOVER = 25e6  # 永续 24h 成交额，代替 OI ≥ 1000 万（2026-10-30 截面 OI/成交额中位 0.38）
OI_PROXY_GRID = [0, 10e6, 25e6, 50e6]
FEES = 2 * 0.0010 + 2 * 0.0005  # 现货、永续 taker 各开平一次
SLIP = 4 * 0.0010  # 每腿每次 0.10%
COST = FEES + SLIP  # 0.70% 名义
HOLD_DAYS = [3, 7, 14, 30]
CAPITAL_MULT = {"A": 2.0, "B": 1.3}  # A：永续 1 倍；B：统一账户取 plan 9.2 区间上限
EPISODE_GAP = pl.duration(days=7)  # 资金费断档超过 7 天视为下架后重新上线
DATA_START_GRACE = pl.datetime(2022, 2, 1, time_zone="UTC")  # 归档起点前就已上线的合约，视为老合约

OUT = data_root() / "research" / "r01"


def settlements() -> pl.DataFrame:
    f = lake.scan("okx_funding").collect().sort("inst_id", "ts")
    f = f.with_columns(
        pl.col("ts").dt.truncate("1h").alias("hour"),
        pl.col("inst_id").str.replace("-SWAP$", "").alias("spot_id"),
        (pl.col("ts").diff().over("inst_id") > EPISODE_GAP).fill_null(True).cum_sum().over("inst_id").alias("episode"),
    )
    f = f.with_columns(
        pl.col("ts").min().over("inst_id", "episode").alias("ep_start"),
        pl.col("ts").max().over("inst_id", "episode").alias("ep_end"),
    )
    # 结算间隔：与上一次结算的间隔，取整到 1/2/4/8 小时；首条用下一条的间隔
    gap = pl.col("ts").diff().over("inst_id", "episode").dt.total_minutes() / 60
    nxt = (pl.col("ts").shift(-1) - pl.col("ts")).over("inst_id", "episode").dt.total_minutes() / 60
    f = f.with_columns(pl.coalesce(gap, nxt, pl.lit(8.0)).alias("raw_iv"))
    f = f.with_columns(
        pl.when(pl.col("raw_iv") < 1.5).then(1).when(pl.col("raw_iv") < 3).then(2)
        .when(pl.col("raw_iv") < 6).then(4).otherwise(8).alias("interval_h")
    )
    old = (pl.col("ep_start") < DATA_START_GRACE) & (pl.col("episode") == 1)
    return f.with_columns(
        (pl.col("funding_rate") * 24 * 365 / pl.col("interval_h")).alias("apr_raw"),
        pl.when(old).then(pl.lit(9999.0))
        .otherwise((pl.col("ts") - pl.col("ep_start")).dt.total_hours() / 24).alias("listed_days"),
    )


def rolling_turnover(dataset: str, id_col: str) -> pl.DataFrame:
    """每小时截至该小时之前的 24h 成交额（不含当前小时，避免前视）。"""
    k = lake.scan(dataset).select("inst_id", "ts", "vol_quote").collect().sort("inst_id", "ts")
    k = k.with_columns(
        pl.col("vol_quote").fill_null(float("nan")).rolling_sum_by("ts", window_size="24h").over("inst_id").alias("t24")
    ).with_columns(
        pl.col("t24").fill_nan(None),
        (pl.col("ts") + pl.duration(hours=1)).alias("hour"),  # 该小时收盘后才可知
    )
    return k.select(pl.col("inst_id").alias(id_col), "hour", "t24")


def collateral_set() -> set[str]:
    """OKX 当前可作统一账户抵押的币种（无历史，只能用当前名单近似）。"""
    d = httpx.get("https://www.okx.com/api/v5/public/discount-rate-interest-free-quota",
                  headers={"User-Agent": "Mozilla/5.0 research"}, timeout=30).json()["data"]
    return {x["ccy"] for x in d if x["details"] and float(x["details"][0]["discountRate"]) > 0}


def forward(f: pl.DataFrame) -> pl.DataFrame:
    """入场在结算 t 之后，收 (t, t+H] 内的资金费；同一上线段内，下架则截断并加倍滑点。"""
    f = f.sort("inst_id", "ts").with_columns(
        pl.col("funding_rate").cum_sum().over("inst_id", "episode").alias("cum")
    )
    out = f
    for h in HOLD_DAYS:
        tgt = f.select("inst_id", "episode", "ts", "cum").rename({"ts": "ts_r", "cum": f"cum_{h}"})
        out = (
            out.with_columns((pl.col("ts") + pl.duration(days=h)).alias("t_end"))
            .sort("t_end")
            .join_asof(tgt.sort("ts_r"), left_on="t_end", right_on="ts_r", by=["inst_id", "episode"], strategy="backward")
            .with_columns(
                (pl.col(f"cum_{h}") - pl.col("cum")).alias(f"fund_{h}"),
                (pl.col("ep_end") < pl.col("t_end")).alias(f"trunc_{h}"),
            )
            .drop("ts_r", f"cum_{h}", "t_end")
        )
        # 窗口超出样本内末尾的不计（不能用封存段补）
        last = f["ts"].max()
        out = out.with_columns(
            pl.when(pl.col("ts") + pl.duration(days=h) <= last).then(pl.col(f"fund_{h}")).alias(f"fund_{h}")
        )
        cost = pl.lit(COST) + pl.when(pl.col(f"trunc_{h}")).then(SLIP).otherwise(0.0)
        out = out.with_columns((pl.col(f"fund_{h}") - cost).alias(f"net_{h}"))
    return out.sort("inst_id", "ts")


def build() -> pl.DataFrame:
    f = settlements()
    spot = rolling_turnover("okx_spot_1h", "spot_id")
    swap = rolling_turnover("okx_swap_1h", "inst_id")
    f = (
        f.join(spot.rename({"t24": "spot_t24"}), on=["spot_id", "hour"], how="left")
        .join(swap.rename({"t24": "perp_t24"}), on=["inst_id", "hour"], how="left")
    )
    coll = collateral_set()
    f = f.with_columns(
        pl.col("spot_t24").is_not_null().alias("has_spot"),
        (pl.col("listed_days") >= MIN_LISTED_DAYS).alias("seasoned"),
        (pl.col("spot_t24") >= MIN_SPOT_TURNOVER).fill_null(False).alias("spot_liq"),
        (pl.col("perp_t24") >= OI_PROXY_TURNOVER).fill_null(False).alias("perp_liq"),
        pl.col("spot_id").str.replace("-USDT$", "").is_in(list(coll)).alias("collateral"),
    )
    f = f.with_columns(
        (pl.col("has_spot") & pl.col("seasoned") & pl.col("spot_liq") & pl.col("perp_liq")).alias("pool_a"),
        (pl.col("has_spot") & ~pl.col("seasoned") & pl.col("spot_liq") & pl.col("perp_liq")).alias("new_listing"),
    ).with_columns((pl.col("pool_a") & pl.col("collateral")).alias("pool_b"))
    return forward(f)


def stat(df: pl.DataFrame) -> list[pl.Expr]:
    return [
        pl.col("inst_id").n_unique().alias("instruments"),
        pl.len().alias("settlements"),
        # 时间加权：持有该集合的全部空头腿时，每小时实际收到的资金费年化（1h 结算的行不会被多算）
        (pl.col("funding_rate").sum() * 24 * 365 / pl.col("interval_h").sum()).alias("apr_tw"),
        pl.col("apr_raw").median().alias("apr_median"),
        (pl.col("apr_raw") > 0.10).mean().alias("share_gt10"),
        (pl.col("apr_raw") > 0.30).mean().alias("share_gt30"),
    ]


def funnel(f: pl.DataFrame) -> pl.DataFrame:
    layers = [
        ("0 全部", pl.lit(True)),
        ("1 有同所 USDT 现货", pl.col("has_spot")),
        ("2 + 上线 ≥30 天", pl.col("has_spot") & pl.col("seasoned")),
        ("3 + 现货 24h ≥ 500 万", pl.col("has_spot") & pl.col("seasoned") & pl.col("spot_liq")),
        ("4 + 永续 24h ≥ 2500 万（变体 A 池）", pl.col("pool_a")),
        ("5 + 可作抵押（变体 B 池）", pl.col("pool_b")),
        ("新上线子池（<30 天）", pl.col("new_listing")),
    ]
    return pl.concat([f.filter(c).select(pl.lit(n).alias("layer"), *stat(f)) for n, c in layers])


def net_table(f: pl.DataFrame, pool: str, variant: str) -> pl.DataFrame:
    """按资本计的净年化：无条件入场 vs 入场时费率处于池内前 10%。"""
    p = f.filter(pl.col(pool))
    q90 = p["apr_raw"].quantile(0.9)
    rows = []
    for cond, sub in (("全部结算", p), (f"入场费率 ≥ 池内 90 分位（{q90:.0%}）", p.filter(pl.col("apr_raw") >= q90))):
        for h in HOLD_DAYS:
            ann = sub[f"net_{h}"].drop_nulls() / CAPITAL_MULT[variant] * 365 / h
            rows.append(dict(
                pool=pool, variant=variant, entry=cond, hold_days=h, n=ann.len(),
                net_apr_mean=ann.mean(), net_apr_median=ann.median(),
                net_apr_p25=ann.quantile(0.25), win_share=(ann > 0).mean(),
            ))
    return pl.DataFrame(rows)


def main() -> None:
    f = build()
    OUT.mkdir(parents=True, exist_ok=True)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_float_precision(4)

    t1 = funnel(f)
    print("表 1 过滤漏斗（原始费率年化，按名义）\n", t1)

    sens = pl.concat([
        f.filter(pl.col("has_spot") & pl.col("seasoned") & pl.col("spot_liq")
                 & (pl.col("perp_t24") >= g).fill_null(False))
        .select(pl.lit(g / 1e6).alias("perp_t24_min_musd"), *stat(f))
        for g in OI_PROXY_GRID
    ])
    print("\n表 2 OI 代替阈值敏感性\n", sens)

    t3 = pl.concat([net_table(f, "pool_a", "A"), net_table(f, "pool_b", "B"), net_table(f, "new_listing", "B")])
    print(f"\n表 3 扣除悲观成本（{COST:.2%} 名义）后按资本计的净年化\n", t3)

    t4 = (
        f.filter(pl.col("pool_b"))
        .group_by(pl.col("ts").dt.year().alias("year"))
        .agg(*stat(f), (pl.col("net_14") / CAPITAL_MULT["B"] * 365 / 14).mean().alias("B_net_apr_h14_mean"))
        .sort("year")
    )
    print("\n表 4 变体 B 池按年份\n", t4)

    tier = (
        pl.when(pl.col("spot_t24") >= 100e6).then(pl.lit("≥1 亿"))
        .when(pl.col("spot_t24") >= 20e6).then(pl.lit("2000 万～1 亿"))
        .otherwise(pl.lit("500 万～2000 万"))
    )
    t5 = (
        f.filter(pl.col("pool_b")).with_columns(tier.alias("spot_tier"))
        .group_by("spot_tier")
        .agg(*stat(f), (pl.col("net_14") / CAPITAL_MULT["B"] * 365 / 14).mean().alias("B_net_apr_h14_mean"))
        .sort("spot_tier")
    )
    print("\n表 5 变体 B 池按现货成交额分层\n", t5)

    # 同一时刻可做的品种数：每天在 B 池中、当期费率年化 ≥ 15% 的品种数
    t6 = (
        f.filter(pl.col("pool_b"))
        .group_by(pl.col("ts").dt.date().alias("day"))
        .agg(
            pl.col("inst_id").n_unique().alias("pool_size"),
            pl.col("inst_id").filter(pl.col("apr_raw") >= 0.15).n_unique().alias("n_ge15"),
        )
        .group_by(pl.col("day").dt.year().alias("year"))
        .agg(
            pl.col("pool_size").median().alias("pool_size_med"),
            pl.col("n_ge15").median().alias("n_ge15_med"),
            pl.col("n_ge15").quantile(0.25).alias("n_ge15_p25"),
            pl.col("n_ge15").quantile(0.75).alias("n_ge15_p75"),
            (pl.col("n_ge15") == 0).mean().alias("days_none"),
        )
        .sort("year")
    )
    print("\n表 6 每天 B 池里当期费率年化 ≥15% 的品种数\n", t6)

    # 高费率入场的净年化按年份：看机会是否集中在某几年
    t7 = pl.concat([
        f.filter(pl.col("pool_b"), pl.col("apr_raw") >= thr, pl.col(f"net_{h}").is_not_null())
        .group_by(pl.col("ts").dt.year().alias("year"))
        .agg(
            pl.lit(thr).alias("entry_apr_min"), pl.lit(h).alias("hold_days"), pl.len().alias("n"),
            (pl.col(f"net_{h}") / CAPITAL_MULT["B"] * 365 / h).mean().alias("net_apr_mean"),
            (pl.col(f"net_{h}") / CAPITAL_MULT["B"] * 365 / h).median().alias("net_apr_median"),
        )
        for thr in (0.15, 0.30) for h in (14, 30)
    ]).sort("entry_apr_min", "hold_days", "year")
    print("\n表 7 变体 B 池高费率入场的净年化（按资本），按年份\n", t7)
    base = f.filter(pl.col("pool_b")).group_by(pl.col("ts").dt.year().alias("year")).agg(
        ((pl.col("apr_raw") - 0.1095).abs() < 0.002).mean().alias("share_at_base_rate")
    ).sort("year")
    print("\n表 8 变体 B 池中费率处于基准 10.95% 附近的结算占比\n", base)

    for name, df in (("funnel", t1), ("oi_proxy_sens", sens), ("net", t3), ("by_year", t4), ("by_tier", t5),
                     ("concurrency", t6), ("high_entry_by_year", t7), ("base_rate_share", base)):
        df.write_parquet(OUT / f"{name}.parquet")

    b = t3.filter(pl.col("pool") == "pool_b", pl.col("variant") == "B", pl.col("hold_days") == 14)
    trials.record(
        "r01_exec_map",
        params=dict(min_listed_days=MIN_LISTED_DAYS, min_spot_turnover=MIN_SPOT_TURNOVER,
                    oi_proxy_turnover=OI_PROXY_TURNOVER, cost=COST, capital_mult=CAPITAL_MULT),
        metrics=dict(zip(b["entry"].to_list(), b["net_apr_mean"].round(4).to_list())),
        sample="in_sample",
        note="描述性统计，非规则回测",
    )


if __name__ == "__main__":
    main()
