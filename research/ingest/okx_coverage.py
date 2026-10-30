"""OKX 入库数据的覆盖率与质量检查（只看样本内，不读封存段）。

用法：uv run python research/ingest/okx_coverage.py
"""

import polars as pl

from lab import lake


def main() -> None:
    funding = lake.scan("okx_funding").collect()
    swap = lake.scan("okx_swap_1h").collect()
    spot = lake.scan("okx_spot_1h").collect()

    f = funding.group_by("inst_id").agg(
        pl.len().alias("n_funding"),
        pl.col("ts").min().alias("first"),
        pl.col("ts").max().alias("last"),
        pl.col("ts").diff().dt.total_hours().mode().first().alias("interval_h"),
        (pl.col("ts").diff().dt.total_hours() > 8.5).sum().alias("gaps_gt_8h"),
    ).with_columns(pl.col("inst_id").str.replace("-SWAP$", "").alias("family"))
    last_ts = funding["ts"].max()
    f = f.with_columns((pl.col("last") < last_ts - pl.duration(days=2)).alias("delisted"))

    s = swap.group_by("inst_id").agg(
        pl.len().alias("swap_hours"),
        (pl.col("n_1m") < 60).mean().alias("swap_gap_share"),
        pl.col("vol_quote").is_null().mean().alias("swap_vol_null_share"),
    )
    p = spot.group_by("inst_id").agg(pl.len().alias("spot_hours")).rename({"inst_id": "family"})
    t = f.join(s, on="inst_id", how="left").join(p, on="family", how="left")

    pl.Config.set_tbl_width_chars(200)
    print(f"样本内资金费最后时间：{last_ts}")
    print(
        t.group_by("delisted").agg(
            pl.len().alias("contracts"),
            pl.col("swap_hours").is_not_null().sum().alias("with_swap_1h"),
            pl.col("spot_hours").is_not_null().sum().alias("with_spot_1h"),
            pl.col("n_funding").sum().alias("funding_rows"),
            pl.col("swap_gap_share").median().alias("med_swap_gap_share"),
            pl.col("swap_vol_null_share").median().alias("med_swap_vol_null"),
        )
    )
    print("\n资金费结算间隔（众数）分布：")
    print(t["interval_h"].value_counts().sort("interval_h"))
    print("\n上线月份分布（按首个资金费）：")
    print(t.group_by(pl.col("first").dt.year().alias("year")).len().sort("year"))
    print(f"\n存在 >8.5h 资金费缺口的合约：{(t['gaps_gt_8h'] > 0).sum()} 个")


if __name__ == "__main__":
    main()
