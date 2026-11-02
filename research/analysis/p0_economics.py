"""P0 经济性测算（plan.md 19 节 P0 第 3 步、附录 B 口径）。

只用假设的净年化，不读任何历史数据，不涉及回测。
用法：uv run python research/analysis/p0_economics.py
"""

import polars as pl

CAPITALS = [10_000, 15_000, 20_000, 50_000, 100_000]  # USD
NET_APRS = [0.05, 0.10, 0.15]  # 按资本计的净年化（已扣交易成本）
BENCHMARK = 0.04  # 被动替代收益（稳定币理财等），R11 再精确
INFRA_PER_YEAR = 240  # 精简版：实盘期 1 台小 VPS + R2，约 20 USD/月；完整版见 6.6.8 节约 840～1680
DEV_HOURS = 600  # 到 G3 的开发投入，12～18 人周取中间（含审阅 Claude 的产出）
OPS_HOURS_PER_YEAR = 180  # 22 节运营节奏：每日 15 分钟 + 每周 1 小时 + 每月 2 小时 + 每季半天
AMORTIZE_YEARS = 2  # 开发投入按 2 年摊销


def main() -> None:
    rows = []
    for cap in CAPITALS:
        for apr in NET_APRS:
            gross = cap * apr
            excess = cap * (apr - BENCHMARK) - INFRA_PER_YEAR
            hours = OPS_HOURS_PER_YEAR + DEV_HOURS / AMORTIZE_YEARS
            rows.append(dict(
                capital=cap, net_apr=apr,
                annual_usd=round(gross - INFRA_PER_YEAR),
                excess_over_benchmark_usd=round(excess),
                usd_per_hour=round(excess / hours, 1),
            ))
    for cap in CAPITALS[:3]:
        print(f"G0 最低净年化 @ {cap:,} USD = {BENCHMARK + INFRA_PER_YEAR / cap:.1%}")
    pl.Config.set_tbl_rows(50)
    print(pl.DataFrame(rows))
    print(f"\n假设：基准 {BENCHMARK:.0%}，基础设施 {INFRA_PER_YEAR} USD/年，"
          f"开发 {DEV_HOURS}h 按 {AMORTIZE_YEARS} 年摊销，运维 {OPS_HOURS_PER_YEAR}h/年 → 每年约 "
          f"{OPS_HOURS_PER_YEAR + DEV_HOURS // AMORTIZE_YEARS}h")


if __name__ == "__main__":
    main()
