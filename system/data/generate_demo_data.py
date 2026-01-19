"""示例数据生成脚本：用于创建演示用的 CSV 数据。"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

# 定位项目根目录与数据目录
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "system" / "data"


def daterange(start: date, end: date):
    # 生成日期序列，包含起止日期
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    # 通用 CSV 写入函数：先写表头再写行
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # 设置演示数据的时间区间
    start = date(2025, 9, 1)
    end = date(2026, 1, 19)

    prices = []
    macro = []
    news = []
    assets = ["GOLD_SPOT", "GOLD_MINER", "GOLD_ETF"]
    themes = ["央行购金", "地缘冲突", "通胀对冲"]

    base_price = 1900.0
    for idx, current in enumerate(daterange(start, end)):
        # 简单的涨跌模拟：一周内 4 天上涨、3 天回落
        drift = 0.3 if idx % 7 < 4 else -0.1
        base_price += drift
        for asset in assets:
            # 每个资产在基准价上略微偏移，模拟不同波动
            close = base_price * (1 + 0.01 * assets.index(asset))
            prices.append(
                {
                    "date": current.isoformat(),
                    "asset_id": asset,
                    "close": f"{close:.2f}",
                    "volume": str(100000 + idx),
                }
            )

        macro.append(
            {
                "date": current.isoformat(),
                "REAL_YIELD": f"{1.2 + 0.2 * (idx % 10) / 10:.2f}",
                "DXY": f"{100 + 0.3 * (idx % 5):.2f}",
                "INFLATION": f"{2.5 + 0.4 * (idx % 8) / 10:.2f}",
                "CB_BUY_INDEX": f"{0.6 + 0.1 * (idx % 6) / 10:.2f}",
                "GEO_RISK_INDEX": f"{0.4 + 0.2 * (idx % 9) / 10:.2f}",
            }
        )
        for theme in themes:
            # 新闻数量按主题与日期做简单循环
            news_count = 4 + (idx + themes.index(theme)) % 4
            news.append(
                {
                    "date": current.isoformat(),
                    "product": "GOLD",
                    "theme": theme,
                    "news_count": str(news_count),
                }
            )

    write_csv(DATA_DIR / "prices.csv", ["date", "asset_id", "close", "volume"], prices)
    # 宏观与新闻数据为示例 stub
    write_csv(
        DATA_DIR / "macro_stub.csv",
        ["date", "REAL_YIELD", "DXY", "INFLATION", "CB_BUY_INDEX", "GEO_RISK_INDEX"],
        macro,
    )
    write_csv(DATA_DIR / "news_stub.csv", ["date", "product", "theme", "news_count"], news)


if __name__ == "__main__":
    main()
