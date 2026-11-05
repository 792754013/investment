"""A 股宽基指数长历史日线（ATREND 快速否证，docs/research/ATREND-plan.md）。

来源（公开，无需登录）：
  中证全收益指数（含分红再投资）：akshare stock_zh_index_hist_csindex，H00300 / H00905 / H00016 / H00852，2004 年起
  创业板指（价格指数）：akshare stock_zh_index_daily（新浪）sz399006，2010 年起
输出 lake 数据集 idx_long：index（代码）、ts（交易日 07:00 UTC，即北京时间 15:00 收盘）、close。按年分区。

用法：
  uv run python research/ingest/cn_index_long.py download   # → data/raw/cn_index_long/
  uv run python research/ingest/cn_index_long.py ingest     # → lake: idx_long
"""

import sys
from datetime import date

import akshare as ak
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

CSI = ["H00300", "H00905", "H00016", "H00852"]


def root():
    return raw_dir() / "cn_index_long"


def cmd_download() -> None:
    root().mkdir(parents=True, exist_ok=True)
    end = date.today().strftime("%Y%m%d")
    for s in CSI:
        x = ak.stock_zh_index_hist_csindex(symbol=s, start_date="20040101", end_date=end)
        x.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]].to_parquet(root() / f"{s}.parquet")
        print(s, len(x))
    x = ak.stock_zh_index_daily(symbol="sz399006")
    x[["date", "close"]].to_parquet(root() / "399006.parquet")
    print("399006", len(x))


def cmd_ingest() -> None:
    rows = []
    for p in sorted(root().glob("*.parquet")):
        x = pd.read_parquet(p)
        x["index"] = p.stem
        rows.append(pl.from_pandas(x[["date", "close", "index"]].astype({"date": str})))
    d = pl.concat(rows).with_columns(pl.col("date").str.slice(0, 10).str.to_date(), pl.col("close").cast(pl.Float64))
    d = d.with_columns((pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=7)).dt.replace_time_zone("UTC").alias("ts"),
                       pl.col("date").dt.year().alias("_y")).drop("date").filter(pl.col("close") > 0)
    for (y,), g in d.group_by(["_y"]):
        lake.write("idx_long", g.drop("_y").sort("index", "ts"), ts_col="ts", part=str(y))
    print("idx_long", d.height, "行")


if __name__ == "__main__":
    {"download": cmd_download, "ingest": cmd_ingest}[sys.argv[1]]()
