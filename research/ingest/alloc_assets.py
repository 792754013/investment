"""配置历史回看用的债券指数、美股指数和汇率（docs/research/ALLOC-history.md）。

来源（公开，无需登录）：
  中证全债 H11001、中证国债 H11006（财富指数，含利息再投资）：akshare stock_zh_index_hist_csindex，2002 年起
  标普 500 价格指数：akshare index_us_stock_sina（新浪 .INX），2004 年起；不含分红，分析时另加假设的股息
  美元兑人民币：FRED DEXCHUS（纽约中午买入价），1981 年起
输出 lake 数据集 alloc_px：asset、ts、close。A 股指数 ts 取 07:00 UTC（北京 15:00 收盘），
美股取 21:00 UTC，汇率取 17:00 UTC（纽约中午）。按年分区；封存起点之后的行由 lake.write 自动写入封存区。

用法：
  uv run python research/ingest/alloc_assets.py download   # → data/raw/alloc_assets/
  uv run python research/ingest/alloc_assets.py ingest     # → lake: alloc_px
"""

import io
import sys
from datetime import date

import akshare as ak
import httpx
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

CSI = {"H11001": "cn_bond_all", "H11006": "cn_bond_gov"}
HOUR = {"cn_bond_all": 7, "cn_bond_gov": 7, "spx_price": 21, "usdcny": 17}


def root():
    return raw_dir() / "alloc_assets"


def cmd_download() -> None:
    root().mkdir(parents=True, exist_ok=True)
    end = date.today().strftime("%Y%m%d")
    for s, name in CSI.items():
        x = ak.stock_zh_index_hist_csindex(symbol=s, start_date="20020101", end_date=end)
        x.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]].to_parquet(root() / f"{name}.parquet")
        print(name, len(x))
    x = ak.index_us_stock_sina(symbol=".INX")
    x[["date", "close"]].to_parquet(root() / "spx_price.parquet")
    print("spx_price", len(x))
    r = httpx.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": "DEXCHUS"}, timeout=60)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text))
    d.columns = ["date", "close"]
    d["close"] = pd.to_numeric(d["close"], errors="coerce")  # FRED 用 "." 表示缺失
    d.dropna().to_parquet(root() / "usdcny.parquet")
    print("usdcny", len(d))


def cmd_ingest() -> None:
    rows = []
    for p in sorted(root().glob("*.parquet")):
        x = pd.read_parquet(p)
        x["asset"] = p.stem
        rows.append(pl.from_pandas(x[["date", "close", "asset"]].astype({"date": str})))
    d = pl.concat(rows).with_columns(pl.col("date").str.slice(0, 10).str.to_date(), pl.col("close").cast(pl.Float64))
    hours = pl.col("asset").replace_strict(HOUR, return_dtype=pl.Int64)
    d = d.with_columns((pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=hours)).dt.replace_time_zone("UTC").alias("ts"),
                       pl.col("date").dt.year().alias("_y")).drop("date").filter(pl.col("close") > 0)
    for (y,), g in d.group_by(["_y"]):
        lake.write("alloc_px", g.drop("_y").sort("asset", "ts"), ts_col="ts", part=str(y))
    print("alloc_px", d.height, "行")


if __name__ == "__main__":
    {"download": cmd_download, "ingest": cmd_ingest}[sys.argv[1]]()
