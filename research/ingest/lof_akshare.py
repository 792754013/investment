"""下载场内 LOF 的价格和净值历史并入库（OPP-003 快速否证，plan.md 5.1 节）。

来源全部通过 akshare（东方财富）：
  lof_list   当前上市的 LOF 清单 + 基金类型 + 当前申购状态和日累计限额。
             申购状态和限额只有 2026-11 的快照，有前视，回测不得用来筛选；ts 取该基金最早的净值日期。
             已终止上市的 LOF 不在清单里，存在幸存者偏差（结论里要写明）。
  lof_price  场内日线（收盘价、成交额），fund_lof_hist_em，不复权
  lof_nav    单位净值，fund_open_fund_info_em（净值日期 T 对应 T 日收盘后的净值，QDII 为境外 T 日收盘）

时间约定：交易日 / 净值日期 → 当天 07:00 UTC（北京时间 15:00），与可转债数据相同。

用法（可重复运行，已下载的文件会跳过）：
  uv run python research/ingest/lof_akshare.py download   # → data/raw/lof/
  uv run python research/ingest/lof_akshare.py ingest     # → lake
"""

import sys
import time
from datetime import date

import akshare as ak
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

START = "20210101"
SLEEP = 0.4


def root():
    return raw_dir() / "lof"


def to_ts(col: str) -> pl.Expr:
    return (pl.col(col).cast(pl.Date).cast(pl.Datetime("us")) + pl.duration(hours=7)).dt.replace_time_zone("UTC")


def fetch(path, f, retries: int = 3) -> bool:
    if path.exists():
        return True
    for i in range(retries):
        try:
            d = f()
            path.parent.mkdir(parents=True, exist_ok=True)
            (d if isinstance(d, pd.DataFrame) else pd.DataFrame()).to_parquet(path)
            time.sleep(SLEEP)
            return True
        except Exception as e:
            print(f"  失败 {path.name} 第 {i + 1} 次：{e!r}"[:200], flush=True)
            time.sleep(2 * (i + 1))
    return False


def cmd_download() -> None:
    fetch(root() / "spot.parquet", ak.fund_lof_spot_em)
    fetch(root() / "names.parquet", ak.fund_name_em)
    fetch(root() / "purchase.parquet", ak.fund_purchase_em)
    codes = pd.read_parquet(root() / "spot.parquet")["代码"].astype(str).tolist()
    end = date.today().strftime("%Y%m%d")
    failed = []
    for i, c in enumerate(codes):
        ok1 = fetch(root() / "price" / f"{c}.parquet",
                    lambda: ak.fund_lof_hist_em(symbol=c, period="daily", start_date=START, end_date=end, adjust=""))
        ok2 = fetch(root() / "nav" / f"{c}.parquet", lambda: ak.fund_open_fund_info_em(symbol=c, indicator="单位净值走势"))
        if not (ok1 and ok2):
            failed.append(c)
        if i % 50 == 0:
            print(f"LOF {i}/{len(codes)}", flush=True)
    print(f"完成：{len(codes)} 只，失败 {len(failed)}：{failed[:20]}")


def read_dir(sub: str, rename: dict) -> pl.DataFrame:
    frames = []
    for p in sorted((root() / sub).glob("*.parquet")):
        d = pd.read_parquet(p)
        if d.empty:
            continue
        d = d.rename(columns=rename)[list(rename.values())]
        d["code"] = p.stem
        frames.append(pl.from_pandas(d))
    return pl.concat(frames, how="vertical_relaxed")


def write_by_year(dataset: str, df: pl.DataFrame) -> None:
    df = df.with_columns(pl.col("ts").dt.year().alias("_y"))
    for (y,), g in df.group_by(["_y"]):
        lake.write(dataset, g.drop("_y").sort("code", "ts"), ts_col="ts", part=str(y))
    print(dataset, df.height, "行")


def cmd_ingest() -> None:
    price = read_dir("price", {"日期": "date", "收盘": "close", "成交额": "amount"})
    price = price.with_columns(pl.col("date").cast(pl.Utf8).str.to_date(), pl.col("close").cast(pl.Float64),
                               pl.col("amount").cast(pl.Float64))
    write_by_year("lof_price", price.with_columns(to_ts("date").alias("ts")).drop("date"))

    nav = read_dir("nav", {"净值日期": "date", "单位净值": "nav"})
    nav = nav.with_columns(pl.col("date").cast(pl.Utf8).str.to_date(), pl.col("nav").cast(pl.Float64))
    write_by_year("lof_nav", nav.with_columns(to_ts("date").alias("ts")).drop("date"))

    spot = pl.from_pandas(pd.read_parquet(root() / "spot.parquet")[["代码", "名称"]]).rename({"代码": "code", "名称": "name"})
    names = pl.from_pandas(pd.read_parquet(root() / "names.parquet")[["基金代码", "基金类型"]]).rename(
        {"基金代码": "code", "基金类型": "fund_type"})
    pur = pl.from_pandas(pd.read_parquet(root() / "purchase.parquet")[["基金代码", "申购状态", "日累计限定金额", "手续费"]]
                         .astype(str)).rename({"基金代码": "code", "申购状态": "purchase_status_now",
                                               "日累计限定金额": "daily_limit_now", "手续费": "fee_now"})
    first = nav.group_by("code").agg(pl.col("date").min().alias("first_nav_date"))
    lst = (spot.join(names, on="code", how="left").join(pur, on="code", how="left").join(first, on="code", how="left")
           .with_columns((pl.col("fund_type").str.starts_with("QDII") | pl.col("fund_type").str.contains("海外"))
                         .fill_null(False).alias("is_qdii"),  # 东方财富把多数 QDII 指数 LOF 归为"指数型-海外股票"
                         to_ts("first_nav_date").alias("ts")).drop_nulls("ts"))
    lake.write("lof_list", lst, ts_col="ts", part="all")
    print("lof_list", lst.height, "行，其中 QDII", lst["is_qdii"].sum())


if __name__ == "__main__":
    {"download": cmd_download, "ingest": cmd_ingest}[sys.argv[1]]()
