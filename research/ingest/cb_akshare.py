"""下载 A 股可转债免费数据并入库（二期 P0 第 5 步，CB-P0-plan.md）。

来源全部通过 akshare（东方财富、新浪、巨潮、中证）：
  cb_list      全部转债清单（含已退市），东方财富 bond_zh_cov。注意：正股价、评级等是当前快照，有前视，回测不得用来筛选
  cb_daily     转债日线，新浪 bond_zh_hs_cov_daily，volume 单位是张（成交额 ≈ volume × 价格，已用实时行情核对）
  cb_value     纯债价值、转股价值、溢价率日序列，东方财富 bond_zh_cov_value_analysis
  cb_issue     发行条款（初始转股价、到期日等），巨潮 bond_cov_issue_cninfo
  stock_daily  正股日线（不复权，算转股价值用），东方财富 stock_zh_a_hist
  cb_index     中证转债指数 000832、中证货币基金指数 H11025 日线，中证指数官网（新浪源只到 2018 年，不用）

时间约定：交易日收盘 15:00 北京时间 = 07:00 UTC，ts 一律取这个时刻。

用法（可重复运行，已下载的文件会跳过）：
  uv run python research/ingest/cb_akshare.py download   # → data/raw/cb/
  uv run python research/ingest/cb_akshare.py ingest     # → lake
"""

import sys
import time
from datetime import date

import akshare as ak
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

SLEEP = 0.4
FIRST_ISSUE_YEAR = 2007
INDEXES = ["000832", "H11025"]  # 中证转债、中证货币基金


def cb_raw():
    return raw_dir() / "cb"


def to_ts(col: str) -> pl.Expr:
    """交易日 → 当天收盘时刻 07:00 UTC。"""
    return (pl.col(col).cast(pl.Date).cast(pl.Datetime("us")) + pl.duration(hours=7)).dt.replace_time_zone("UTC")


def exchange_prefix(code: str) -> str:
    return "sh" if code.startswith(("11", "13")) else "sz"


def fetch(path, f, retries: int = 3) -> bool:
    """下载一次并保存为 parquet；已存在则跳过。空结果也落一个空文件，避免反复请求。"""
    if path.exists():
        return True
    for i in range(retries):
        try:
            df = f()
            path.parent.mkdir(parents=True, exist_ok=True)
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_parquet(path)
            time.sleep(SLEEP)
            return True
        except Exception as e:  # 网络错误、接口改版等，记下来继续
            print(f"  失败 {path.name} 第 {i + 1} 次：{e!r}"[:200], flush=True)
            time.sleep(2 * (i + 1))
    return False


def cmd_download() -> None:
    root = cb_raw()
    fetch(root / "cb_list.parquet", ak.bond_zh_cov)
    lst = pd.read_parquet(root / "cb_list.parquet")
    for code in INDEXES:
        fetch(root / f"cb_index_{code}.parquet", lambda code=code: ak.stock_zh_index_hist_csindex(
            symbol=code, start_date="20020101", end_date=date.today().strftime("%Y%m%d")))
    for y in range(FIRST_ISSUE_YEAR, date.today().year + 1):
        fetch(root / "issue" / f"{y}.parquet",
              lambda y=y: ak.bond_cov_issue_cninfo(start_date=f"{y}0101", end_date=f"{y}1231"))

    bonds = lst[lst["上市时间"].notna()]["债券代码"].astype(str).tolist()
    stocks = sorted(set(lst["正股代码"].dropna().astype(str)))
    failed = []
    for i, code in enumerate(bonds):
        ok1 = fetch(root / "daily" / f"{code}.parquet",
                    lambda: ak.bond_zh_hs_cov_daily(symbol=exchange_prefix(code) + code))
        ok2 = fetch(root / "value" / f"{code}.parquet", lambda: ak.bond_zh_cov_value_analysis(symbol=code))
        if not (ok1 and ok2):
            failed.append(code)
        if i % 100 == 0:
            print(f"转债 {i}/{len(bonds)}", flush=True)
    for i, code in enumerate(stocks):
        if not fetch(root / "stock" / f"{code}.parquet",
                     lambda: ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20060101",
                                                end_date=date.today().strftime("%Y%m%d"), adjust="")):
            failed.append(code)
        if i % 100 == 0:
            print(f"正股 {i}/{len(stocks)}", flush=True)
    print(f"完成。转债 {len(bonds)}，正股 {len(stocks)}，失败 {len(failed)}：{failed[:30]}")


def read_dir(sub: str, rename: dict, code_col: str) -> pl.DataFrame:
    frames = []
    for p in sorted((cb_raw() / sub).glob("*.parquet")):
        d = pd.read_parquet(p)
        if d.empty:
            continue
        d = d.rename(columns=rename)[list(rename.values())]
        d[code_col] = p.stem
        frames.append(pl.from_pandas(d))
    return pl.concat(frames, how="vertical_relaxed")


def write_by_year(dataset: str, df: pl.DataFrame) -> None:
    df = df.with_columns(pl.col("ts").dt.year().alias("_y"))
    for (y,), g in df.group_by(["_y"]):
        lake.write(dataset, g.drop("_y").sort("ts"), ts_col="ts", part=str(y))
    print(dataset, df.height, "行")


def cmd_ingest() -> None:
    root = cb_raw()
    lst = pl.from_pandas(pd.read_parquet(root / "cb_list.parquet")).rename({
        "债券代码": "bond_code", "债券简称": "bond_name", "申购日期": "subscribe_date", "正股代码": "stock_code",
        "正股简称": "stock_name", "转股价": "conv_price_now", "发行规模": "issue_size_100m",
        "上市时间": "list_date", "信用评级": "rating_now",
    }).select("bond_code", "bond_name", "subscribe_date", "stock_code", "stock_name", "conv_price_now",
              "issue_size_100m", "list_date", "rating_now")
    lst = lst.with_columns(to_ts("subscribe_date").alias("ts"))
    lake.write("cb_list", lst, ts_col="ts", part="all")
    print("cb_list", lst.height, "行")

    daily = read_dir("daily", {"date": "date", "open": "open", "high": "high", "low": "low",
                               "close": "close", "volume": "volume"}, "bond_code")
    write_by_year("cb_daily", daily.with_columns(to_ts("date").alias("ts")).drop("date"))

    value = read_dir("value", {"日期": "date", "收盘价": "close", "纯债价值": "bond_value",
                               "转股价值": "conv_value", "纯债溢价率": "bond_premium",
                               "转股溢价率": "conv_premium"}, "bond_code")
    write_by_year("cb_value", value.with_columns(to_ts("date").alias("ts")).drop("date"))

    stock = read_dir("stock", {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
                               "成交量": "volume", "成交额": "amount", "换手率": "turnover"}, "stock_code")
    write_by_year("stock_daily", stock.with_columns(to_ts("date").alias("ts")).drop("date"))

    issue = pl.concat([pl.from_pandas(pd.read_parquet(p).astype(str)) for p in sorted((root / "issue").glob("*.parquet"))
                       if not pd.read_parquet(p).empty], how="vertical_relaxed")
    issue = issue.rename({"债券代码": "bond_code", "公告日期": "announce_date"}).with_columns(
        pl.col("announce_date").str.to_date(strict=False)).with_columns(to_ts("announce_date").alias("ts"))
    lake.write("cb_issue", issue, ts_col="ts", part="all")
    print("cb_issue", issue.height, "行")

    for code in INDEXES:
        idx = pl.from_pandas(pd.read_parquet(root / f"cb_index_{code}.parquet")).rename(
            {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close"}
        ).select("date", "open", "high", "low", "close")
        idx = idx.with_columns(to_ts("date").alias("ts"), pl.lit(code).alias("index_code")).drop("date")
        lake.write("cb_index", idx, ts_col="ts", part=code)
        print("cb_index", code, idx.height, "行")


if __name__ == "__main__":
    {"download": cmd_download, "ingest": cmd_ingest}[sys.argv[1]]()
