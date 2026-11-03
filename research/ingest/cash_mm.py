"""E3 现金管理的数据（plan.md 4.2 节）。

来源（公开，无需登录）：
  - 交易所国债逆回购日线：akshare bond_zh_hs_daily（新浪），沪市 2040xx、深市 1318xx 各期限，利率为年化 %，2016-11 起。
  - 货币基金：fund_money_fund_info_em 取每只的每万份收益、7 日年化、申购状态（"场内买入"表示场内货基）。
    单只要翻页，约 30 秒，所以只取样本：
      场内货基：fund_name_em 里 511 / 159 开头的货币型，剔除 B 类（大额份额），全部取；
      场外货基：fund_money_fund_daily_em 当前名单里简称以 A 结尾或不带份额字母的，按代码排序每 6 只取 1 只。
    只覆盖目前仍存续的基金，已清盘的取不到。
  - 中证同业存单 AAA 指数（931059）：stock_zh_index_hist_csindex。
  - 交易日历：tool_trade_date_hist_sina（含未来日期，逆回购计息天数要用）。

输出 lake：repo_daily、mmf_daily、cash_idx、trade_cal。交易日按 07:00 UTC 记。

用法：
  uv run python research/ingest/cash_mm.py download   # → data/raw/cash/
  uv run python research/ingest/cash_mm.py ingest
"""

import sys
import time

import akshare as ak
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

# 代码 → 名义期限（天）
REPO = {"sh204001": 1, "sh204002": 2, "sh204003": 3, "sh204004": 4, "sh204007": 7, "sh204014": 14, "sh204028": 28,
        "sh204091": 91, "sh204182": 182,
        "sz131810": 1, "sz131811": 2, "sz131800": 3, "sz131809": 4, "sz131801": 7, "sz131802": 14, "sz131803": 28,
        "sz131805": 91, "sz131806": 182}


def root():
    return raw_dir() / "cash"


def retry(f, n: int = 3):
    for k in range(n):
        try:
            return f()
        except Exception:
            if k == n - 1:
                raise
            time.sleep(10 * (k + 1))


def cmd_download() -> None:
    for d in ["repo", "mmf"]:
        (root() / d).mkdir(parents=True, exist_ok=True)
    for c in REPO:
        p = root() / "repo" / f"{c}.parquet"
        if not p.exists():
            retry(lambda: ak.bond_zh_hs_daily(symbol=c)).to_parquet(p)
            time.sleep(1)
    p = root() / "cal.parquet"
    if not p.exists():
        ak.tool_trade_date_hist_sina().to_parquet(p)
    p = root() / "idx_931059.parquet"
    if not p.exists():
        ak.stock_zh_index_hist_csindex(symbol="931059", start_date="20160101", end_date="20261231").to_parquet(p)
    p = root() / "mmf_list.parquet"
    if not p.exists():
        ak.fund_money_fund_daily_em().to_parquet(p)
    q = root() / "fund_name_em.parquet"
    if not q.exists():
        ak.fund_name_em().to_parquet(q)
    fn = pd.read_parquet(q)
    on = fn[fn["基金类型"].str.contains("货币") & fn["基金代码"].str[:3].isin(["511", "159"]) & ~fn["基金简称"].str.endswith("B")]
    otc = pd.read_parquet(p)[["基金代码", "基金简称"]].astype(str)
    otc = otc[otc["基金简称"].str.endswith("A") | otc["基金简称"].str.endswith("币")].sort_values("基金代码")
    codes = on["基金代码"].tolist() + otc["基金代码"].iloc[::6].tolist()
    print("场内", len(on), "只，场外样本", len(otc.iloc[::6]), "只", flush=True)
    failed = []
    for i, c in enumerate(codes):
        q = root() / "mmf" / f"{c}.parquet"
        if q.exists():
            continue
        try:
            retry(lambda: ak.fund_money_fund_info_em(symbol=c)).to_parquet(q)
        except Exception as e:
            failed.append(c)
            print("失败", c, repr(e)[:100], flush=True)
        time.sleep(0.8)
        if i % 10 == 0:
            print(f"{i}/{len(codes)}", flush=True)
    print("完成，失败", len(failed), failed[:20])


def ts(col: str) -> pl.Expr:
    return (pl.col(col).cast(pl.Utf8).str.slice(0, 10).str.to_date().cast(pl.Datetime("us")) + pl.duration(hours=7)).dt.replace_time_zone("UTC")


def cmd_ingest() -> None:
    frames = []
    for c, n in REPO.items():
        x = pl.from_pandas(pd.read_parquet(root() / "repo" / f"{c}.parquet"))
        frames.append(x.select(pl.lit(c[2:]).alias("code"), pl.lit(c[:2]).alias("exch"), pl.lit(n).alias("tenor"),
                               ts("date").alias("ts"), *[pl.col(k).cast(pl.Float64) for k in ["open", "high", "low", "close", "volume"]]))
    r = pl.concat(frames).with_columns(pl.col("ts").dt.year().alias("_y"))
    for (y,), g in r.group_by(["_y"]):
        lake.write("repo_daily", g.drop("_y").sort("code", "ts"), ts_col="ts", part=str(y))
    print("repo_daily", r.height, "行", r["code"].n_unique(), "个代码")

    cal = pl.from_pandas(pd.read_parquet(root() / "cal.parquet")).select(ts("trade_date").alias("ts"))
    lake.write("trade_cal", cal.sort("ts"), ts_col="ts", part="all")
    print("trade_cal", cal.height, "天")

    ix = pl.from_pandas(pd.read_parquet(root() / "idx_931059.parquet")).select(
        pl.lit("同业存单AAA").alias("index"), ts("日期").alias("ts"), pl.col("收盘").cast(pl.Float64).alias("close"))
    lake.write("cash_idx", ix.sort("ts"), ts_col="ts", part="all")
    print("cash_idx", ix.height, "行")

    names = pl.from_pandas(pd.read_parquet(root() / "fund_name_em.parquet")[["基金代码", "基金简称"]].astype(str)).rename(
        {"基金代码": "code", "基金简称": "name"})
    frames = []
    for p in sorted((root() / "mmf").glob("*.parquet")):
        x = pd.read_parquet(p)
        if x.empty:
            continue
        x = pl.from_pandas(x.astype(str))
        frames.append(x.select(pl.lit(p.stem).alias("code"), ts("净值日期").alias("ts"),
                               pl.col("每万份收益").cast(pl.Float64, strict=False).alias("w_income"),
                               pl.col("7日年化收益率").cast(pl.Float64, strict=False).alias("y7"),
                               (pl.col("申购状态") == "场内买入").alias("on_exch")))
    m = pl.concat(frames).join(names, on="code", how="left").with_columns(pl.col("ts").dt.year().alias("_y"))
    for (y,), g in m.group_by(["_y"]):
        lake.write("mmf_daily", g.drop("_y").sort("code", "ts"), ts_col="ts", part=str(y))
    print("mmf_daily", m.height, "行", m["code"].n_unique(), "只")


if __name__ == "__main__":
    {"download": cmd_download, "ingest": cmd_ingest}[sys.argv[1]]()
