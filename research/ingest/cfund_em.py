"""封闭 / 定开基金的开放事件和累计净值（OPP-004 快速否证）。

候选基金：OPP-003 已入库的 LOF 里，样本内折价 < -2% 的天数占比 ≥ 20% 的（由分析脚本选出后作为参数传入）。

来源（公开，无需登录）：
  公告标题  东方财富基金 F10 公告接口 api.fund.eastmoney.com/f10/JJGG，type=0（全部类别）
  累计净值  akshare fund_open_fund_info_em(indicator="累计净值走势")，用来还原分红
只看标题，挑出"进入开放期"类公告：
  open_window  开放申购、赎回 / 开放日常申购、赎回 / 进入开放期 / 封闭期届满 / 转为上市开放式（LOF）
  排除 暂停、恢复（节假日类）、终止销售机构、风险提示等
开放期的具体起止日写在正文里，标题拿不到；分析时以公告日为准，退出价取公告日之后的场内价格（见分析脚本）。

用法：
  uv run python research/ingest/cfund_em.py download 506000 160325 ...   # → data/raw/cfund/
  uv run python research/ingest/cfund_em.py ingest                       # → lake: cfund_events, cfund_accnav
"""

import re
import sys
import time

import akshare as ak
import httpx
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

API = "https://api.fund.eastmoney.com/f10/JJGG"
HEADERS = {"User-Agent": "Mozilla/5.0 research", "Referer": "https://fundf10.eastmoney.com/"}


def root():
    return raw_dir() / "cfund"


def titles(c: httpx.Client, code: str) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        d = c.get(API, params=dict(fundcode=code, pageIndex=page, pageSize=100, type=0)).json()
        items = d.get("Data") or []
        rows += [dict(code=code, title=x.get("TITLE", ""), date=str(x.get("PUBLISHDATEDesc") or x.get("PUBLISHDATE"))[:10])
                 for x in items]
        if len(rows) >= (d.get("TotalCount") or 0) or not items:
            break
        page += 1
        time.sleep(0.5)
    return pd.DataFrame(rows, columns=["code", "title", "date"])


def cmd_download(codes: list[str]) -> None:
    (root() / "notices").mkdir(parents=True, exist_ok=True)
    (root() / "accnav").mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=30) as c:
        for code in codes:
            p = root() / "notices" / f"{code}.parquet"
            if not p.exists():
                titles(c, code).to_parquet(p)
                time.sleep(0.5)
            q = root() / "accnav" / f"{code}.parquet"
            if not q.exists():
                ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势").to_parquet(q)
                time.sleep(0.4)
            print(code, flush=True)


def classify(title: str) -> str | None:
    if re.search(r"暂停|恢复|节假|终止.*销售|风险提示|提示性|定期报告|季度报告|年度报告|中期报告|招募说明书|资料概要", title):
        return None
    if re.search(r"开放(日常)?申购.{0,3}赎回|进入(第.{1,3}个)?开放期|开放期.{0,6}(安排|申购|赎回)|封闭期届满|转为上市开放式|转型为.*上市开放式",
                 title):
        return "open_window"
    return None


def to_ts(col: str) -> pl.Expr:
    return (pl.col(col).cast(pl.Date).cast(pl.Datetime("us")) + pl.duration(hours=7)).dt.replace_time_zone("UTC")


def cmd_ingest() -> None:
    n = pl.from_pandas(pd.concat([pd.read_parquet(p) for p in sorted((root() / "notices").glob("*.parquet"))]))
    n = n.with_columns(pl.col("title").map_elements(classify, return_dtype=pl.String).alias("kind"),
                       pl.col("date").str.to_date(strict=False)).drop_nulls(["kind", "date"]).unique(["code", "title", "date"])
    n = n.with_columns(to_ts("date").alias("ts")).drop("date")
    print(n.group_by("kind").len())
    lake.write("cfund_events", n.sort("code", "ts"), ts_col="ts", part="all")

    frames = []
    for p in sorted((root() / "accnav").glob("*.parquet")):
        d = pd.read_parquet(p)
        if d.empty:
            continue
        d = d.rename(columns={"净值日期": "date", "累计净值": "accnav"})[["date", "accnav"]]
        d["code"] = p.stem
        frames.append(pl.from_pandas(d))
    a = pl.concat(frames, how="vertical_relaxed").with_columns(pl.col("date").cast(pl.Utf8).str.to_date(),
                                                               pl.col("accnav").cast(pl.Float64))
    a = a.with_columns(to_ts("date").alias("ts"), pl.col("date").dt.year().alias("_y")).drop("date")
    for (y,), g in a.group_by(["_y"]):
        lake.write("cfund_accnav", g.drop("_y").sort("code", "ts"), ts_col="ts", part=str(y))
    print("cfund_accnav", a.height, "行")


if __name__ == "__main__":
    if sys.argv[1] == "download":
        cmd_download(sys.argv[2:])
    else:
        cmd_ingest()
