"""下载 LOF 的销售类公告，按标题重建申购状态时间线（OPP-003 快速否证）。

来源：东方财富基金 F10 公告接口（api.fund.eastmoney.com/f10/JJGG，type=5 基金销售），公开，无需登录。
只看标题，不解析正文，所以限额的具体金额拿不到，只区分三种状态：
  suspend   暂停申购（标题不带具体日期的，视为持续到下一次"恢复申购"）
  limited   恢复申购但暂停或限制大额申购，或调整大额申购限制
  open      恢复申购（不带大额限制字样）
节假日、休市类公告，以及标题带具体日期的"X 月 X 日暂停申购、赎回"，只影响当天，单独记为 holiday，不改变状态。
生效日：公告日之后的第一个交易日（悲观：公告当天不能按新状态操作）。

用法：
  uv run python research/ingest/fund_notices_em.py download 161129 161116 ...   # → data/raw/lof/notices/
  uv run python research/ingest/fund_notices_em.py ingest                        # → lake: lof_status_events
"""

import re
import sys
import time

import httpx
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

API = "https://api.fund.eastmoney.com/f10/JJGG"
HEADERS = {"User-Agent": "Mozilla/5.0 research", "Referer": "https://fundf10.eastmoney.com/"}


def root():
    return raw_dir() / "lof" / "notices"


def cmd_download(codes: list[str]) -> None:
    root().mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=30) as c:
        for code in codes:
            path = root() / f"{code}.parquet"
            if path.exists():
                continue
            rows, page = [], 1
            while True:
                d = c.get(API, params=dict(fundcode=code, pageIndex=page, pageSize=100, type=5)).json()
                items = d.get("Data") or []
                rows += [dict(code=code, title=x.get("TITLE", ""), date=str(x.get("PUBLISHDATEDesc") or x.get("PUBLISHDATE"))[:10])
                         for x in items]
                if len(rows) >= (d.get("TotalCount") or 0) or not items:
                    break
                page += 1
                time.sleep(0.5)
            pd.DataFrame(rows, columns=["code", "title", "date"]).to_parquet(path)
            print(code, len(rows), flush=True)
            time.sleep(0.5)


def classify(title: str) -> str | None:
    if "申购" not in title:
        return None
    if re.search(r"渠道|支付|直销|系统维护", title):
        return None
    if re.search(r"节假日|休市|圣诞|元旦|新年|假期|暂停及恢复", title):
        return "holiday"
    if "暂停" in title and "赎回" in title and "大额" not in title:
        return "holiday"  # 申购、赎回同时暂停，几乎都是境外休市
    if "大额" not in title and "暂停" in title and re.search(r"\d{1,2}月\d{1,2}日", title):
        return "holiday"  # 标题写明具体日期的暂停，多为境外休市，只影响当天
    if "恢复大额申购" in title and not re.search(r"限制|暂停大额", title):
        return "open"
    if re.search(r"(调整|限制|暂停)大额申购|大额申购.*限制|限额", title):
        return "limited"
    if "恢复" in title:
        return "open"
    if "暂停" in title:
        return "suspend"
    return None


def cmd_ingest() -> None:
    frames = [pd.read_parquet(p) for p in sorted(root().glob("*.parquet"))]
    n = pl.from_pandas(pd.concat(frames)).with_columns(
        pl.col("title").map_elements(classify, return_dtype=pl.String).alias("state"),
        pl.col("date").str.to_date(strict=False),
    ).drop_nulls(["state", "date"]).unique(["code", "title", "date"])
    n = n.with_columns((pl.col("date").cast(pl.Datetime("us")) + pl.duration(hours=7)).dt.replace_time_zone("UTC").alias("ts"))
    print(n.group_by("state").len().sort("state"))
    n = n.with_columns(pl.col("ts").dt.year().alias("_y"))
    for (y,), g in n.group_by(["_y"]):
        lake.write("lof_status_events", g.drop("_y").sort("code", "ts"), ts_col="ts", part=str(y))


if __name__ == "__main__":
    if sys.argv[1] == "download":
        cmd_download(sys.argv[2:])
    else:
        cmd_ingest()
