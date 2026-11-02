"""下载可转债下修、强赎相关公告并分类入库（C3、C4 事件研究，CB-P0-plan.md 第 5 节）。

来源：巨潮资讯公告检索（akshare stock_zh_a_disclosure_report_cninfo），按关键词全市场检索，逐年拉取。
  关键词"向下修正"：下修预警、董事会提议、不下修、下修结果
  关键词"提前赎回"：强赎预警、强赎、不强赎、提示性公告
分类只看标题，规则见 classify()；券商核查意见、律师意见书等重复公告剔除。
标题里有转债简称的按简称匹配到转债；没有简称时，公告日该正股只有一只存续转债才匹配，否则丢弃。

时间：巨潮的"公告时间"是北京时间。时分为 00:00:00 的视为开盘前已可得（通常是前一晚发布、标注为当天），
有具体时分的按发布时刻。ts 统一转为 UTC。

用法（可重复运行，已下载的文件会跳过）：
  uv run python research/ingest/cb_notices.py download   # → data/raw/cb/notices/
  uv run python research/ingest/cb_notices.py ingest     # → lake: cb_notices
"""

import re
import sys
import time
from datetime import date

import akshare as ak
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

KEYWORDS = {"down": "向下修正", "call": "提前赎回"}
FIRST_YEAR = 2017
NOISE = re.compile(r"核查|法律意见|意见书|现金管理|理财|募集资金|保荐|独立董事|监事会|摘要")


def root():
    return raw_dir() / "cb" / "notices"


def cmd_download() -> None:
    for tag, kw in KEYWORDS.items():
        for y in range(FIRST_YEAR, date.today().year + 1):
            path = root() / f"{tag}_{y}.parquet"
            if path.exists():
                continue
            for i in range(3):
                try:
                    d = ak.stock_zh_a_disclosure_report_cninfo(
                        symbol="", market="沪深京", keyword=kw, category="",
                        start_date=f"{y}0101", end_date=f"{y}1231")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    d.to_parquet(path)
                    print(tag, y, len(d), flush=True)
                    break
                except Exception as e:
                    print(f"  失败 {tag} {y} 第 {i + 1} 次：{e!r}"[:200], flush=True)
                    time.sleep(5 * (i + 1))
            time.sleep(1)


def classify(title: str, tag: str) -> str | None:
    if NOISE.search(title):
        return None
    if tag == "down":
        if "向下修正" not in title:
            return None
        if re.search(r"预计|可能|即将|触发.*提示", title) and "不" not in title:
            return "down_warn"
        if re.search(r"不向下修正|不下修|暂不", title):
            return "down_no"
        if re.search(r"提议|拟向下修正|董事会", title):
            return "down_propose"
        return "down_done"  # 其余都是"向下修正 XX 转股价格（暨转股停复牌）的公告"，即下修结果
    if "赎回" not in title:
        return None
    if re.search(r"不提前赎回|暂不", title):
        return "call_no"
    if re.search(r"预计|可能|即将", title):
        return "call_warn"
    if re.search(r"提示性|第.次|结果|摘牌|停止交易|停止转股|最后", title):
        return "call_reminder"
    return "call_yes"


def cmd_ingest() -> None:
    frames = []
    for p in sorted(root().glob("*.parquet")):
        d = pd.read_parquet(p)
        if d.empty:
            continue
        d["tag"] = p.stem.split("_")[0]
        frames.append(d)
    raw = pd.concat(frames).rename(columns={"代码": "stock_code", "简称": "stock_name", "公告标题": "title",
                                            "公告时间": "ann_time", "公告链接": "url"})
    raw["title"] = raw["title"].str.replace(r"</?em>", "", regex=True)
    raw["kind"] = [classify(t, g) for t, g in zip(raw["title"], raw["tag"])]
    raw = raw[raw["kind"].notna()].drop_duplicates(["stock_code", "title", "ann_time"])
    n = pl.from_pandas(raw[["stock_code", "stock_name", "title", "ann_time", "url", "kind"]].astype(str))

    bonds = lake.scan("cb_list").select("bond_code", "bond_name", "stock_code", "list_date").collect()
    last = lake.scan("cb_daily").group_by("bond_code").agg(pl.col("ts").max().alias("last_ts")).collect()
    bonds = bonds.join(last, on="bond_code", how="left")

    n = n.with_columns(pl.when(pl.col("ann_time").str.len_chars() == 10).then(pl.col("ann_time") + " 00:00:00")
                       .otherwise(pl.col("ann_time")).alias("ann_time"))
    n = n.with_columns(pl.col("ann_time").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("t_cst"))
    n = n.with_columns(
        (pl.col("t_cst") - pl.duration(hours=8)).dt.replace_time_zone("UTC").alias("ts"),
        pl.col("t_cst").dt.date().alias("ann_date"),
        (pl.col("t_cst").dt.time() == pl.time(0)).alias("pre_open"),
    )
    cand = n.join(bonds, on="stock_code", how="inner")
    cand = cand.with_columns(
        pl.struct("title", "bond_name").map_elements(lambda r: r["bond_name"] in r["title"],
                                                     return_dtype=pl.Boolean).alias("name_hit"),
        (pl.col("list_date").cast(pl.Date) <= pl.col("ann_date")).alias("listed"),
    )
    # 存续：已上市，且样本内最后一条行情在公告日之后（样本末仍存续的券 last_ts 也在公告日之后）
    cand = cand.with_columns(
        (pl.col("listed") & (pl.col("last_ts").is_null() | (pl.col("last_ts").dt.date() >= pl.col("ann_date")))).alias("alive"))
    key = ["stock_code", "title", "ann_time"]
    by_name = cand.filter(pl.col("name_hit"))
    rest = cand.join(by_name.select(key).unique(), on=key, how="anti").filter(pl.col("alive"))
    single = rest.group_by(key).agg(pl.len().alias("k")).filter(pl.col("k") == 1).select(key)
    matched = pl.concat([by_name, rest.join(single, on=key)]).select(
        "bond_code", "bond_name", "stock_code", "kind", "title", "ann_time", "pre_open", "ts", "url")
    dropped = n.join(matched.select(key).unique(), on=key, how="anti")
    print("公告", n.height, "匹配到转债", matched.height, "未匹配", dropped.height)
    print(matched.group_by("kind").len().sort("kind"))

    matched = matched.with_columns(pl.col("ts").dt.year().alias("_y"))
    for (y,), g in matched.group_by(["_y"]):
        lake.write("cb_notices", g.drop("_y").sort("ts"), ts_col="ts", part=str(y))


if __name__ == "__main__":
    {"download": cmd_download, "ingest": cmd_ingest}[sys.argv[1]]()
