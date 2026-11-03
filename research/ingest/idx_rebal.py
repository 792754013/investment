"""中小盘指数定期调样名单（OPP-005 快速否证）。

来源（公开，无需登录）：
  中证  www.csindex.com.cn 公告接口：
        POST /csindex-home/announcement/queryAnnouncementByVo  按关键词列出公告
        GET  /csindex-home/announcement/queryAnnouncementById 公告正文 + 附件（PDF / xlsx）
        正文里有"于 YYYY 年 M 月 D 日收市后生效"；名单在附件或正文表格里。只取中证 500、中证 1000、中证 2000 三条。
  国证  akshare index_detail_hist_adjust_cni（cnindex 历史调样，399303 国证 2000），只有生效日，没有公告日。

输出 lake 数据集 idx_rebal：index、code、side（in / out）、ann_ts（公告日，国证为空）、eff_ts（生效日，
按"收市后生效"记为生效日当天 07:00 UTC，即该日收盘为最后一个旧样本交易日）。

用法：
  uv run python research/ingest/idx_rebal.py list       # 列出候选公告（检查用）
  uv run python research/ingest/idx_rebal.py download   # → data/raw/idx/
  uv run python research/ingest/idx_rebal.py ingest     # → lake: idx_rebal
  uv run python research/ingest/idx_rebal.py download-px  # 名单里出现过的个股 + 4 条指数的日线 → data/raw/idx/px/
  uv run python research/ingest/idx_rebal.py ingest-px    # → lake: eq_daily（后复权收盘、成交额）、idx_daily

个股日线：后复权。前 126 只来自 akshare stock_zh_a_hist（东方财富），其余来自 stock_zh_a_daily（新浪）；
每只只用一个来源，收益按比值算，复权基准不同不影响。取不到的列在输出里。
指数日线：中证三条用 stock_zh_index_hist_csindex，国证 2000 用 stock_zh_index_daily（新浪）。
"""

import io
import json
import re
import sys
import time
from datetime import date, timedelta

import akshare as ak
import httpx
import pandas as pd
import polars as pl

from lab import lake
from lab.paths import raw_dir

BASE = "https://www.csindex.com.cn/csindex-home/announcement"
HEADERS = {"User-Agent": "Mozilla/5.0 research", "Referer": "https://www.csindex.com.cn/", "Content-Type": "application/json"}
INDEXES = {"中证500": "000905", "中证1000": "000852", "中证2000": "932000"}
# 中证 500 / 1000 / 2000 的定期调整都和沪深 300 在同一份"关于…沪深300…"公告里发布（2022 年以前标题为"关于调整沪深300和中证香港100等指数样本的公告"）
TITLE = re.compile(r"^关于(调整)?沪深300(?!精明).*(指数样本|定期调整结果)")


def root():
    return raw_dir() / "idx"


def retry(f, n: int = 5):
    for k in range(n):
        try:
            return f()
        except httpx.HTTPError:
            if k == n - 1:
                raise
            time.sleep(5)


def search(c: httpx.Client, kw: str) -> list[dict]:
    out, p = [], 1
    while True:
        for k in range(5):
            try:
                d = c.post(f"{BASE}/queryAnnouncementByVo",
                           json=dict(page=dict(desc="", key="", page=p, rows=50, sortBy=""), searchInput=kw)).json()
                break
            except httpx.HTTPError:
                if k == 4:
                    raise
                time.sleep(5)
        out += d.get("data") or []
        if not d.get("data") or len(d["data"]) < 50:
            return out
        p += 1
        time.sleep(0.5)


def candidates(c: httpx.Client) -> list[dict]:
    seen, out = set(), []
    for kw in ["沪深300"]:
        for x in search(c, kw):
            if (x["id"] in seen or not TITLE.search(x["title"]) or x["publishDate"] < "2020-01-01"
                    or x["publishDate"][5:7] not in ("05", "06", "11", "12")):  # 其余是临时调整
                continue
            seen.add(x["id"])
            out.append(x)
    return sorted(out, key=lambda x: x["publishDate"])


def cmd_list() -> None:
    with httpx.Client(headers=HEADERS, timeout=30) as c:
        for x in candidates(c):
            print(x["publishDate"], x["id"], x["title"])


def cmd_download() -> None:
    (root() / "csi").mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=60) as c:
        for x in candidates(c):
            p = root() / "csi" / f"{x['id']}.json"
            if not p.exists():
                d = retry(lambda: c.get(f"{BASE}/queryAnnouncementById", params=dict(id=x["id"])).json()["data"])
                p.write_text(json.dumps(d, ensure_ascii=False))
                time.sleep(0.5)
            d = json.loads(p.read_text())
            for i, e in enumerate(d.get("enclosureList") or []):
                ext = e["fileName"].rsplit(".", 1)[-1].lower()
                q = root() / "csi" / f"{x['id']}_{i}.{ext}"
                if not q.exists():
                    q.write_bytes(retry(lambda: httpx.get(e["fileUrl"], headers={"User-Agent": "Mozilla/5.0"}, timeout=120).content))
                    time.sleep(0.5)
            print(x["publishDate"], x["id"], x["title"], flush=True)
    q = root() / "cni_399303.parquet"
    if not q.exists():
        ak.index_detail_hist_adjust_cni(symbol="399303").to_parquet(q)


def parse_lines(lines: list[str]) -> list[dict]:
    """按"××指数样本调整名单"分段；每行"调出代码 名称 调入代码 名称"，也可能只有一边。备选名单跳过。"""
    rows, cur = [], None
    for ln in lines:
        ln = ln.strip()
        m = re.match(r"^(\S+?)\s*指数样本(股)?调整名单", ln)
        if m:
            cur = m.group(1) if m.group(1) in INDEXES else None
            continue
        if re.match(r"^\S+?\s*指数\S*备选", ln):
            cur = None
            continue
        if cur is None:
            continue
        codes = re.findall(r"(?<!\d)(\d{6})(?!\d)", ln)
        if len(codes) == 2:
            rows += [dict(index=cur, code=codes[0], side="out"), dict(index=cur, code=codes[1], side="in")]
        elif len(codes) == 1:
            rows.append(dict(index=cur, code=codes[0], side="?"))
    return rows


def parse_one(d: dict, files: list) -> list[dict]:
    lines, rows = [], []
    for f in files:
        if f.suffix == ".pdf":
            import pdfplumber
            with pdfplumber.open(f) as pdf:
                for pg in pdf.pages:
                    lines += (pg.extract_text() or "").splitlines()
        elif f.suffix in (".xlsx", ".xls"):  # 2021-11 ～ 2022-11：调入 / 调出两张表，列为 指数代码 指数简称 证券代码 证券简称
            xs = pd.read_excel(f, sheet_name=None, dtype=str)
            code2idx = {v: k for k, v in INDEXES.items()}
            for sheet, side in (("调入", "in"), ("调出", "out")):
                df = xs[sheet]
                df = df[df["指数代码"].isin(code2idx)]
                rows += [dict(index=code2idx[a], code=b, side=side) for a, b in zip(df["指数代码"], df["证券代码"])]
    if rows:
        return rows
    if not lines:  # 名单在正文表格里
        html = d.get("content") or ""
        for tr in re.findall(r"<tr.*?</tr>|<p.*?</p>", html, flags=re.S):
            lines.append(" ".join(re.sub(r"<[^>]+>", " ", tr).split()))
    return parse_lines(lines)


def eff_date(text: str) -> date | None:
    m = re.search(r"[于自](\d{4})年(\d{1,2})月(\d{1,2})日(收市后|起)?.{0,4}生效", re.sub(r"<[^>]+>|\s", "", text))
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # 2020 年的写法："决定于 2020 年 6 月 15 日调整"，这是新样本第一天；生效日取前一天，下游向前对齐到交易日
    m = re.search(r"决定于(\d{4})年(\d{1,2})月(\d{1,2})日调整", re.sub(r"<[^>]+>|\s", "", text))
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) - timedelta(days=1) if m else None


def cmd_ingest() -> None:
    out = []
    for p in sorted((root() / "csi").glob("*.json")):
        d = json.loads(p.read_text())
        files = sorted((root() / "csi").glob(f"{p.stem}_*"))
        rows = parse_one(d, files)
        eff = eff_date(d.get("content") or "")
        cnt = pd.DataFrame(rows).groupby(["index", "side"]).size().to_dict() if rows else {}
        print(d["publishDate"], p.stem, eff, cnt, d["title"][:40])
        for r in rows:
            out.append(r | dict(ann=d["publishDate"], eff=eff))
    csi = pl.DataFrame(out).with_columns(pl.col("ann").str.to_date(), pl.col("eff").cast(pl.Date))
    bad = csi.filter(pl.col("side") == "?")
    if bad.height:
        print("无法判断调入还是调出的行（丢弃）：", bad.height)
    csi = csi.filter(pl.col("side") != "?").unique(["index", "code", "side", "eff"])

    c = pl.from_pandas(pd.read_parquet(root() / "cni_399303.parquet").astype(str))
    c = c.rename({"开始日期": "start", "样本代码": "code", "调整类型": "typ"}).with_columns(pl.col("start").str.to_date())
    # 国证 2000：开始日期是新样本的第一个交易日；生效日（最后一个旧样本交易日）取前一个交易日，下游按交易日对齐
    c = c.filter(pl.col("typ").is_in(["+", "-"])).select(
        pl.lit("国证2000").alias("index"), "code", pl.when(pl.col("typ") == "+").then(pl.lit("in")).otherwise(pl.lit("out")).alias("side"),
        pl.lit(None, dtype=pl.Date).alias("ann"), (pl.col("start") - pl.duration(days=1)).alias("eff"))
    df = pl.concat([csi.select("index", "code", "side", "ann", "eff"), c])

    def ts(col):
        return (pl.col(col).cast(pl.Datetime("us")) + pl.duration(hours=7)).dt.replace_time_zone("UTC")

    df = df.with_columns(ts("ann").alias("ann_ts"), ts("eff").alias("ts")).drop("ann", "eff").drop_nulls("ts")
    print(df.group_by("index", "side").len().sort("index", "side"))
    df = df.with_columns(pl.col("ts").dt.year().alias("_y"))
    for (y,), g in df.group_by(["_y"]):
        lake.write("idx_rebal", g.drop("_y").sort("index", "ts", "code"), ts_col="ts", part=str(y))


def cmd_download_px() -> None:
    d = root() / "px"
    d.mkdir(parents=True, exist_ok=True)
    codes = sorted(lake.scan("idx_rebal").select("code").unique().collect()["code"].to_list())
    end = date.today().strftime("%Y%m%d")
    failed = []
    for i, c in enumerate(codes):
        p = d / f"{c}.parquet"
        if p.exists():
            continue
        for k in range(3):
            try:
                # 东方财富 push2 在 2026-11-03 被限流，前 126 只来自它；其余改用新浪（后复权，有成交额），列名对齐成东方财富的
                pre = "sh" if c[0] == "6" else "bj" if c[0] in "489" else "sz"
                x = ak.stock_zh_a_daily(symbol=pre + c, start_date="20200101", end_date=end, adjust="hfq")
                x.rename(columns={"date": "日期", "close": "收盘", "amount": "成交额"}).assign(src="sina").to_parquet(p)
                break
            except Exception as e:
                if k == 2:
                    failed.append(c)
                    print("失败", c, repr(e)[:120], flush=True)
                time.sleep(60 * (k + 1))
        time.sleep(1.0)
        if i % 200 == 0:
            print(f"{i}/{len(codes)}", flush=True)
    for sym in ["000905", "000852", "932000"]:
        p = d / f"idx_{sym}.parquet"
        if not p.exists():
            ak.stock_zh_index_hist_csindex(symbol=sym, start_date="20200101", end_date=end).to_parquet(p)
    p = d / "idx_399303.parquet"
    if not p.exists():
        x = ak.stock_zh_index_daily(symbol="sz399303")  # 新浪；东方财富 index_zh_a_hist 被限流
        x = x[x["date"].astype(str) >= "2020-01-01"]
        x.rename(columns={"date": "日期", "close": "收盘"}).to_parquet(p)
    print("完成，失败", len(failed), failed[:30])


def cmd_ingest_px() -> None:
    def ts(col):
        return (pl.col(col).cast(pl.Date).cast(pl.Datetime("us")) + pl.duration(hours=7)).dt.replace_time_zone("UTC")

    frames = []
    for p in sorted((root() / "px").glob("[0-9]*.parquet")):
        x = pd.read_parquet(p)
        if x.empty:
            continue
        x = x.rename(columns={"日期": "date", "收盘": "close", "成交额": "amount"})[["date", "close", "amount"]]
        x["code"] = p.stem
        frames.append(pl.from_pandas(x))
    e = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.col("date").cast(pl.Utf8).str.to_date(), pl.col("close").cast(pl.Float64), pl.col("amount").cast(pl.Float64))
    e = e.with_columns(ts("date").alias("ts"), pl.col("date").dt.year().alias("_y")).drop("date")
    for (y,), g in e.group_by(["_y"]):
        lake.write("eq_daily", g.drop("_y").sort("code", "ts"), ts_col="ts", part=str(y))
    print("eq_daily", e.height, "行", e["code"].n_unique(), "只")

    rows = []
    names = {"000905": "中证500", "000852": "中证1000", "932000": "中证2000", "399303": "国证2000"}
    for sym, nm in names.items():
        x = pd.read_parquet(root() / "px" / f"idx_{sym}.parquet")
        x = x.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]
        x["index"] = nm
        rows.append(pl.from_pandas(x))
    i = pl.concat(rows, how="vertical_relaxed").with_columns(pl.col("date").cast(pl.Utf8).str.to_date(), pl.col("close").cast(pl.Float64))
    i = i.with_columns(ts("date").alias("ts")).drop("date")
    lake.write("idx_daily", i.sort("index", "ts"), ts_col="ts", part="all")
    print("idx_daily", i.height, "行")


if __name__ == "__main__":
    {"list": cmd_list, "download": cmd_download, "ingest": cmd_ingest,
     "download-px": cmd_download_px, "ingest-px": cmd_ingest_px}[sys.argv[1]]()
