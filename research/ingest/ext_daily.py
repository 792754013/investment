"""下载外部日度数据（情绪、新闻、链上、稳定币、宏观、Coinbase 价格）并入库，供加密日线研究使用。

全部是公开免费接口，不需要 API Key；遇到要 Key 或人机验证的源直接跳过。
统一长表：ts（数值所属那一天的 00:00 UTC）、series（字符串）、value（Float64）；每个来源一个数据集，part = series。
  ext_sentiment  alternative.me 恐惧贪婪指数（fng）；维基百科英文页面日浏览量（wiki_bitcoin 等，agent=user）
  ext_news       GDELT DOC 2.0 每日新闻量（gdelt_<q>_vol，及同期监测到的全部文章数 gdelt_<q>_total）和情绪（gdelt_<q>_tone）
  ext_onchain    blockchain.com charts（bc_hash_rate 等）；CoinMetrics community（cm_btc_CapMVRVCur 等）
  ext_stable     DefiLlama 稳定币总流通量（stable_total_usd）
  ext_macro      美股和宏观：首选 FRED，FRED 不通时改用 akshare（新浪、东方财富）和 CBOE 官网 CSV
  ext_coinbase   Coinbase Exchange 日 K 收盘价（cb_btc_close、cb_eth_close）
lake.write 会把 2026-05-01 之后的行自动放进封存段。

用法（可重复运行；已下载的文件跳过，单文件的全量快照也只在不存在时下载）：
  uv run python research/ingest/ext_daily.py download   # → data/raw/ext/
  uv run python research/ingest/ext_daily.py ingest     # → lake
各来源的下载结果（成功、跳过、失败原因）记在 data/raw/ext/status.json。
"""

import json
import sys
import time
from datetime import date, timedelta

import httpx
import polars as pl

from lab import lake
from lab.paths import raw_dir

UA = {"User-Agent": "investment-research/0.1"}
TODAY = date.today()

WIKI_ARTICLES = {"Bitcoin": "wiki_bitcoin", "Ethereum": "wiki_ethereum", "Cryptocurrency": "wiki_cryptocurrency"}
GDELT_QUERIES = {"bitcoin": "bitcoin", "cryptocurrency": "cryptocurrency"}
GDELT_FIRST = date(2017, 1, 1)  # DOC 2.0 全文检索的起点，实际能拿到多少以下载结果为准
BC_CHARTS = ["hash-rate", "n-transactions", "n-unique-addresses", "miners-revenue", "transaction-fees-usd", "difficulty"]
# CoinMetrics community 免费档（catalog-v2 里 community=true）里选的指标；NVT 类不在免费档
CM_METRICS = ["CapMVRVCur", "AdrActCnt", "TxCnt", "TxTfrCnt", "FeeTotNtv", "FlowInExNtv", "FlowOutExNtv",
              "FlowInExUSD", "FlowOutExUSD", "SplyExNtv", "CapMrktCurUSD", "PriceUSD", "HashRate", "SplyCur",
              "IssTotUSD", "volume_reported_spot_usd_1d"]
CM_ASSETS = ["btc", "eth"]
FRED = {"SP500": "sp500", "NASDAQCOM": "nasdaq", "VIXCLS": "vix", "DTWEXBGS": "usd_broad", "DGS10": "us10y"}
CB_PRODUCTS = {"BTC-USD": "cb_btc_close", "ETH-USD": "cb_eth_close"}


def root():
    return raw_dir() / "ext"


def status() -> dict:
    p = root() / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def set_status(key: str, msg: str) -> None:
    s = status()
    s[key] = msg
    root().mkdir(parents=True, exist_ok=True)
    (root() / "status.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    print(f"  [{key}] {msg}", flush=True)


def get(c: httpx.Client, url: str, params=None, tries: int = 5, base_wait: float = 2.0) -> httpx.Response:
    """带重试的 GET。429/5xx/网络错误重试；其余状态码原样返回。"""
    last = None
    for i in range(tries):
        try:
            r = c.get(url, params=params)
            if r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}"
                ra = r.headers.get("Retry-After", "")
                wait = max(base_wait * 2 ** i, float(ra) if ra.isdigit() else 0)
            else:
                return r
        except httpx.TransportError as e:
            last, wait = repr(e), base_wait * 2 ** i
        print(f"    重试 {i + 1}/{tries}（{last}），等 {wait:.0f} 秒：{url}", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{tries} 次重试失败（{last}）：{url}")


def save(path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def looks_blocked(r: httpx.Response) -> bool:
    t = r.text[:2000].lower()
    return r.headers.get("content-type", "").startswith("text/html") and any(
        k in t for k in ("captcha", "challenge", "cf-chl", "just a moment"))


# ---------- 下载 ----------

def dl_sentiment(c: httpx.Client) -> None:
    p = root() / "sentiment" / "fng.json"
    if not p.exists():
        r = get(c, "https://api.alternative.me/fng/", params={"limit": 0, "format": "json"})
        r.raise_for_status()
        save(p, r.content)
    set_status("fng", "ok")
    end = TODAY.strftime("%Y%m%d")
    for art, name in WIKI_ARTICLES.items():
        p = root() / "sentiment" / f"{name}.json"
        if not p.exists():
            r = get(c, f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
                       f"{art}/daily/20150701/{end}")
            r.raise_for_status()
            save(p, r.content)
            time.sleep(0.5)
        set_status(name, "ok（Wikimedia pageviews 最早只到 2015-07-01）")


def dl_news(c: httpx.Client) -> None:
    """GDELT：每次请求间隔 ≥ 6 秒，遇 429 退避。按年分块；某块持续失败就放弃整个来源。"""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    chunks = []
    d = GDELT_FIRST
    while d < TODAY:
        e = min(date(d.year + 1, 1, 1), TODAY)
        chunks.append((d, e))
        d = e
    fails = 0
    for q, name in GDELT_QUERIES.items():
        for mode in ("timelinevolraw", "timelinetone"):
            for s, e in chunks:
                p = root() / "news" / f"{name}_{mode}_{s:%Y}.json"
                if p.exists():
                    continue
                params = {"query": q, "mode": mode, "format": "json", "timelinesmooth": 0,
                          "startdatetime": f"{s:%Y%m%d}000000", "enddatetime": f"{e:%Y%m%d}000000"}
                time.sleep(7)
                try:
                    r = get(c, url, params=params, tries=5, base_wait=30)
                    r.raise_for_status()
                    json.loads(r.content)  # 不是 JSON（如限流提示文本）就算失败
                except Exception as ex:  # noqa: BLE001
                    fails += 1
                    print(f"    GDELT {q} {mode} {s:%Y} 失败：{ex!r}"[:300], flush=True)
                    if fails >= 3:
                        set_status("gdelt", f"放弃：连续失败 {fails} 次，最后一次 {ex!r}"[:300])
                        return
                    continue
                fails = 0
                save(p, r.content)
                print(f"    GDELT {q} {mode} {s:%Y} ok", flush=True)
    missing = [f"{n}_{m}_{s:%Y}" for n in GDELT_QUERIES.values() for m in ("timelinevolraw", "timelinetone")
               for s, _ in chunks if not (root() / "news" / f"{n}_{m}_{s:%Y}.json").exists()]
    set_status("gdelt", f"部分缺失（429 限流），缺 {len(missing)} 块：{missing}；重跑 download 可补" if missing else "ok")


def dl_onchain(c: httpx.Client) -> None:
    for name in BC_CHARTS:
        p = root() / "onchain" / f"bc_{name}.json"
        if not p.exists():
            r = get(c, f"https://api.blockchain.info/charts/{name}",
                    params={"timespan": "all", "format": "json", "sampled": "false"}, base_wait=5)
            r.raise_for_status()
            save(p, r.content)
            time.sleep(1)
        set_status(f"bc_{name}", "ok")
    # CoinMetrics：先看 catalog 确认免费档可用，再翻页下载
    cat = get(c, "https://community-api.coinmetrics.io/v4/catalog-v2/asset-metrics",
              params={"assets": ",".join(CM_ASSETS), "page_size": 10000}).json()
    save(root() / "onchain" / "cm_catalog.json", json.dumps(cat).encode())
    for a in cat["data"]:
        free = {m["metric"] for m in a["metrics"]
                if any(f["frequency"] == "1d" and f.get("community") for f in m["frequencies"])}
        metrics = [m for m in CM_METRICS if m in free]
        p = root() / "onchain" / f"cm_{a['asset']}.json"
        if not p.exists():
            rows, page_url = [], "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
            params = {"assets": a["asset"], "metrics": ",".join(metrics), "frequency": "1d",
                      "start_time": "2009-01-01", "page_size": 10000, "ignore_unsupported_errors": "true"}
            while page_url:
                r = get(c, page_url, params=params)
                r.raise_for_status()
                j = r.json()
                rows += j["data"]
                page_url, params = j.get("next_page_url"), None
                time.sleep(1)
            save(p, json.dumps(rows).encode())
        set_status(f"cm_{a['asset']}", f"ok，指标 {metrics}；未取到（不在免费档）{sorted(set(CM_METRICS) - free)}")


def dl_stable(c: httpx.Client) -> None:
    p = root() / "stable" / "stablecoincharts_all.json"
    if not p.exists():
        r = get(c, "https://stablecoins.llama.fi/stablecoincharts/all")
        r.raise_for_status()
        save(p, r.content)
    set_status("stable_total_usd", "ok")


def dl_macro(c: httpx.Client) -> None:
    fred_ok = True
    for fid in FRED:
        p = root() / "macro" / f"fred_{fid}.csv"
        if p.exists():
            continue
        if not fred_ok:
            break
        try:
            r = get(c, "https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": fid}, tries=2, base_wait=5)
            r.raise_for_status()
            if looks_blocked(r) or not r.text.startswith(("DATE", "observation_date")):
                raise RuntimeError(f"返回的不是 CSV：{r.text[:100]!r}")
            save(p, r.content)
        except Exception as ex:  # noqa: BLE001
            fred_ok = False
            set_status("fred", f"不通，改用 akshare/CBOE：{ex!r}"[:300])
    if fred_ok:
        set_status("fred", "ok")
    have = {fid for fid in FRED if (root() / "macro" / f"fred_{fid}.csv").exists()}

    import akshare as ak  # 只在需要时导入（慢）

    fallbacks = {  # FRED 缺的序列 → akshare / CBOE；黄金 FRED 没有免费序列，始终走 akshare
        "SP500": ("sina_spx", lambda: ak.index_us_stock_sina(symbol=".INX")),
        "NASDAQCOM": ("sina_ixic", lambda: ak.index_us_stock_sina(symbol=".IXIC")),
        "DTWEXBGS": ("em_dxy", lambda: ak.index_global_hist_em(symbol="美元指数")),
        "DGS10": ("em_us_rate", lambda: ak.bond_zh_us_rate(start_date="19900101")),
        "GOLD": ("sina_xau", lambda: ak.futures_foreign_hist(symbol="XAU")),
    }
    for fid, (fname, fn) in fallbacks.items():
        if fid in have:
            continue
        p = root() / "macro" / f"{fname}.csv"
        if not p.exists():
            try:
                df = fn()
                save(p, df.to_csv(index=False).encode())
                time.sleep(1)
            except Exception as ex:  # noqa: BLE001
                set_status(fname, f"失败：{ex!r}"[:300])
                continue
        set_status(fname, "ok")
    if "VIXCLS" not in have:
        p = root() / "macro" / "cboe_vix.csv"
        if not p.exists():
            r = get(c, "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv")
            r.raise_for_status()
            save(p, r.content)
        set_status("cboe_vix", "ok")


def dl_coinbase(c: httpx.Client) -> None:
    for prod in CB_PRODUCTS:
        d = date(2015, 1, 1)
        while d < TODAY:
            e = min(d + timedelta(days=299), TODAY)
            p = root() / "coinbase" / prod / f"{d:%Y-%m-%d}.json"
            # 最后一段（含今天）每次重下；之前的段已下载就跳过
            if not p.exists() or e >= TODAY - timedelta(days=1):
                r = get(c, f"https://api.exchange.coinbase.com/products/{prod}/candles",
                        params={"granularity": 86400, "start": f"{d:%Y-%m-%d}T00:00:00Z",
                                "end": f"{e:%Y-%m-%d}T00:00:00Z"})
                r.raise_for_status()
                save(p, r.content)
                time.sleep(0.5)
            d = e + timedelta(days=1)
        set_status(CB_PRODUCTS[prod], "ok")


def cmd_download() -> None:
    with httpx.Client(timeout=120, follow_redirects=True, headers=UA) as c:
        for name, fn in [("sentiment", dl_sentiment), ("onchain", dl_onchain), ("stable", dl_stable),
                         ("coinbase", dl_coinbase), ("macro", dl_macro), ("news", dl_news)]:
            print(f"== {name}", flush=True)
            try:
                fn(c)
            except Exception as ex:  # noqa: BLE001
                set_status(name, f"失败：{ex!r}"[:300])


# ---------- 入库 ----------

def day_ts(expr: pl.Expr) -> pl.Expr:
    """日期 → 当天 00:00 UTC。"""
    return expr.cast(pl.Date).cast(pl.Datetime("us")).dt.replace_time_zone("UTC")


def epoch_day(expr: pl.Expr) -> pl.Expr:
    """unix 秒 → 所属 UTC 日的 00:00。"""
    return day_ts(pl.from_epoch(expr.cast(pl.Int64), time_unit="s").dt.date())


def long(df: pl.DataFrame, series: str) -> pl.DataFrame:
    """df 含 ts、value；同一天多个值取最后一个。"""
    return (df.select("ts", pl.col("value").cast(pl.Float64)).drop_nulls().sort("ts")
            .unique("ts", keep="last").sort("ts").with_columns(pl.lit(series).alias("series"))
            .select("ts", "series", "value"))


def write(dataset: str, df: pl.DataFrame) -> None:
    s = df["series"][0]
    lake.write(dataset, df, ts_col="ts", part=s)
    print(f"  {dataset}/{s}: {df.height} 行，{df['ts'].min():%Y-%m-%d}～{df['ts'].max():%Y-%m-%d}", flush=True)


def ing_sentiment() -> None:
    p = root() / "sentiment" / "fng.json"
    if p.exists():
        d = pl.DataFrame(json.loads(p.read_text())["data"])
        write("ext_sentiment", long(d.select(epoch_day(pl.col("timestamp")).alias("ts"),
                                             pl.col("value").cast(pl.Float64)), "fng"))
    for name in WIKI_ARTICLES.values():
        p = root() / "sentiment" / f"{name}.json"
        if p.exists():
            d = pl.DataFrame(json.loads(p.read_text())["items"])
            write("ext_sentiment", long(d.select(
                day_ts(pl.col("timestamp").str.slice(0, 8).str.strptime(pl.Date, "%Y%m%d")).alias("ts"),
                pl.col("views").alias("value")), name))


def ing_news() -> None:
    for name in GDELT_QUERIES.values():
        for mode, suffix in (("timelinevolraw", "vol"), ("timelinetone", "tone")):
            # 分块的终点日会同时出现在相邻两块里（前一块里可能不完整），按日期去重，后一块覆盖前一块
            vals, norms = {}, {}
            for p in sorted((root() / "news").glob(f"{name}_{mode}_*.json")):
                for tl in json.loads(p.read_text()).get("timeline", []):
                    for x in tl["data"]:
                        vals[x["date"]] = x["value"]
                        if "norm" in x:
                            norms[x["date"]] = x["norm"]
            if not vals:
                continue
            rows, totals = list(vals.items()), list(norms.items())

            def frame(rs):
                d = pl.DataFrame(rs, schema=["date", "value"], orient="row")
                return d.select(day_ts(pl.col("date").str.strptime(pl.Datetime, "%Y%m%dT%H%M%SZ").dt.date()).alias("ts"),
                                pl.col("value").cast(pl.Float64))

            d = frame(rows)
            # 正常应是日分辨率；万一返回日内分辨率，量求和、情绪取均值
            agg = pl.col("value").sum() if suffix == "vol" else pl.col("value").mean()
            write("ext_news", long(d.group_by("ts").agg(agg), f"gdelt_{name}_{suffix}"))
            if totals:
                write("ext_news", long(frame(totals).group_by("ts").agg(pl.col("value").sum()), f"gdelt_{name}_total"))


def ing_onchain() -> None:
    for name in BC_CHARTS:
        p = root() / "onchain" / f"bc_{name}.json"
        if p.exists():
            d = pl.DataFrame(json.loads(p.read_text())["values"])
            write("ext_onchain", long(d.select(epoch_day(pl.col("x")).alias("ts"), pl.col("y").alias("value")),
                                      "bc_" + name.replace("-", "_")))
    for a in CM_ASSETS:
        p = root() / "onchain" / f"cm_{a}.json"
        if not p.exists():
            continue
        d = pl.DataFrame(json.loads(p.read_text()), infer_schema_length=None)
        ts = day_ts(pl.col("time").str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d")).alias("ts")
        for m in CM_METRICS:
            if m in d.columns:
                write("ext_onchain", long(d.select(ts, pl.col(m).cast(pl.Float64, strict=False).alias("value")),
                                          f"cm_{a}_{m}"))


def ing_stable() -> None:
    p = root() / "stable" / "stablecoincharts_all.json"
    if p.exists():
        rows = [(int(x["date"]), sum((x.get("totalCirculatingUSD") or {}).values())) for x in json.loads(p.read_text())]
        d = pl.DataFrame(rows, schema=["x", "value"], orient="row")
        write("ext_stable", long(d.select(epoch_day(pl.col("x")).alias("ts"), "value"), "stable_total_usd"))


def ing_macro() -> None:
    m = root() / "macro"

    def csv(name):
        return pl.read_csv(m / name, infer_schema_length=None) if (m / name).exists() else None

    for fid, series in FRED.items():
        d = csv(f"fred_{fid}.csv")
        if d is not None:
            dc, vc = d.columns[0], d.columns[1]
            write("ext_macro", long(d.select(day_ts(pl.col(dc).str.strptime(pl.Date, "%Y-%m-%d")).alias("ts"),
                                             pl.col(vc).cast(pl.Float64, strict=False).alias("value")), series))
    for fname, series, dcol, vcol in [("sina_spx.csv", "sp500", "date", "close"),
                                      ("sina_ixic.csv", "nasdaq", "date", "close"),
                                      ("em_dxy.csv", "dxy", "日期", "最新价"),
                                      ("em_us_rate.csv", "us10y", "日期", "美国国债收益率10年"),
                                      ("sina_xau.csv", "gold", "date", "close")]:
        d = csv(fname)
        if d is not None:
            write("ext_macro", long(d.select(day_ts(pl.col(dcol).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d")).alias("ts"),
                                             pl.col(vcol).cast(pl.Float64, strict=False).alias("value")), series))
    d = csv("cboe_vix.csv")
    if d is not None:
        write("ext_macro", long(d.select(day_ts(pl.col("DATE").str.strptime(pl.Date, "%m/%d/%Y")).alias("ts"),
                                         pl.col("CLOSE").alias("value")), "vix"))


def ing_coinbase() -> None:
    for prod, series in CB_PRODUCTS.items():
        rows = []
        for p in sorted((root() / "coinbase" / prod).glob("*.json")):
            rows += json.loads(p.read_text())  # [time, low, high, open, close, volume]
        if rows:
            d = pl.DataFrame([(r[0], r[4]) for r in rows], schema=["x", "value"], orient="row")
            write("ext_coinbase", long(d.select(epoch_day(pl.col("x")).alias("ts"), "value"), series))


def cmd_ingest() -> None:
    for fn in (ing_sentiment, ing_news, ing_onchain, ing_stable, ing_macro, ing_coinbase):
        fn()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        cmd_download()
    elif cmd == "ingest":
        cmd_ingest()
    else:
        sys.exit(__doc__)
