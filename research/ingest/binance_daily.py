"""下载 Binance 现货日线和 U 本位永续资金费的公开归档并入库（OPP-007，docs/research/OPP007-trend-plan.md）。

来源 data.binance.vision，不需要 API Key。只下载到 2026-04；2026-05 之后的封存段本轮不需要，也不下载。
  - 现货日线 → lake: bn_spot_1d（inst_id = BTCUSDT 等）
  - 资金费   → lake: bn_funding

用法（可重复运行，已下载的文件跳过）：
  uv run python research/ingest/binance_daily.py download
  uv run python research/ingest/binance_daily.py ingest

全币种面板（含已下架币，避免幸存者偏差）：
  uv run python research/ingest/binance_daily.py panel-download
  uv run python research/ingest/binance_daily.py panel-ingest
  用 S3 列表接口列出 data/spot/monthly/klines/ 下全部交易对，保留 USDT 计价、剔除杠杆代币、
  稳定币/法币、代币化股票（见 panel_symbols），再逐个交易对列出已有的月度文件，只下载列表里存在的
  2017-08～2026-04 文件（不盲猜 URL）。U 本位永续资金费同理；没有永续的交易对记为 missing。
  写入 bn_spot_1d / bn_funding，part = 交易对。
"""

import asyncio
import io
import json
import re
import sys
import zipfile
from datetime import date

import httpx
import polars as pl

from lab import lake
from lab.paths import raw_dir

BASE = "https://data.binance.vision/data"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "EOSUSDT", "XLMUSDT", "NEOUSDT", "TRXUSDT", "BNBUSDT"]
FIRST, LAST = date(2017, 8, 1), date(2026, 4, 1)
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
         "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def root():
    return raw_dir() / "binance_daily"


def months() -> list[str]:
    out, d = [], FIRST
    while d <= LAST:
        out.append(f"{d.year}-{d.month:02d}")
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def cmd_download() -> None:
    with httpx.Client(timeout=120, follow_redirects=True) as c:
        for s in SYMBOLS:
            for ym in months():
                for url, rel in ((f"{BASE}/spot/monthly/klines/{s}/1d/{s}-1d-{ym}.zip", f"klines/{s}/{ym}.zip"),
                                 (f"{BASE}/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{ym}.zip", f"funding/{s}/{ym}.zip")):
                    p = root() / rel
                    if p.exists() or p.with_name(p.name + ".missing").exists():
                        continue
                    p.parent.mkdir(parents=True, exist_ok=True)
                    r = c.get(url)
                    if r.status_code == 404:
                        p.with_name(p.name + ".missing").touch()
                        continue
                    r.raise_for_status()
                    p.write_bytes(r.content)
            print(f"  {s} 下载完成", flush=True)


def read_csv(path, names=None) -> pl.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(next(n for n in z.namelist() if n.endswith(".csv")))
    has_header = not raw[:1].isdigit()
    return pl.read_csv(io.BytesIO(raw), has_header=has_header, new_columns=None if has_header else names,
                       infer_schema_length=10_000)


def to_ts(col: str) -> pl.Expr:
    # 2025 年起现货归档的时间戳改为微秒
    x = pl.col(col).cast(pl.Int64)
    return pl.when(x > 10 ** 14).then(x // 1000).otherwise(x).pipe(pl.from_epoch, time_unit="ms").dt.replace_time_zone("UTC")


def cmd_ingest() -> None:
    for s in SYMBOLS:
        files = sorted((root() / "klines" / s).glob("*.zip"))
        if files:
            k = pl.concat([read_csv(p, KCOLS).select(
                to_ts("open_time").alias("ts"),
                *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume", "quote_volume")),
            ) for p in files]).unique("ts").sort("ts").with_columns(pl.lit(s).alias("inst_id"))
            lake.write("bn_spot_1d", k, ts_col="ts", part=s)
            print(f"  bn_spot_1d {s}: {k.height} 天，{k['ts'].min():%Y-%m-%d}～{k['ts'].max():%Y-%m-%d}", flush=True)
        files = sorted((root() / "funding" / s).glob("*.zip"))
        if files:
            f = pl.concat([read_csv(p, ["calc_time", "funding_interval_hours", "last_funding_rate"]).select(
                to_ts("calc_time").alias("ts"), pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
            ) for p in files]).unique("ts").sort("ts").with_columns(pl.lit(s).alias("inst_id"))
            lake.write("bn_funding", f, ts_col="ts", part=s)
            print(f"  bn_funding {s}: {f.height} 条，{f['ts'].min():%Y-%m-%d} 起", flush=True)


# ---------- 全币种面板 ----------

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
UA = {"User-Agent": "investment-research/0.1"}
CONCURRENCY = 16
# 稳定币、法币（核查过 S3 上全部 *USDT 基础币）。UST 是 Terra 脱锚前的稳定币；USTC 是脱锚后改名的投机代币，保留。
STABLE_FIAT = {
    "USDC", "BUSD", "TUSD", "PAX", "USDP", "FDUSD", "DAI", "USDS", "USDSB", "USDSOLD", "EUR", "GBP", "AUD", "TRY",
    "BRL", "RUB", "BIDR", "IDRT", "UAH", "NGN", "ZAR", "AEUR", "EURI", "USDE", "PYUSD", "XUSD", "BKRW", "BFUSD",
    "RLUSD", "USD1", "SUSD", "UST", "VAI", "JPY", "PLN", "ARS", "MXN", "COP", "CZK", "GUSD", "RWUSD", "U",
}
LEVERAGED_SUFFIX = ("UP", "DOWN", "BULL", "BEAR")


def panel_root():
    return root() / "panel"


# 2026 年上线的代币化美股/ETF，不是加密资产，剔除（核查过名单）
TOKENIZED_STOCKS = set("""
AAOIB AAPLB ADBEB AGPUB ALABB AMATB AMCB AMDB AMZNB APPB ARMB ASMLB ASTSB AVGOB AXTIB BABAB BMNRB BNCB BOTB BRKBB
BSPB CBRSB COHRB COINB CRCLB CRDOB CRMB CRWDB CRWVB CYPHB DELLB DJTB DRAMB EWYB FLNCB FWDIB GEVB GLWB GMEB GOOGLB
GPROB GSB HIMSB HOODB HPEB IBMB INTCB INTWB IRENB JPMB KORUB KSTRB LITEB LLYB MDBB METAB MRNAB MRVLB MSFTB MSTRB
MUB MUUB MVLLB NBISB NFLXB NOKB NVDAB ORCLB PDDB PENGB PLTRB PYPLB QCOMB QNTB QQQB RDDTB RKLBB SECZB SHAZB SKDDB
SKHYB SKUUB SMCIB SMHB SNDKB SNXXB SOXLB SOXSB SPCXB SPYB SQQQB STXB TMFB TQQQB TSLAB TSMB USARB USDEB UVXYB VSTB
WDCB WENB
""".split())


def is_leveraged(base: str, bases: set[str]) -> bool:
    # 杠杆代币：BTCUP/BTCDOWN/BNBBULL/BULL/BEAR 等。要求去掉后缀后的部分也是已上架基础币（或为空），
    # 避免误伤 JUP、SYRUP 这类正常币
    for suf in LEVERAGED_SUFFIX:
        if base.endswith(suf):
            stem = base[: -len(suf)]
            if stem == "" or stem in bases:
                return True
    return False


async def _get(c: httpx.AsyncClient, url: str, params=None) -> httpx.Response:
    for i in range(6):
        try:
            r = await c.get(url, params=params)
            if r.status_code in (429, 418) or r.status_code >= 500:
                raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
            return r
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            wait = 2 ** i
            if isinstance(e, httpx.HTTPStatusError) and e.response.headers.get("Retry-After", "").isdigit():
                wait = max(wait, int(e.response.headers["Retry-After"]))
            print(f"    重试 {i + 1}/6 {url}: {e!r}", flush=True)
            await asyncio.sleep(wait)
    raise RuntimeError(f"多次重试失败：{url}")


async def _s3_list(c: httpx.AsyncClient, prefix: str, kind: str) -> list[str]:
    """kind = "dirs"（CommonPrefixes）或 "keys"（Contents/Key），自动翻页。"""
    out, marker = [], ""
    while True:
        r = await _get(c, S3, params={"delimiter": "/", "prefix": prefix, "marker": marker})
        r.raise_for_status()
        x = r.text
        if kind == "dirs":
            out += re.findall(r"<Prefix>(" + re.escape(prefix) + r"[^<]+)</Prefix>", x)
        else:
            out += re.findall(r"<Key>([^<]+)</Key>", x)
        if "<IsTruncated>true</IsTruncated>" not in x:
            return out
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", x)
        marker = m.group(1) if m else (out[-1] if out else "")


def panel_symbols() -> list[str]:
    return json.loads((panel_root() / "symbols.json").read_text())["kept"]


async def _panel_download() -> None:
    panel_root().mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    ym_ok = set(months())
    async with httpx.AsyncClient(timeout=120, follow_redirects=True, headers=UA,
                                 limits=httpx.Limits(max_connections=CONCURRENCY)) as c:
        # 1. 全部交易对
        dirs = await _s3_list(c, "data/spot/monthly/klines/", "dirs")
        pairs = sorted(d.rstrip("/").rsplit("/", 1)[1] for d in dirs)
        usdt = [p for p in pairs if p.endswith("USDT")]
        bases = {p[:-4] for p in usdt}
        kept, dropped = [], {"leveraged": [], "stable_fiat": [], "tokenized_stock": []}
        for p in usdt:
            b = p[:-4]
            if b in STABLE_FIAT:
                dropped["stable_fiat"].append(p)
            elif is_leveraged(b, bases):
                dropped["leveraged"].append(p)
            elif b in TOKENIZED_STOCKS:
                dropped["tokenized_stock"].append(p)
            else:
                kept.append(p)
        (panel_root() / "symbols.json").write_text(json.dumps(
            {"all_pairs": len(pairs), "usdt_pairs": len(usdt), "kept": kept, "dropped": dropped}, indent=1))
        print(f"  交易对 {len(pairs)} 个，USDT {len(usdt)} 个，保留 {len(kept)} 个；剔除 "
              + "，".join(f"{k} {len(v)}" for k, v in dropped.items()), flush=True)

        # 2. 逐个交易对列出月度文件（结果缓存，重跑不再列）
        async def listing(kind: str, s: str) -> list[str]:
            cache = panel_root() / "listing" / kind / f"{s}.json"
            if cache.exists():
                return json.loads(cache.read_text())
            prefix = (f"data/spot/monthly/klines/{s}/1d/" if kind == "klines"
                      else f"data/futures/um/monthly/fundingRate/{s}/")
            async with sem:
                keys = [k for k in await _s3_list(c, prefix, "keys") if k.endswith(".zip")]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(keys))
            return keys

        lists = await asyncio.gather(*(listing(k, s) for s in kept for k in ("klines", "funding")))
        jobs = []
        for i, s in enumerate(kept):
            for kind, keys in (("klines", lists[2 * i]), ("funding", lists[2 * i + 1])):
                if kind == "funding" and not keys:
                    (root() / "funding" / s).mkdir(parents=True, exist_ok=True)
                    (root() / "funding" / s / "ALL.missing").touch()   # 没有 U 本位永续
                for k in keys:
                    ym = re.search(r"(\d{4}-\d{2})\.zip$", k).group(1)
                    if ym in ym_ok:
                        jobs.append((f"https://data.binance.vision/{k}", root() / kind / s / f"{ym}.zip"))
        todo = [(u, p) for u, p in jobs if not p.exists() and not p.with_name(p.name + ".missing").exists()]
        print(f"  月度文件 {len(jobs)} 个，待下载 {len(todo)} 个", flush=True)

        done = 0

        async def fetch(url, path):
            nonlocal done
            async with sem:
                r = await _get(c, url)
                await asyncio.sleep(0.05)
            path.parent.mkdir(parents=True, exist_ok=True)
            if r.status_code == 404:
                path.with_name(path.name + ".missing").touch()
            else:
                r.raise_for_status()
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(path)
            done += 1
            if done % 1000 == 0:
                print(f"    已下载 {done}/{len(todo)}", flush=True)

        await asyncio.gather(*(fetch(u, p) for u, p in todo))
    print("  面板下载完成", flush=True)


def cmd_panel_download() -> None:
    asyncio.run(_panel_download())


def cmd_panel_ingest() -> None:
    n_spot = n_fund = 0
    for s in panel_symbols():
        files = sorted((root() / "klines" / s).glob("*.zip"))
        if files:
            k = pl.concat([read_csv(p, KCOLS).select(
                to_ts("open_time").alias("ts"),
                *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close", "volume", "quote_volume")),
            ) for p in files]).unique("ts").sort("ts").with_columns(pl.lit(s).alias("inst_id"))
            lake.write("bn_spot_1d", k, ts_col="ts", part=s)
            n_spot += 1
        files = sorted((root() / "funding" / s).glob("*.zip"))
        if files:
            f = pl.concat([read_csv(p, ["calc_time", "funding_interval_hours", "last_funding_rate"]).select(
                to_ts("calc_time").alias("ts"), pl.col("last_funding_rate").cast(pl.Float64).alias("funding_rate"),
            ) for p in files]).unique("ts").sort("ts").with_columns(pl.lit(s).alias("inst_id"))
            lake.write("bn_funding", f, ts_col="ts", part=s)
            n_fund += 1
    print(f"  bn_spot_1d 写入 {n_spot} 个交易对，bn_funding 写入 {n_fund} 个", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        cmd_download()
    elif cmd == "ingest":
        cmd_ingest()
    elif cmd == "panel-download":
        cmd_panel_download()
    elif cmd == "panel-ingest":
        cmd_panel_ingest()
    else:
        sys.exit(__doc__)
