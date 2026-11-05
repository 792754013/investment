"""下载 OKX 历史数据并入库（P1 第 1 步，plan.md 19 节）。

数据来源都是公开的，不需要 API Key：
  - 资金费：public/market-data-history module=3 的月度归档（最早到 2022-01）
  - 1 分钟 K 线：module=2 的月度归档，永续和现货都有；入库时聚合为 1 小时

为了尽量减少幸存者偏差，候选合约族取三处的并集：OKX 当前合约、Bybit 当前合约、Binance 有过的全部 USDT 永续。
然后逐个用 OKX 归档接口确认，哪些合约族在哪些月份确实存在。
仍可能漏掉只在 OKX 上过、又已经下架的合约，数据说明里会写明这一点。

用法（可重复运行，已下载的文件会跳过）：
  uv run python research/ingest/okx_history.py manifest   # 确认合约族与月份 → data/raw/okx/manifest.parquet
  uv run python research/ingest/okx_history.py download   # 下载归档 zip → data/raw/okx/
  uv run python research/ingest/okx_history.py ingest     # 解压聚合 → lake: okx_funding / okx_swap_1h / okx_spot_1h
  uv run python research/ingest/okx_history.py ingest1m   # 永续 1 分钟原样入库 → lake: okx_swap_1m（OPP-006 插针）
"""

import asyncio
import io
import re
import sys
import time
import zipfile
from datetime import UTC, datetime

import httpx
import polars as pl

from lab import lake
from lab.paths import data_root, raw_dir

API = "https://www.okx.com/api/v5"
CDN = "https://static.okx.com/cdn/okex/traderecords"
FIRST_MONTH = (2022, 1)
HEADERS = {"User-Agent": "Mozilla/5.0 research"}
_SCALE = re.compile(r"^(1000000|100000|10000|1000|100)(?=[A-Z])")


def okx_raw():
    return raw_dir() / "okx"


def months() -> list[tuple[int, int]]:
    """FIRST_MONTH 到上个月（当月归档还没有生成）。"""
    now = datetime.now(UTC)
    y, m, out = *FIRST_MONTH, []
    while (y, m) < (now.year, now.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def norm(base: str) -> str:
    return _SCALE.sub("", base.upper())


# ---------- manifest ----------

def candidate_families(c: httpx.Client) -> list[str]:
    bases = set()
    for x in c.get(f"{API}/public/instruments", params={"instType": "SWAP"}).json()["data"]:
        if x["settleCcy"] == "USDT":
            bases.add(norm(x["ctValCcy"]))
    cursor = ""
    while True:
        d = c.get("https://api.bybit.com/v5/market/instruments-info",
                  params={"category": "linear", "limit": 1000, "cursor": cursor}).json()["result"]
        bases |= {norm(x["baseCoin"]) for x in d["list"] if x["quoteCoin"] == "USDT"}
        cursor = d.get("nextPageCursor")
        if not cursor:
            break
    s3 = c.get("https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
               params={"delimiter": "/", "prefix": "data/futures/um/monthly/fundingRate/"}).text
    for sym in re.findall(r"fundingRate/([A-Z0-9]+)USDT/</Prefix>", s3):
        bases.add(norm(sym))
    return sorted(f"{b}-USDT" for b in bases if re.fullmatch(r"[A-Z0-9]+", b))


def archive_listing(c: httpx.Client, families: list[str]) -> pl.DataFrame:
    """module=3 每次最多 5 个合约族、最长 10 个月。"""
    ms = months()
    windows = [ms[i:i + 10] for i in range(0, len(ms), 10)]
    rows = []
    batches = [families[i:i + 5] for i in range(0, len(families), 5)]
    total = len(batches) * len(windows)
    n = 0
    for w in windows:
        begin = int(datetime(*w[0], 1, tzinfo=UTC).timestamp() * 1000)
        end = int(datetime(*w[-1], 1, tzinfo=UTC).timestamp() * 1000)
        for batch in batches:
            n += 1
            params = dict(module=3, instType="SWAP", dateAggrType="monthly",
                          begin=begin, end=end, instFamilyList=",".join(batch))
            for attempt in range(6):
                d = c.get(f"{API}/public/market-data-history", params=params).json()
                if d["code"] == "0":
                    break
                time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"归档查询失败：{batch} {w[0]}～{w[-1]}：{d}")
            for x in d["data"]:
                for det in x["details"]:
                    for g in det["groupDetails"]:
                        m = re.search(r"(\d{4})-(\d{2})\.zip", g["filename"])
                        rows.append(dict(family=det["instFamily"], year=int(m[1]), month=int(m[2])))
            time.sleep(0.6)
            if n % 50 == 0:
                print(f"  manifest {n}/{total}", flush=True)
    return pl.DataFrame(rows, schema={"family": pl.String, "year": pl.Int32, "month": pl.Int32}).unique()


def cmd_manifest() -> None:
    with httpx.Client(timeout=30, headers=HEADERS) as c:
        fams = candidate_families(c)
        print(f"候选合约族 {len(fams)} 个，查询 OKX 归档…", flush=True)
        df = archive_listing(c, fams)
    out = okx_raw() / "manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.sort("family", "year", "month").write_parquet(out)
    print(f"OKX 归档中存在的合约族 {df['family'].n_unique()} 个，合约族×月份 {df.height} 条 → {out}")


# ---------- download ----------

def file_jobs(manifest: pl.DataFrame) -> list[tuple[str, str]]:
    jobs = []
    for fam, y, m in manifest.iter_rows():
        ym, ymd = f"{y}{m:02d}", f"{y}-{m:02d}"
        jobs.append((f"{CDN}/swaprates/monthly/{ym}/{fam}-SWAP-fundingrates-{ymd}.zip",
                     f"funding/{ym}/{fam}-SWAP.zip"))
        jobs.append((f"{CDN}/candlesticks/monthly/{ym}/{fam}-SWAP-candlesticks-{ymd}.zip",
                     f"swap_1m/{ym}/{fam}-SWAP.zip"))
        # 现货只取永续存续期间；不存在时 CDN 返回 404，记录为缺失
        jobs.append((f"{CDN}/candlesticks/monthly/{ym}/{fam}-candlesticks-{ymd}.zip",
                     f"spot_1m/{ym}/{fam}.zip"))
    return jobs


async def cmd_download_async() -> None:
    manifest = pl.read_parquet(okx_raw() / "manifest.parquet")
    jobs = [j for j in file_jobs(manifest) if not (okx_raw() / j[1]).exists()
            and not (okx_raw() / (j[1] + ".missing")).exists()]
    print(f"待下载 {len(jobs)} 个文件", flush=True)
    sem = asyncio.Semaphore(12)
    done = 0

    async def fetch(c: httpx.AsyncClient, url: str, rel: str) -> None:
        nonlocal done
        path = okx_raw() / rel
        async with sem:
            for attempt in range(5):
                try:
                    r = await c.get(url, params={"v": "1001"})
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(2 ** attempt)
            else:
                print(f"  失败 {url}", flush=True)
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        if r.status_code == 404:
            path.with_name(path.name + ".missing").touch()
        elif r.status_code == 200:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(path)
        else:
            print(f"  HTTP {r.status_code} {url}", flush=True)
        done += 1
        if done % 2000 == 0:
            print(f"  已完成 {done}/{len(jobs)}", flush=True)

    async with httpx.AsyncClient(timeout=120, headers=HEADERS) as c:
        await asyncio.gather(*(fetch(c, u, r) for u, r in jobs))
    print("下载结束", flush=True)


# ---------- ingest ----------

def read_zip_csv(path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        return pl.read_csv(io.BytesIO(z.read(name)), infer_schema_length=0, null_values=["None", ""])


def ms_to_utc(col: str) -> pl.Expr:
    return pl.from_epoch(pl.col(col).cast(pl.Int64), time_unit="ms").dt.replace_time_zone("UTC")


def quote_volume(kind: str, ct_val: float | None) -> pl.Expr:
    """归档里成交量列的含义随时间变化：
    早期只有 vol（永续是张数，现货是币数），vol_ccy / vol_quote 为空；
    后期 vol_quote 是计价币成交额；永续的 vol_ccy 是币数，现货的 vol_ccy 是计价币。
    """
    vol, vol_ccy, vol_quote, close = (pl.col(c).cast(pl.Float64) for c in ("vol", "vol_ccy", "vol_quote", "close"))
    if kind == "spot":
        return pl.coalesce(vol_quote, vol_ccy, vol * close)
    fallback = vol * ct_val * close if ct_val else pl.lit(None, dtype=pl.Float64)
    return pl.coalesce(vol_quote, vol_ccy * close, fallback)


def to_1h(df: pl.DataFrame, inst_id: str, kind: str, ct_val: float | None) -> pl.DataFrame:
    return (
        df.select(
            ms_to_utc("open_time").alias("ts"),
            *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close")),
            quote_volume(kind, ct_val).alias("vol_quote"),
        )
        .unique("ts")
        .sort("ts")
        .group_by_dynamic("ts", every="1h")
        .agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
            pl.col("close").last(),
            pl.when(pl.col("vol_quote").is_not_null().any()).then(pl.col("vol_quote").sum()).alias("vol_quote"),  # 无法换算时为空，不当作 0
            pl.len().alias("n_1m"),  # 不足 60 表示该小时有缺口
        )
        .with_columns(pl.lit(inst_id).alias("inst_id"))
    )


def current_ct_val() -> dict[str, float]:
    """当前在线永续的合约面值；已下架合约查不到，早期成交额只能留空。"""
    d = httpx.get(f"{API}/public/instruments", params={"instType": "SWAP"}, headers=HEADERS, timeout=30).json()
    return {x["instId"]: float(x["ctVal"]) for x in d["data"] if x["settleCcy"] == "USDT"}


def cmd_ingest() -> None:
    root = okx_raw()
    ct_val = current_ct_val()
    for kind, dataset in (("funding", "okx_funding"), ("swap_1m", "okx_swap_1h"), ("spot_1m", "okx_spot_1h")):
        by_inst: dict[str, list] = {}
        for p in sorted((root / kind).glob("*/*.zip")):
            by_inst.setdefault(p.stem, []).append(p)
        print(f"{dataset}: {len(by_inst)} 个合约", flush=True)
        for inst_id, paths in by_inst.items():
            raw = pl.concat([read_zip_csv(p) for p in paths], how="vertical_relaxed")
            if kind == "funding":
                df = raw.select(
                    pl.lit(inst_id).alias("inst_id"),
                    ms_to_utc("funding_time").alias("ts"),
                    pl.col("funding_rate").cast(pl.Float64),
                ).unique("ts").sort("ts")
            else:
                df = to_1h(raw, inst_id, "spot" if kind == "spot_1m" else "swap", ct_val.get(inst_id))
            lake.write(dataset, df, ts_col="ts", part=inst_id)
    print("入库结束", flush=True)


def cmd_ingest_1m() -> None:
    root = okx_raw()
    ct_val = current_ct_val()
    by_inst: dict[str, list] = {}
    for p in sorted((root / "swap_1m").glob("*/*.zip")):
        by_inst.setdefault(p.stem, []).append(p)
    print(f"okx_swap_1m: {len(by_inst)} 个合约", flush=True)
    for i, (inst_id, paths) in enumerate(by_inst.items(), 1):
        raw = pl.concat([read_zip_csv(p) for p in paths], how="vertical_relaxed")
        df = (
            raw.select(
                ms_to_utc("open_time").alias("ts"),
                *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close")),
                quote_volume("swap", ct_val.get(inst_id)).alias("vol_quote"),
            )
            .unique("ts")
            .sort("ts")
            .with_columns(pl.lit(inst_id).alias("inst_id"))
        )
        lake.write("okx_swap_1m", df, ts_col="ts", part=inst_id)
        if i % 25 == 0:
            print(f"  {i}/{len(by_inst)}", flush=True)
    print("入库结束", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "manifest":
        cmd_manifest()
    elif cmd == "download":
        asyncio.run(cmd_download_async())
    elif cmd == "ingest":
        cmd_ingest()
    elif cmd == "ingest1m":
        cmd_ingest_1m()
    else:
        sys.exit(__doc__)
