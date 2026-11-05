"""下载 Binance U 本位永续 BTCUSDT、ETHUSDT 的公开历史数据并入库（OPP-006 第 2 轮 D4，docs/research/OPP006-ml-plan.md）。

数据来源都是公开的（data.binance.vision），不需要 API Key：
  - 逐笔成交 aggTrades：月度归档，入库时聚合为 1 秒 K 线 → lake: bn_swap_1s
  - 持仓量、多空比 metrics：日度归档，5 分钟一条 → lake: bn_metrics
只下载到 2026-04（之后是封存段，本轮用不到）。

用法（可重复运行，已下载的文件会跳过）：
  uv run python research/ingest/binance_trades.py download
  uv run python research/ingest/binance_trades.py ingest
"""

import asyncio
import io
import sys
import zipfile
from datetime import date, timedelta

import httpx
import polars as pl

from lab import lake
from lab.paths import raw_dir

BASE = "https://data.binance.vision/data/futures/um"
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
FIRST, LAST = date(2022, 1, 1), date(2026, 4, 30)
AGG_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]


def root():
    return raw_dir() / "binance"


def months() -> list[str]:
    out, d = [], FIRST
    while d <= LAST:
        out.append(f"{d.year}-{d.month:02d}")
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def jobs() -> list[tuple[str, str]]:
    out = []
    for s in SYMBOLS:
        for ym in months():
            out.append((f"{BASE}/monthly/aggTrades/{s}/{s}-aggTrades-{ym}.zip", f"aggTrades/{s}/{ym}.zip"))
        d = FIRST
        while d <= LAST:
            out.append((f"{BASE}/daily/metrics/{s}/{s}-metrics-{d}.zip", f"metrics/{s}/{d}.zip"))
            d += timedelta(days=1)
    return out


async def cmd_download_async() -> None:
    todo = [j for j in jobs() if not (root() / j[1]).exists() and not (root() / (j[1] + ".missing")).exists()]
    print(f"待下载 {len(todo)} 个文件", flush=True)
    big = asyncio.Semaphore(3)
    small = asyncio.Semaphore(12)
    done = 0

    async def fetch(c: httpx.AsyncClient, url: str, rel: str) -> None:
        nonlocal done
        path = root() / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        async with (big if "aggTrades" in rel else small):
            for attempt in range(5):
                try:
                    tmp = path.with_suffix(".tmp")
                    async with c.stream("GET", url) as r:
                        if r.status_code == 404:
                            path.with_name(path.name + ".missing").touch()
                            break
                        r.raise_for_status()
                        with tmp.open("wb") as f:
                            async for chunk in r.aiter_bytes(1 << 20):
                                f.write(chunk)
                    tmp.replace(path)
                    break
                except (httpx.HTTPError, OSError):
                    await asyncio.sleep(2 ** attempt)
            else:
                print(f"  失败 {url}", flush=True)
        done += 1
        if "aggTrades" in rel:
            print(f"  {rel} 完成（{done}/{len(todo)}）", flush=True)

    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as c:
        await asyncio.gather(*(fetch(c, u, r) for u, r in todo))
    print("下载结束", flush=True)


def read_zip_csv(path, names: list[str] | None = None) -> pl.DataFrame:
    with zipfile.ZipFile(path) as z:
        raw = z.read(next(n for n in z.namelist() if n.endswith(".csv")))
    has_header = not raw[:1].isdigit()
    return pl.read_csv(io.BytesIO(raw), has_header=has_header, new_columns=names if not has_header else None,
                       infer_schema_length=10_000)


def to_1s(df: pl.DataFrame, inst: str) -> pl.DataFrame:
    quote = pl.col("price") * pl.col("quantity")
    return (
        df.select(
            pl.from_epoch(pl.col("transact_time").cast(pl.Int64), time_unit="ms").dt.replace_time_zone("UTC").alias("t"),
            pl.col("agg_trade_id"), pl.col("price").cast(pl.Float64), pl.col("quantity").cast(pl.Float64),
            pl.col("is_buyer_maker").cast(pl.String).str.to_lowercase() == "true",
        )
        .sort("agg_trade_id")
        .with_columns(pl.col("t").dt.truncate("1s").alias("ts"))
        .group_by("ts", maintain_order=True)
        .agg(
            pl.col("price").first().alias("open"), pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"), pl.col("price").last().alias("close"),
            quote.sum().alias("vol_quote"),
            quote.filter(~pl.col("is_buyer_maker")).sum().alias("buy_quote"),  # 买方主动成交
            pl.len().alias("n_trades"),
            quote.max().alias("max_trade_quote"),
        )
        .with_columns(pl.lit(inst).alias("inst_id"))
    )


def cmd_ingest() -> None:
    for s in SYMBOLS:
        inst = f"{s[:-4]}-USDT-SWAP"
        for p in sorted((root() / "aggTrades" / s).glob("*.zip")):
            df = to_1s(read_zip_csv(p, AGG_COLS), inst)
            lake.write("bn_swap_1s", df, ts_col="ts", part=f"{inst}-{p.stem}")
            print(f"  bn_swap_1s {inst} {p.stem}: {df.height} 秒", flush=True)
        frames = [read_zip_csv(p) for p in sorted((root() / "metrics" / s).glob("*.zip"))]
        m = pl.concat([f.select(
            pl.col("create_time").cast(pl.String).str.to_datetime(time_zone="UTC").alias("ts"),
            pl.col("sum_open_interest_value").cast(pl.Float64, strict=False).alias("oi_value"),
            pl.col("count_long_short_ratio").cast(pl.Float64, strict=False).alias("ls_ratio"),
            pl.col("sum_toptrader_long_short_ratio").cast(pl.Float64, strict=False).alias("top_ls_ratio"),
            pl.col("sum_taker_long_short_vol_ratio").cast(pl.Float64, strict=False).alias("taker_ls_vol"),
        ) for f in frames]).unique("ts").sort("ts").with_columns(pl.lit(inst).alias("inst_id"))
        lake.write("bn_metrics", m, ts_col="ts", part=inst)
        print(f"  bn_metrics {inst}: {m.height} 条", flush=True)
    print("入库结束", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "download":
        asyncio.run(cmd_download_async())
    elif cmd == "ingest":
        cmd_ingest()
    else:
        sys.exit(__doc__)
