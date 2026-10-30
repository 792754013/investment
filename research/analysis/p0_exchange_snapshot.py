"""P0 交易所对比：用公开接口拉一次当前快照（不需要 API Key）。

对每家交易所的 USDT 本位永续，统计：
  - 是否有同所 USDT 现货（变体 A/B 的前提）
  - 现货能否作统一账户抵押品及一档折扣率（变体 B 的前提；Binance 需 Key，留空）
  - 上线天数、资金费结算间隔、当期费率、永续 OI、现货 24h 成交额
  - 是否满足 plan.md 9.6 节品种池初值（上线 ≥30 天、现货成交额 ≥500 万、OI ≥1000 万）

输出 data/p0/exchange_snapshot_<UTC 日期>.parquet，并打印汇总。
用法：uv run python research/analysis/p0_exchange_snapshot.py
"""

import asyncio
import re
from datetime import UTC, datetime

import httpx
import polars as pl

from lab.paths import data_root

NOW = datetime.now(UTC)
MIN_LISTED_DAYS = 30
MIN_SPOT_TURNOVER = 5e6
MIN_PERP_OI = 1e7

# 永续常用 1000PEPE 这类缩放名，现货是 PEPE；1INCH 这类不受影响
_SCALE = re.compile(r"^(1000000|100000|10000|1000|100)(?=[A-Z])")


def norm(base: str) -> str:
    return _SCALE.sub("", base.upper())


async def get(c: httpx.AsyncClient, url: str, **params) -> dict | list:
    r = await c.get(url, params=params)
    r.raise_for_status()
    return r.json()


async def binance(c: httpx.AsyncClient) -> pl.DataFrame:
    F, S = "https://fapi.binance.com", "https://api.binance.com"
    info, finfo, prem, fticks, sinfo, sticks = await asyncio.gather(
        get(c, f"{F}/fapi/v1/exchangeInfo"),
        get(c, f"{F}/fapi/v1/fundingInfo"),
        get(c, f"{F}/fapi/v1/premiumIndex"),
        get(c, f"{F}/fapi/v1/ticker/24hr"),
        get(c, f"{S}/api/v3/exchangeInfo", permissions="SPOT"),
        get(c, f"{S}/api/v3/ticker/24hr"),
    )
    perps = [
        s for s in info["symbols"]
        if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]
    interval = {x["symbol"]: x["fundingIntervalHours"] for x in finfo}
    p = {x["symbol"]: x for x in prem}
    ft = {x["symbol"]: float(x["quoteVolume"]) for x in fticks}
    spot = {s["symbol"]: s["baseAsset"] for s in sinfo["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
    spot_by_base = {norm(b): sym for sym, b in spot.items()}
    st = {x["symbol"]: float(x["quoteVolume"]) for x in sticks}

    sem = asyncio.Semaphore(10)

    async def oi(sym: str) -> float:
        async with sem:
            d = await get(c, f"{F}/fapi/v1/openInterest", symbol=sym)
            return float(d["openInterest"]) * float(p[sym]["markPrice"])

    ois = await asyncio.gather(*(oi(s["symbol"]) for s in perps))
    rows = []
    for s, o in zip(perps, ois):
        sym, base = s["symbol"], norm(s["baseAsset"])
        spot_sym = spot_by_base.get(base)
        rows.append(dict(
            venue="binance", symbol=sym, base=base,
            listed=datetime.fromtimestamp(s["onboardDate"] / 1000, UTC),
            funding_interval_h=float(interval.get(sym, 8)),
            funding_rate=float(p[sym]["lastFundingRate"]),
            perp_oi_usd=o, perp_turnover_usd=ft.get(sym, 0.0),
            has_spot=spot_sym is not None,
            spot_turnover_usd=st.get(spot_sym, 0.0) if spot_sym else 0.0,
            collateral_ratio=None,
        ))
    return pl.DataFrame(rows)


async def okx(c: httpx.AsyncClient) -> pl.DataFrame:
    B = "https://www.okx.com/api/v5"
    inst, fr, tk, oi, sinst, stk, disc = await asyncio.gather(
        get(c, f"{B}/public/instruments", instType="SWAP"),
        get(c, f"{B}/public/funding-rate", instId="ANY"),
        get(c, f"{B}/market/tickers", instType="SWAP"),
        get(c, f"{B}/public/open-interest", instType="SWAP"),
        get(c, f"{B}/public/instruments", instType="SPOT"),
        get(c, f"{B}/market/tickers", instType="SPOT"),
        get(c, f"{B}/public/discount-rate-interest-free-quota"),
    )
    perps = [
        x for x in inst["data"]
        if x["ctType"] == "linear" and x["settleCcy"] == "USDT" and x["state"] == "live"
    ]
    f = {x["instId"]: x for x in fr["data"]}
    t = {x["instId"]: x for x in tk["data"]}
    o = {x["instId"]: float(x["oiUsd"]) for x in oi["data"]}
    spot = {x["instId"]: norm(x["baseCcy"]) for x in sinst["data"] if x["quoteCcy"] == "USDT" and x["state"] == "live"}
    spot_by_base = {b: i for i, b in spot.items()}
    st = {x["instId"]: float(x["volCcy24h"]) for x in stk["data"]}  # 现货 volCcy24h 以计价币计
    col = {
        norm(x["ccy"]): float(x["details"][0]["discountRate"])
        for x in disc["data"] if x["details"] and not x["collateralRestrict"]
    }

    rows = []
    for x in perps:
        iid, base = x["instId"], norm(x["ctValCcy"])
        fx = f.get(iid, {})
        tx = t.get(iid, {})
        spot_id = spot_by_base.get(base)
        interval = (
            (int(fx["nextFundingTime"]) - int(fx["fundingTime"])) / 3.6e6
            if fx.get("nextFundingTime") else None
        )
        rows.append(dict(
            venue="okx", symbol=iid, base=base,
            listed=datetime.fromtimestamp(int(x["listTime"]) / 1000, UTC),
            funding_interval_h=interval,
            funding_rate=float(fx["fundingRate"]) if fx else None,
            perp_oi_usd=o.get(iid, 0.0),
            perp_turnover_usd=float(tx.get("volCcy24h") or 0) * float(tx.get("last") or 0),
            has_spot=spot_id is not None,
            spot_turnover_usd=st.get(spot_id, 0.0) if spot_id else 0.0,
            collateral_ratio=col.get(base),
        ))
    return pl.DataFrame(rows)


async def bybit(c: httpx.AsyncClient) -> pl.DataFrame:
    B = "https://api.bybit.com/v5"

    async def instruments(category: str) -> list[dict]:
        out, cursor = [], ""
        while True:
            d = await get(c, f"{B}/market/instruments-info", category=category, limit=1000, cursor=cursor)
            out += d["result"]["list"]
            cursor = d["result"].get("nextPageCursor")
            if not cursor:
                return out

    inst, tk, sinst, stk, coll = await asyncio.gather(
        instruments("linear"),
        get(c, f"{B}/market/tickers", category="linear"),
        instruments("spot"),
        get(c, f"{B}/market/tickers", category="spot"),
        get(c, f"{B}/spot-margin-trade/collateral"),
    )
    perps = [
        x for x in inst
        if x["contractType"] == "LinearPerpetual" and x["quoteCoin"] == "USDT" and x["status"] == "Trading"
    ]
    t = {x["symbol"]: x for x in tk["result"]["list"]}
    spot = {x["symbol"]: norm(x["baseCoin"]) for x in sinst if x["quoteCoin"] == "USDT" and x["status"] == "Trading"}
    spot_by_base = {b: s for s, b in spot.items()}
    st = {x["symbol"]: float(x["turnover24h"]) for x in stk["result"]["list"]}
    col = {
        norm(x["currency"]): float(x["collateralRatioList"][0]["collateralRatio"])
        for x in coll["result"]["list"] if x["collateralRatioList"]
    }

    rows = []
    for x in perps:
        sym, base = x["symbol"], norm(x["baseCoin"])
        tx = t.get(sym, {})
        spot_sym = spot_by_base.get(base)
        rows.append(dict(
            venue="bybit", symbol=sym, base=base,
            listed=datetime.fromtimestamp(int(x["launchTime"]) / 1000, UTC),
            funding_interval_h=int(x["fundingInterval"]) / 60,
            funding_rate=float(tx["fundingRate"]) if tx.get("fundingRate") else None,
            perp_oi_usd=float(tx.get("openInterestValue") or 0),
            perp_turnover_usd=float(tx.get("turnover24h") or 0),
            has_spot=spot_sym is not None,
            spot_turnover_usd=st.get(spot_sym, 0.0) if spot_sym else 0.0,
            collateral_ratio=col.get(base),
        ))
    return pl.DataFrame(rows)


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    eligible = (
        pl.col("has_spot")
        & (pl.col("listed_days") >= MIN_LISTED_DAYS)
        & (pl.col("spot_turnover_usd") >= MIN_SPOT_TURNOVER)
        & (pl.col("perp_oi_usd") >= MIN_PERP_OI)
    )
    return (
        df.with_columns(eligible.alias("pool_a"))
        .with_columns((pl.col("pool_a") & (pl.col("collateral_ratio") > 0)).alias("pool_b"))
        .group_by("venue")
        .agg(
            pl.len().alias("usdt_perps"),
            pl.col("has_spot").sum().alias("with_spot"),
            pl.col("pool_a").sum().alias("pool_a"),
            pl.col("pool_b").sum().alias("pool_b"),
            (pl.col("funding_interval_h") < 8).sum().alias("interval_lt_8h"),
            (pl.col("perp_oi_usd").sum() / 1e9).round(1).alias("oi_bn_usd"),
            pl.col("collateral_ratio").is_not_null().sum().alias("perps_w_collateral"),
        )
        .sort("venue")
    )


async def main() -> None:
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "Mozilla/5.0 research"}) as c:
        frames = await asyncio.gather(binance(c), okx(c), bybit(c))
    df = pl.concat(frames, how="vertical_relaxed").with_columns(
        ((pl.lit(NOW) - pl.col("listed")).dt.total_days()).alias("listed_days"),
        pl.lit(NOW).alias("snapshot_ts"),
    )
    out = data_root() / "p0" / f"exchange_snapshot_{NOW:%Y%m%d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)

    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    print(summarize(df))
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
