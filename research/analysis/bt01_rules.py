"""粗略规则回测（P1 第 5 步）：按结算粒度逐小时推进，只用样本内。

规则（plan.md 9.6 初值的简化版，没有 R3 预期费率模型，用当期费率代替）：
  入场：在池内；当期费率年化 ≥ ENTRY，且上一期 ≥ 0；按当期费率从高到低填满 N 个仓位
  出场：费率连续 2 期为负；或年化 < 5% 连续 3 期；或上线段结束（下架，滑点加倍）
  仓位：等名义，每仓名义 = 资金 / (N × 资本倍数)
  成本：G1-criteria 第 2 节悲观口径，开、平各 0.35% 名义
  损益：资金费 + 基差变化（空永续多现货：基差收窄为盈利）
  回撤：在每个结算小时按最新基差盯市

用法：uv run python research/analysis/bt01_rules.py
"""

import itertools
import time
from datetime import UTC, datetime

import httpx
import polars as pl

from lab import lake, trials
from lab.paths import data_root
from lab.seal import load_seal
from r01_exec_map import CAPITAL_MULT, SLIP, build

CAPITAL = 15_000
LEG_COST = 0.0035  # 单边（开或平）：现货 0.10% + 永续 0.05% + 两腿滑点各 0.10%
EXIT_APR = 0.05
GRID = dict(variant=["A", "B"], entry=[0.20, 0.30, 0.50], n_pos=[3, 5, 10])
OUT = data_root() / "research" / "bt01"


# ---------- 比较基准：OKX USDT 活期出借利率 ----------

def benchmark() -> pl.DataFrame:
    path = data_root() / "research" / "benchmark" / "okx_usdt_lending.parquet"
    if path.exists():
        return pl.read_parquet(path)
    end = int(load_seal().start.timestamp() * 1000)
    rows, after = [], end
    with httpx.Client(headers={"User-Agent": "Mozilla/5.0 research"}, timeout=30) as c:
        while True:
            d = c.get("https://www.okx.com/api/v5/finance/savings/lending-rate-history",
                      params=dict(ccy="USDT", limit=100, after=after)).json()
            if d["code"] != "0":
                time.sleep(2)
                continue
            if not d["data"]:
                break
            rows += [(int(x["ts"]), float(x["lendingRate"])) for x in d["data"]]
            after = min(int(x["ts"]) for x in d["data"])
            time.sleep(0.25)
    df = (
        pl.DataFrame(rows, schema=["ts", "lending_rate"], orient="row")
        .with_columns(pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC"))
        .unique("ts").sort("ts")
        .filter(pl.col("ts") < load_seal().start)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


# ---------- 数据 ----------

def closes(dataset: str, id_col: str) -> pl.DataFrame:
    """小时 K 线收盘价，对齐到收盘时刻（ts + 1h），与结算时刻同一口径。"""
    return lake.scan(dataset).select(
        pl.col("inst_id").alias(id_col), (pl.col("ts") + pl.duration(hours=1)).alias("hour"), "close"
    ).collect()


def frame() -> pl.DataFrame:
    f = build()
    f = (
        f.join(closes("okx_swap_1h", "inst_id").rename({"close": "perp_px"}), on=["inst_id", "hour"], how="left")
        .join(closes("okx_spot_1h", "spot_id").rename({"close": "spot_px"}), on=["spot_id", "hour"], how="left")
        .with_columns((pl.col("perp_px") / pl.col("spot_px") - 1).alias("basis"))
        .sort("inst_id", "ts")
        .with_columns(pl.col("apr_raw").shift(1).over("inst_id", "episode").alias("apr_prev"))
    )
    return f.select("hour", "ts", "inst_id", "episode", "ep_end", "funding_rate", "apr_raw", "apr_prev",
                    "basis", "pool_a", "pool_b").sort("hour")


# ---------- 回测 ----------

def run(groups: list, variant: str, entry: float, n_pos: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    pool = "pool_a" if variant == "A" else "pool_b"
    notional = CAPITAL / (n_pos * CAPITAL_MULT[variant])
    pos: dict[tuple, dict] = {}
    realized, trades, curve = 0.0, [], []
    last_basis: dict[tuple, float] = {}

    for hour, rows in groups:
        for r in rows:
            key = (r["inst_id"], r["episode"])
            if r["basis"] is not None:
                last_basis[key] = r["basis"]
            p = pos.get(key)
            if p is None:
                continue
            fr = r["funding_rate"] or 0.0
            p["funding"] += fr * notional
            p["neg"] = p["neg"] + 1 if fr < 0 else 0
            p["low"] = p["low"] + 1 if (r["apr_raw"] or 0) < EXIT_APR else 0
            delist = r["ts"] == r["ep_end"]
            if p["neg"] >= 2 or p["low"] >= 3 or delist:
                b_exit = last_basis.get(key, p["b_entry"])
                cost = notional * (LEG_COST + (SLIP if delist else 0))
                pnl = p["funding"] + notional * (p["b_entry"] - b_exit) - p["cost_open"] - cost
                realized += pnl
                trades.append(dict(inst_id=r["inst_id"], entry_ts=p["ts"], exit_ts=r["ts"], pnl=pnl,
                                   funding=p["funding"], reason="delist" if delist else ("neg" if p["neg"] >= 2 else "low")))
                del pos[key]

        free = n_pos - len(pos)
        if free > 0:
            cands = sorted(
                (r for r in rows if r[pool] and r["apr_raw"] is not None and r["apr_raw"] >= entry
                 and (r["apr_prev"] or 0) >= 0 and r["basis"] is not None and r["ts"] != r["ep_end"]
                 and (r["inst_id"], r["episode"]) not in pos),
                key=lambda r: -r["apr_raw"],
            )[:free]
            for r in cands:
                pos[(r["inst_id"], r["episode"])] = dict(
                    ts=r["ts"], b_entry=r["basis"], funding=0.0, neg=0, low=0, cost_open=notional * LEG_COST)

        unreal = sum(
            p["funding"] + notional * (p["b_entry"] - last_basis.get(k, p["b_entry"])) - p["cost_open"]
            for k, p in pos.items()
        )
        curve.append((hour, realized + unreal, len(pos)))

    eq = pl.DataFrame(curve, schema=["hour", "pnl", "n_open"], orient="row")
    return eq, pl.DataFrame(trades) if trades else pl.DataFrame()


def btc_regime() -> pl.DataFrame:
    b = lake.scan("okx_spot_1h").filter(pl.col("inst_id") == "BTC-USDT").select("ts", "close").collect()
    d = b.group_by(pl.col("ts").dt.date().alias("day")).agg(pl.col("close").last()).sort("day")
    return d.with_columns((pl.col("close") / pl.col("close").shift(90) - 1).alias("r90")).with_columns(
        pl.when(pl.col("r90") > 0.2).then(pl.lit("牛市")).when(pl.col("r90") < -0.2).then(pl.lit("熊市"))
        .otherwise(pl.lit("震荡")).alias("regime")
    ).select("day", "regime")


def evaluate(eq: pl.DataFrame, trades: pl.DataFrame, bench: float, regime: pl.DataFrame) -> dict:
    daily = (
        eq.group_by(pl.col("hour").dt.date().alias("day")).agg(pl.col("pnl").last(), pl.col("n_open").mean())
        .sort("day").with_columns((pl.col("pnl").diff().fill_null(pl.col("pnl"))).alias("d_pnl"))
    )
    years = (daily["day"].max() - daily["day"].min()).days / 365
    equity = CAPITAL + daily["pnl"]
    mdd = float((1 - equity / equity.cum_max()).max())
    roll = daily.with_columns(
        (pl.col("pnl") - pl.col("pnl").shift(180)).alias("r180")
    )["r180"].min()
    reg = daily.join(regime, on="day", how="left").group_by("regime").agg(
        (pl.col("d_pnl").sum() / CAPITAL).alias("ret"), pl.len().alias("days"))
    yearly = daily.group_by(pl.col("day").dt.year().alias("year")).agg(
        (pl.col("d_pnl").sum() / CAPITAL).alias("ret"), pl.len().alias("days"))
    net_apr = float(daily["pnl"][-1]) / CAPITAL / years
    return dict(
        net_apr=net_apr, excess_vs_bench=net_apr - bench, mdd=mdd,
        worst_180d=float(roll) / CAPITAL if roll is not None else None,
        trades=trades.height, avg_open=float(daily["n_open"].mean()),
        regimes={r: round(v, 4) for r, v in zip(reg["regime"], reg["ret"])},
        years={str(y): round(v * 365 / d, 4) for y, v, d in zip(yearly["year"], yearly["ret"], yearly["days"])},
    )


def main() -> None:
    bench_df = benchmark()
    bench = float(bench_df["lending_rate"].mean())
    bench_year = bench_df.group_by(pl.col("ts").dt.year().alias("year")).agg(pl.col("lending_rate").mean()).sort("year")
    print(f"样本内 USDT 活期出借利率均值 {bench:.2%}；G1 门槛 = {bench + 0.05:.2%}\n", bench_year)

    f = frame()
    groups = [(h, g.to_dicts()) for (h,), g in f.group_by(["hour"], maintain_order=True)]
    regime = btc_regime()
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for variant, entry, n_pos in itertools.product(*GRID.values()):
        eq, tr = run(groups, variant, entry, n_pos)
        m = evaluate(eq, tr, bench, regime)
        tag = f"{variant}_e{int(entry * 100)}_n{n_pos}"
        eq.write_parquet(OUT / f"equity_{tag}.parquet")
        if tr.height:
            tr.write_parquet(OUT / f"trades_{tag}.parquet")
        trials.record("bt01_rules", variant=variant,
                      params=dict(entry=entry, n_pos=n_pos, exit_apr=EXIT_APR, leg_cost=LEG_COST, capital=CAPITAL),
                      metrics={k: m[k] for k in ("net_apr", "mdd", "worst_180d", "trades")} | {"years": m["years"]},
                      sample="in_sample")
        res.append(dict(variant=variant, entry=entry, n_pos=n_pos, **{k: v for k, v in m.items() if k not in ("regimes", "years")},
                        **{f"reg_{k}": v for k, v in m["regimes"].items()}, **{f"y{k}": v for k, v in m["years"].items()}))
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)

    t = pl.DataFrame(res)
    t.write_parquet(OUT / "grid.parquet")
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    print("\n参数网格\n", t.sort("variant", "entry", "n_pos"))


if __name__ == "__main__":
    main()
