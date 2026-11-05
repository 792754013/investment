"""OPP-007 日线趋势跟踪：快速否证（docs/research/OPP007-trend-plan.md，看结果之前写定）。

数据：bn_spot_1d（Binance 现货日线）、bn_funding（U 本位永续资金费）。只用样本内。
信号在第 t 天 UTC 收盘计算；按第 t+1 天收盘成交（LAG=1），持有期收益从 t+2 天起算；延迟敏感性 LAG=2。
仓位 = 趋势强度 × 目标年化波动 / 近 60 日年化波动，截到 ±杠杆上限；仓位变化 < 0.05 不调。
成本：调仓成交额的 0.1%（吃单 0.05% + 滑点 0.05%）；资金费按实际，缺数据时做多按每天 0.03% 付、做空不收；现金不计息。

用法：uv run python research/analysis/r08_trend_daily.py
"""

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl

from lab import lake, trials
from lab.paths import data_root

OUT = data_root() / "research" / "r08_trend"
EXPERIMENT = "r08_trend"
MAIN = ["BTCUSDT", "ETHUSDT"]
AUX = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "ADAUSDT", "LTCUSDT", "EOSUSDT", "XLMUSDT", "NEOUSDT", "TRXUSDT", "BNBUSDT"]
MA_N, MOM_N, DON_N = [20, 50, 100, 200], [20, 60, 120, 250], [20, 55, 100]
COST, MISSING_FUND_LONG, REBAL_BAND = 0.001, 0.0003, 0.05
TEST_START = datetime(2023, 1, 1, tzinfo=UTC)
CAPITAL_RMB = 300_000
WARMUP = 260


def load(insts: list[str]) -> dict[str, pl.DataFrame]:
    px = lake.scan("bn_spot_1d").filter(pl.col("inst_id").is_in(insts)).select("ts", "inst_id", "high", "low", "close").collect()
    fu = (lake.scan("bn_funding").filter(pl.col("inst_id").is_in(insts)).collect()
          .with_columns(pl.col("ts").dt.truncate("1d").alias("ts")).group_by("inst_id", "ts")
          .agg(pl.col("funding_rate").sum().alias("fund")))
    out = {}
    for i in insts:
        d = px.filter(pl.col("inst_id") == i).sort("ts").join(fu.filter(pl.col("inst_id") == i).drop("inst_id"), on="ts", how="left")
        out[i] = d.with_columns((pl.col("close") / pl.col("close").shift() - 1).alias("ret"),
                                (pl.col("low") / pl.col("close").shift() - 1).alias("lo_ret"),
                                (pl.col("high") / pl.col("close").shift() - 1).alias("hi_ret"))
    return out


def signals(d: pl.DataFrame) -> dict[str, np.ndarray]:
    c = d["close"].to_numpy()
    s = {}
    cs = pl.Series(c)
    for n in MA_N:
        ma = cs.rolling_mean(n).to_numpy()
        s[f"ma{n}"] = np.where(np.isnan(ma), np.nan, np.sign(c - ma))
    for n in MOM_N:
        prev = np.concatenate([np.full(n, np.nan), c[:-n]])
        s[f"mom{n}"] = np.where(np.isnan(prev), np.nan, np.sign(c / prev - 1))
    hi, lo = d["high"].to_numpy(), d["low"].to_numpy()
    for n in DON_N:
        hh = pl.Series(hi).rolling_max(n).shift(1).to_numpy()
        ll = pl.Series(lo).rolling_min(n // 2).shift(1).to_numpy()
        lll = pl.Series(lo).rolling_min(n).shift(1).to_numpy()
        hhh = pl.Series(hi).rolling_max(n // 2).shift(1).to_numpy()
        st, out = 0, np.full(len(c), np.nan)
        for t in range(len(c)):
            if np.isnan(hh[t]) or np.isnan(lll[t]):
                continue
            if st <= 0 and c[t] > hh[t]:
                st = 1
            elif st >= 0 and c[t] < lll[t]:
                st = -1
            elif st == 1 and c[t] < ll[t]:
                st = 0
            elif st == -1 and c[t] > hhh[t]:
                st = 0
            out[t] = st
        s[f"don{n}"] = out
    return s


def strength(sig: dict[str, np.ndarray], names: list[str], long_only: bool) -> np.ndarray:
    x = np.vstack([sig[n] for n in names])
    if long_only:
        x = np.where(np.isnan(x), np.nan, np.maximum(x, 0))
    return np.nanmean(x, axis=0) if len(names) > 1 else x[0]


def sleeve(d: pl.DataFrame, st: np.ndarray, target: float, cap: float, lag: int) -> dict[str, np.ndarray]:
    """单个品种的每日收益（以分配给它的资金为单位）。"""
    r = d["ret"].fill_null(0.0).to_numpy()
    lo, hi = d["lo_ret"].fill_null(0.0).to_numpy(), d["hi_ret"].fill_null(0.0).to_numpy()
    fund = d["fund"].to_numpy()
    vol = pl.Series(np.log1p(r)).rolling_std(60).to_numpy() * np.sqrt(365)
    want = np.clip(np.nan_to_num(st) * target / np.where(vol > 0, vol, np.nan), -cap, cap)
    want = np.nan_to_num(want)
    n = len(r)
    pos = np.zeros(n)
    cur = 0.0
    for t in range(n):
        k = t - lag  # 第 t 天收盘执行的是 t−lag 天的信号
        if k >= 0 and abs(want[k] - cur) >= REBAL_BAND:
            cur = want[k]
        elif k >= 0 and want[k] == 0 and cur != 0:
            cur = 0.0
        pos[t] = cur
    held = np.concatenate([[0.0], pos[:-1]])  # 第 t 天的收益由 t−1 天收盘的仓位承担
    trade = np.abs(np.diff(np.concatenate([[0.0], pos])))
    f = np.where(np.isnan(fund), np.where(held > 0, MISSING_FUND_LONG, 0.0), fund)
    pnl = held * r - trade * COST - held * f
    # 当天最不利的盘中波动乘以仓位，≤ −(1/仓位 − 0.5%) 视为这一份资金爆仓
    worst = np.where(held > 0, held * lo, np.where(held < 0, held * hi, 0.0))
    bust = worst <= -(1 - 0.005 * np.abs(held))
    pnl = np.where(bust, -1.0, pnl)
    return {"pnl": pnl, "pos": held, "trade": trade, "bust": bust, "fund_cost": held * f}


def portfolio(data, insts, fn) -> pl.DataFrame:
    """等权组合：每天在有足够历史的品种间平分资金（每天再平衡）。"""
    frames = []
    for i in insts:
        d = data[i]
        res = fn(d)
        frames.append(pl.DataFrame({"ts": d["ts"], i: res["pnl"], f"{i}_ok": np.arange(d.height) >= WARMUP,
                                    f"{i}_trade": res["trade"], f"{i}_bust": res["bust"], f"{i}_pos": res["pos"]}))
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, on="ts", how="full", coalesce=True)
    df = df.sort("ts")
    ok = np.column_stack([df[f"{i}_ok"].fill_null(False).to_numpy() for i in insts])
    pnl = np.column_stack([df[i].fill_null(0.0).to_numpy() for i in insts])
    tr = np.column_stack([df[f"{i}_trade"].fill_null(0.0).to_numpy() for i in insts])
    pos = np.column_stack([df[f"{i}_pos"].fill_null(0.0).to_numpy() for i in insts])
    bust = np.column_stack([df[f"{i}_bust"].fill_null(False).to_numpy() for i in insts])
    w = ok / np.maximum(ok.sum(axis=1, keepdims=True), 1)
    out = pl.DataFrame({"ts": df["ts"], "ret": (w * pnl).sum(axis=1), "turnover": (w * tr).sum(axis=1),
                        "gross": (w * np.abs(pos)).sum(axis=1), "bust": (bust & ok).any(axis=1), "live": ok.any(axis=1)})
    return out.filter(pl.col("live"))


def stats(p: pl.DataFrame) -> dict:
    r = p["ret"].to_numpy()
    eq = np.cumprod(1 + r)
    yrs = len(r) / 365
    dead = bool((eq <= 0).any())
    cagr = float(eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0
    dd = float((1 - eq / np.maximum.accumulate(eq)).max())
    yr = (p.with_columns(pl.col("ts").dt.year().alias("y")).group_by("y")
          .agg(((1 + pl.col("ret")).product() - 1).alias("r"), pl.len().alias("n")).sort("y"))
    by = {int(a): round(float(b), 4) for a, b in zip(yr["y"], yr["r"])}
    best = max(by, key=by.get)
    rest = p.filter(pl.col("ts").dt.year() != best)["ret"].to_numpy()
    rest_cagr = float(np.prod(1 + rest) ** (365 / len(rest)) - 1) if len(rest) else None
    tp = p.filter(pl.col("ts") >= TEST_START)["ret"].to_numpy()
    test_cagr = float(np.prod(1 + tp) ** (365 / len(tp)) - 1) if len(tp) else None
    sd = r.std() * np.sqrt(365)
    return {
        "start": str(p["ts"][0].date()), "cagr": round(cagr, 4), "vol": round(float(sd), 4),
        "sharpe": round(float(r.mean() * 365 / sd), 3) if sd > 0 else None, "max_dd": round(dd, 4),
        "years": by, "pos_years": round(sum(v > 0 for v in by.values()) / len(by), 3),
        "best_year": best, "cagr_ex_best": round(rest_cagr, 4) if rest_cagr is not None else None,
        "test_cagr": round(test_cagr, 4) if test_cagr is not None else None,
        "turnover_yr": round(float(p["turnover"].sum() / yrs), 2), "avg_gross": round(float(p["gross"].mean()), 3),
        "bust_days": int(p["bust"].sum()), "dead": dead, "annual_rmb": round(cagr * CAPITAL_RMB),
    }


def bh(data, insts) -> pl.DataFrame:
    def fn(d):
        n = d.height
        return {"pnl": d["ret"].fill_null(0.0).to_numpy(), "trade": np.zeros(n), "bust": np.zeros(n, bool), "pos": np.ones(n)}
    return portfolio(data, insts, fn)


def run(data, insts, names, long_only, target, cap, lag, sigs):
    return portfolio(data, insts, lambda d: sleeve(d, strength(sigs[d["inst_id"][0]], names, long_only), target, cap, lag))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(AUX)
    sigs = {i: signals(d) for i, d in data.items()}
    res = {}

    def rec(variant, params, p, ref=None):
        s = stats(p if ref is None else p.filter(pl.col("ts") >= ref["ts"][0]))
        res[variant] = {"params": params, **s}
        trials.record(EXPERIMENT, variant=variant, sample="in_sample", params=params, metrics=s,
                      note="OPP-007 日线趋势快速否证（样本内，封存段未动）")
        return s

    main_names = [f"ma{n}" for n in MA_N] + [f"mom{n}" for n in MOM_N]
    for uni_name, uni in (("main", MAIN), ("aux", AUX)):
        b = bh(data, uni)
        ref = run(data, uni, main_names, True, 0.4, 1.0, 1, sigs)  # 主策略的起点（预热之后）作为基准起点
        b = b.filter(pl.col("ts") >= ref["ts"][0])
        rec(f"{uni_name}_buyhold", {"universe": uni}, b)
        for long_only in (True, False):
            side = "long" if long_only else "ls"
            for target in (0.2, 0.4, 0.6):
                for cap in (1.0, 2.0, 3.0):
                    for lag in ((1, 2) if (target, cap) == (0.4, 1.0) else (1,)):
                        p = run(data, uni, main_names, long_only, target, cap, lag, sigs)
                        rec(f"{uni_name}_ens_{side}_tv{target}_cap{cap}_lag{lag}",
                            {"universe": uni_name, "signals": "ens8", "side": side, "target_vol": target, "cap": cap, "lag": lag}, p)
            for nm in list(sigs[MAIN[0]].keys()):
                p = run(data, uni, [nm], long_only, 0.4, 1.0, 1, sigs)
                rec(f"{uni_name}_{nm}_{side}", {"universe": uni_name, "signal": nm, "side": side, "target_vol": 0.4,
                                                "cap": 1.0, "lag": 1}, p, ref)
        # 主策略和基准的净值曲线
        if uni_name == "main":
            m = run(data, uni, main_names, True, 0.4, 1.0, 1, sigs)
            b.select("ts", pl.col("ret").alias("bh")).join(m.select("ts", pl.col("ret").alias("trend"), "gross"), on="ts") \
                .write_parquet(OUT / "main_curve.parquet")

    # 判定（主策略：main、只做多、目标波动 40%、上限 1 倍、LAG=1）
    m, b = res["main_ens_long_tv0.4_cap1.0_lag1"], res["main_buyhold"]
    d = res["main_ens_long_tv0.4_cap1.0_lag2"]
    verdict = {
        "1_cagr_gt_4pct": m["cagr"] > 0.04,
        "2_better_than_hold": (m["sharpe"] or 0) > (b["sharpe"] or 0) or m["max_dd"] <= b["max_dd"] * 2 / 3,
        "3_pos_years_60": m["pos_years"] >= 0.6,
        "4_ex_best_year_pos": (m["cagr_ex_best"] or -1) > 0,
        "5_test_pos": (m["test_cagr"] or -1) > 0,
        "6_g0": m["annual_rmb"] >= 3000,
    }
    verdict["pass"] = all(verdict.values())
    verdict["note_delay_halves"] = d["cagr"] < m["cagr"] / 2
    res["verdict"] = verdict
    trials.record(EXPERIMENT, variant="verdict", sample="in_sample", params={"main": "main_ens_long_tv0.4_cap1.0_lag1"},
                  metrics=verdict, note="OPP-007 快速否证判定")
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({"main": m, "buyhold": b, "delay": d, "verdict": verdict}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
