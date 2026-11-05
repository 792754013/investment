"""A 股宽基指数日线趋势（OPP-007 主策略 B0 移植）快速否证。方案：docs/research/ATREND-plan.md（看结果之前写定）。

数据：lake idx_long（中证全收益指数 H00300/H00905/H00016/H00852，创业板指 399006 价格指数），cash_idx（同业存单 AAA 指数）。
口径：第 t 天收盘出信号，第 t+1 天收盘成交，收益从 t+2 天起；成交额 0.1%；权重变化 < 0.05 不调；只做多不加杠杆。
年化按 243 个交易日。只用样本内（2026-05-01 之前）。

用法：uv run python research/analysis/r10_atrend.py
"""

import json
import math

import numpy as np
import pandas as pd

from lab import lake, trials
from lab.paths import data_root

EXPERIMENT = "r10_atrend"
OUT = data_root() / "research" / "r10_atrend"
AN = 243
COST, BAND = 0.001, 0.05
START = pd.Timestamp("2006-01-01", tz="UTC")  # 全收益指数 2005-01 起才有日线，250 日信号要到 2006 年初才齐（方案写 2005，报告注明）
DEV_END = pd.Timestamp("2019-01-01", tz="UTC")
MA_N, MOM_N, DON_N = [20, 50, 100, 200], [20, 60, 120, 250], [20, 55, 100]
MAIN = ["H00300", "H00905"]
NAMES = {"H00300": "沪深300全收益", "H00905": "中证500全收益", "H00016": "上证50全收益", "H00852": "中证1000全收益（2014-10 前为回算）",
         "399006": "创业板指（价格指数）"}


def load() -> pd.DataFrame:
    d = lake.scan("idx_long").collect().to_pandas()
    C = d.pivot_table(index="ts", columns="index", values="close").sort_index()
    return C


def cash_ret(idx: pd.DatetimeIndex) -> pd.Series:
    """闲置资金日收益：2016 年起用同业存单 AAA 指数；之前没有数据，按年 2.5% 计（方案外的补充，报告注明）。"""
    c = lake.scan("cash_idx").collect().to_pandas().set_index("ts")["close"].sort_index()
    r = (c / c.shift() - 1).reindex(idx)
    fixed = pd.Series(0.025 / AN, index=idx)
    return r.where(idx >= c.index.min(), fixed).fillna(0.0)


def sigs(c: pd.Series) -> dict[str, pd.Series]:
    s = {}
    for n in MA_N:
        s[f"ma{n}"] = np.sign(c - c.rolling(n).mean()).where(c.rolling(n).mean().notna())
    for n in MOM_N:
        s[f"mom{n}"] = np.sign(c / c.shift(n) - 1)
    for n in DON_N:  # 只有收盘价，用收盘价通道
        hh, ll = c.rolling(n).max().shift(), c.rolling(n // 2).min().shift()
        lll, hhh = c.rolling(n).min().shift(), c.rolling(n // 2).max().shift()
        st, out = 0, np.full(len(c), np.nan)
        cv, a, b, e, f = c.to_numpy(), hh.to_numpy(), ll.to_numpy(), lll.to_numpy(), hhh.to_numpy()
        for t in range(len(c)):
            if np.isnan(a[t]) or np.isnan(e[t]):
                continue
            if st <= 0 and cv[t] > a[t]:
                st = 1
            elif st >= 0 and cv[t] < e[t]:
                st = -1
            elif st == 1 and cv[t] < b[t]:
                st = 0
            elif st == -1 and cv[t] > f[t]:
                st = 0
            out[t] = st
        s[f"don{n}"] = pd.Series(out, index=c.index)
    return s


def strength(c: pd.Series, names: list[str]) -> pd.Series:
    s = sigs(c)
    x = pd.concat({n: s[n].clip(lower=0) for n in names}, axis=1)
    return x.mean(1, skipna=False) if len(names) > 1 else x.iloc[:, 0]


ENS8 = [f"ma{n}" for n in MA_N] + [f"mom{n}" for n in MOM_N]


def vol(r: pd.Series, n=60) -> pd.Series:
    return np.log1p(r).rolling(n, min_periods=n // 2).std() * math.sqrt(AN)


def weights(C: pd.DataFrame, insts: list[str], tv: float | None, names=ENS8) -> pd.DataFrame:
    share = 1 / len(insts)
    W = {}
    for a in insts:
        c = C[a].dropna()
        st = strength(c, names)
        r = c / c.shift() - 1
        w = st if tv is None else (st * tv / vol(r)).clip(0, 1)
        W[a] = share * w.fillna(0.0)
    return pd.DataFrame(W).reindex(C.index)


def execute(W: pd.DataFrame, R: pd.DataFrame, lag=1, cash: pd.Series | None = None) -> pd.Series:
    w = W.fillna(0.0).to_numpy()
    r = R[W.columns].to_numpy()
    alive = ~np.isnan(r)
    rr = np.nan_to_num(r)
    T, N = w.shape
    pos, cur = np.zeros((T, N)), np.zeros(N)
    for t in range(T):
        k = t - lag
        if k >= 0:
            tgt = w[k] * alive[t]
            ch = (np.abs(tgt - cur) >= BAND) | ((tgt == 0) & (cur != 0))
            cur = np.where(ch, tgt, cur)
        pos[t] = cur
    held = np.vstack([np.zeros(N), pos[:-1]])
    trade = np.abs(np.diff(np.vstack([np.zeros(N), pos]), axis=0))
    pnl = (held * rr).sum(1) - trade.sum(1) * COST
    if cash is not None:
        pnl = pnl + (1 - held.sum(1)) * cash.to_numpy()
    out = pd.Series(pnl, index=R.index)
    out.attrs["gross"] = pd.Series(held.sum(1), index=R.index)
    out.attrs["turn"] = pd.Series(trade.sum(1), index=R.index)
    return out


def stats(p: pd.Series) -> dict:
    p = p.dropna()
    if len(p) < 60:
        return {}
    eq = (1 + p).cumprod()
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    sd = p.std() * math.sqrt(AN)
    dd = float((1 - eq / eq.cummax()).max())
    return {"cagr": round(float(cagr), 4), "sharpe": round(float(p.mean() * AN / sd), 3) if sd > 0 else 0.0,
            "max_dd": round(dd, 4), "calmar": round(float(cagr / dd), 3) if dd > 0 else None, "vol": round(float(sd), 4)}


def years(p: pd.Series) -> dict:
    return {int(y): round(float((1 + g).prod() - 1), 4) for y, g in p.groupby(p.index.year)}


def full(p: pd.Series, extra=None) -> dict:
    p = p[p.index >= START]
    d = {"full": stats(p), "dev": stats(p[p.index < DEV_END]), "test": stats(p[p.index >= DEV_END]), "years": years(p)}
    if "gross" in p.attrs or extra is not None:
        pass
    return d


def episodes(p: pd.Series, b: pd.Series) -> dict:
    eps = {"2008 金融危机": ("2007-10-16", "2008-11-04"), "2015 股灾": ("2015-06-12", "2016-01-28"),
           "2018 熊市": ("2018-01-24", "2019-01-03"), "2021-2024 熊市": ("2021-02-18", "2024-09-13"),
           "2006-2007 牛市": ("2006-01-01", "2007-10-16"), "2014-2015 牛市": ("2014-07-01", "2015-06-12"),
           "2024-09 至 2026-04": ("2024-09-13", "2026-04-30")}
    out = {}
    for k, (a, z) in eps.items():
        m = (p.index >= pd.Timestamp(a, tz="UTC")) & (p.index <= pd.Timestamp(z, tz="UTC") + pd.Timedelta(days=1))
        out[k] = {"strategy": round(float((1 + p[m]).prod() - 1), 4), "hold": round(float((1 + b[m]).prod() - 1), 4)}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    C = load()
    res = {}

    def rec(variant, params, metrics, note):
        trials.record(EXPERIMENT, variant=variant, sample="in_sample", params=params, metrics=metrics, note=note)

    def strat(insts, tv=0.2, names=ENS8, lag=1, cash=None):
        sub = C[insts].dropna()  # 只用这些指数都有收盘价的交易日
        W = weights(sub, insts, tv, names)
        return execute(W, (sub / sub.shift() - 1), lag, None if cash is None else cash.reindex(sub.index).fillna(0.0))

    def hold(insts):
        sub = C[insts].dropna()
        return (sub / sub.shift() - 1).mean(1).fillna(0.0)

    # 主组合
    p = strat(MAIN)
    bh = hold(MAIN)
    g = p.attrs["gross"][p.index >= START]
    t = p.attrs["turn"][p.index >= START]
    main_r = full(p)
    main_r["avg_gross"] = round(float(g.mean()), 3)
    main_r["turnover_per_year"] = round(float(t.sum() / ((t.index[-1] - t.index[0]).days / 365.25)), 2)
    bh_r = full(bh)
    # 同平均仓位的固定比例持有（事后定比例，只作参考）
    fx = bh * main_r["avg_gross"]
    fx_r = full(fx)
    pd1 = strat(MAIN, lag=2)
    cash = cash_ret(C.index)
    pc = strat(MAIN, cash=cash)
    res["main"] = main_r
    res["hold"] = bh_r
    res["fixed_mix"] = fx_r
    res["delay1"] = full(pd1)
    res["with_cash"] = full(pc)
    res["episodes"] = episodes(p[p.index >= START], bh[bh.index >= START])
    rec("main_tv0.2", {"insts": MAIN, "tv": 0.2}, {k: main_r[k] for k in ("full", "dev", "test", "avg_gross", "turnover_per_year")}, "A 股 B0 移植主口径")
    rec("hold", {"insts": MAIN}, bh_r["full"], "一直持有（各半，每日再平衡）")
    rec("fixed_mix", {"w": main_r["avg_gross"]}, fx_r["full"], "同平均仓位固定比例持有（事后定比例）")
    rec("main_delay1", {"lag": 2}, res["delay1"]["full"], "延迟 1 天")
    rec("main_cash", {"cash": "cash_idx(2016+)/2.5%"}, res["with_cash"]["full"], "闲置资金计息")

    # 目标波动敏感性
    res["tv"] = {}
    for tv in (0.15, 0.25, None):
        q = strat(MAIN, tv=tv)
        k = "none" if tv is None else str(tv)
        r_ = full(q)
        r_["avg_gross"] = round(float(q.attrs["gross"][q.index >= START].mean()), 3)
        res["tv"][k] = r_
        rec(f"tv_{k}", {"tv": tv}, {kk: r_[kk] for kk in ("full", "dev", "test", "avg_gross")}, "目标波动敏感性")

    # 单指数
    res["single"] = {}
    for a in ["H00300", "H00905", "H00016", "H00852", "399006"]:
        q, b = strat([a]), hold([a])
        first = C[a].first_valid_index()
        st0 = max(START, first + pd.Timedelta(days=400))
        q, b = q[q.index >= st0], b[b.index >= st0]
        res["single"][a] = {"name": NAMES[a], "start": str(st0.date()), "strategy": stats(q), "hold": stats(b),
                            "avg_gross": round(float(q.attrs["gross"].reindex(q.index).mean()), 3) if "gross" in q.attrs else None,
                            "test_strategy": stats(q[q.index >= DEV_END]), "test_hold": stats(b[b.index >= DEV_END])}
        rec(f"single_{a}", {"inst": a}, {k: res["single"][a][k] for k in ("strategy", "hold", "test_strategy", "test_hold")}, "单指数")

    # 单信号
    res["signals"] = {}
    for n in ENS8 + [f"don{n}" for n in DON_N]:
        q = strat(MAIN, names=[n])
        res["signals"][n] = stats(q[q.index >= START])
        rec(f"sig_{n}", {"signal": n}, res["signals"][n], "单信号，主组合，tv 0.2")

    # 停止条件
    m, h = res["main"]["full"], res["hold"]["full"]
    best_year = max(res["main"]["years"], key=lambda y: res["main"]["years"][y])
    pp = p[(p.index >= START) & (p.index.year != best_year)]
    single_bad = sum(1 for v in res["single"].values() if v["strategy"]["sharpe"] <= v["hold"]["sharpe"])
    stop = {
        "1_no_benefit_either_way": bool(m["cagr"] <= h["cagr"] - 0.03 and m["max_dd"] >= 0.6 * h["max_dd"]),
        "2_sharpe_not_higher": bool(m["sharpe"] <= h["sharpe"]),
        "3_test_cagr_le_0": bool(res["main"]["test"]["cagr"] <= 0),
        "4_ex_best_year_le_0": bool(stats(pp)["cagr"] <= 0),
        "5_delay_halves": bool(res["delay1"]["full"]["cagr"] < 0.5 * m["cagr"]),
        "6_half_singles_fail": bool(single_bad > len(res["single"]) / 2),
    }
    res["stop"] = stop
    res["ex_best_year"] = {"year": best_year, "stats": stats(pp)}
    res["single_bad"] = single_bad
    rec("verdict", {}, {"stop": stop, "ex_best": res["ex_best_year"]}, "A 股 B0 移植停止条件")
    pd.DataFrame({"main": p, "hold": bh, "delay1": pd1, "with_cash": pc}).to_parquet(OUT / "series.parquet")
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
