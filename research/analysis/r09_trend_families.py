"""OPP-007 第 2 轮：日线趋势 8 大类 50 个方向（docs/research/OPP007-round2-plan.md，看结果之前写定），只用样本内。

所有方向都输出"目标权重矩阵 W"（日期 × 品种，第 t 天收盘时已知信息算出），由同一个引擎执行：
第 t+1 天收盘按 W[t] 调仓（变化 < 0.05 不调），收益从 t+2 天起算；成交额收 0.1%；做空方向按永续计资金费。
外部数据一律只用日期 ≤ t−1 的值。开发段 2018-05～2022-12 选变体，检验段 2023-01～2026-04 只看一次。

用法：uv run python research/analysis/r09_trend_families.py
"""

import json
import math
import warnings
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from lab import lake, trials
from lab.paths import data_root

warnings.filterwarnings("ignore")
OUT = data_root() / "research" / "r09_trend"
EXPERIMENT = "r09_trend_families"
START = pd.Timestamp("2018-05-04", tz="UTC")
DEV_END = pd.Timestamp("2023-01-01", tz="UTC")
END = pd.Timestamp("2026-05-01", tz="UTC")
COST, BAND, TV = 0.001, 0.05, 0.4
MAJ = ["BTCUSDT", "ETHUSDT"]
EXCL_SUFFIX = ("UP", "DOWN", "BULL", "BEAR")
STABLE = {"USDC", "BUSD", "TUSD", "PAX", "USDP", "FDUSD", "DAI", "USDS", "USDSB", "EUR", "GBP", "AUD", "TRY", "BRL",
          "RUB", "BIDR", "IDRT", "UAH", "NGN", "ZAR", "AEUR", "EURI", "USDE", "PYUSD", "XUSD", "SUSD", "USD1", "BFUSD",
          "RLUSD", "USDB", "PAXG", "WBTC", "WBETH", "BETH", "XAUT", "BNSOL", "RWUSD", "BKRW", "UST", "USDSOLD", "U"}
RNG = np.random.default_rng(0)


# ---------------- 数据 ----------------

def load_prices() -> dict[str, pd.DataFrame]:
    df = lake.scan("bn_spot_1d").select("ts", "inst_id", "high", "low", "close", "quote_volume").collect().to_pandas()
    df = df[(df.ts < END)]
    df = df[~df.inst_id.str[:-4].isin(STABLE)]  # 杠杆代币入库时已剔除（按"去后缀后是已上架币"判定，避免误删 JUP、SYRUP）
    out = {k: df.pivot_table(index="ts", columns="inst_id", values=k) for k in ("high", "low", "close", "quote_volume")}
    idx = pd.date_range(df.ts.min(), END - pd.Timedelta(days=1), freq="D", tz="UTC")
    return {k: v.reindex(idx) for k, v in out.items()}


def load_funding(cols) -> pd.DataFrame:
    f = lake.scan("bn_funding").collect().to_pandas()
    f["ts"] = f.ts.dt.floor("D")
    f = f.groupby(["ts", "inst_id"]).funding_rate.sum().unstack()
    return f.reindex(columns=cols)


def ext(dataset: str, series: str) -> pd.Series | None:
    """外部日度序列，按日期对齐后晚 1 天使用（返回值在第 t 天给出的是 ≤ t−1 的最新值）。"""
    try:
        d = lake.scan(dataset).filter(pl.col("series") == series).select("ts", "value").collect().to_pandas()
    except Exception:  # noqa: BLE001
        return None
    if d.empty:
        return None
    s = d.groupby(d.ts.dt.floor("D")).value.last().sort_index()
    idx = pd.date_range(s.index.min(), END - pd.Timedelta(days=1), freq="D", tz="UTC")
    return s.reindex(idx).ffill().shift(1)


# ---------------- 引擎 ----------------

def execute(W: pd.DataFrame, R: pd.DataFrame, fund: pd.DataFrame | None = None, extra_short=0.0, lag=1, rel_band=False) -> pd.Series:
    """W[t] 是第 t 天收盘的目标权重；第 t+lag 天收盘调仓；第 t+lag+1 天开始承担收益。"""
    cols = W.columns
    w = W.reindex(R.index).fillna(0.0).to_numpy()
    r = R[cols].to_numpy()
    alive = ~np.isnan(r)
    rr = np.nan_to_num(r)
    f = fund[cols].reindex(R.index).to_numpy() if fund is not None else None
    T, N = w.shape
    pos = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(T):
        k = t - lag
        if k >= 0:
            tgt = w[k] * alive[t]  # 当天没有价格（下架后）就清仓
            th = np.minimum(BAND, 0.25 * np.maximum(np.abs(tgt), np.abs(cur))) if rel_band else BAND  # 小权重组合用相对阈值
            ch = (np.abs(tgt - cur) >= th) | ((tgt == 0) & (cur != 0)) | ~alive[t]
            cur = np.where(ch, tgt, cur)
        pos[t] = cur
    held = np.vstack([np.zeros(N), pos[:-1]])
    trade = np.abs(np.diff(np.vstack([np.zeros(N), pos]), axis=0))
    short_trade = np.abs(np.diff(np.vstack([np.zeros(N), np.clip(-pos, 0, None)]), axis=0))
    pnl = (held * rr).sum(1) - trade.sum(1) * COST - short_trade.sum(1) * extra_short  # 做空另加的每笔成本
    if f is not None:
        pnl -= np.nansum(held * np.where(np.isnan(f), 0.0, f), axis=1)
    return pd.Series(pnl, index=R.index)


def vol(R: pd.DataFrame, n=60) -> pd.DataFrame:
    return np.log1p(R).rolling(n, min_periods=n // 2).std() * math.sqrt(365)


def size(S: pd.DataFrame, R: pd.DataFrame, share: float, cap=1.0, tv=TV, lo=0.0) -> pd.DataFrame:
    """强度 S（−1～1）→ 权重：share × clip(S × tv / 波动, lo×cap, cap)。"""
    v = vol(R)[S.columns]
    return share * (S * tv / v).clip(lo * cap, cap).fillna(0.0)


def stats(p: pd.Series) -> dict:
    p = p.dropna()
    if len(p) < 30:
        return {}
    eq = (1 + p).cumprod()
    yrs = len(p) / 365
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    sd = p.std() * math.sqrt(365)
    dd = float((1 - eq / eq.cummax()).max())
    return {"cagr": round(float(cagr), 4), "sharpe": round(float(p.mean() * 365 / sd), 3) if sd > 0 else 0.0,
            "max_dd": round(dd, 4), "calmar": round(float(cagr / dd), 3) if dd > 0 else None, "vol": round(float(sd), 4)}


def boot_p(a: pd.Series, b: pd.Series, block=20, n=5000) -> float:
    """检验段夏普差（a − b）的分块重抽样单侧 p 值。"""
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    T = len(x)
    nb = T // block
    starts = RNG.integers(0, T - block, (n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)
    xa, xb = x[idx, 0], x[idx, 1]
    sh = lambda z: z.mean(1) / z.std(1)  # noqa: E731
    return float(((sh(xa) - sh(xb)) <= 0).mean())


def deflated_sharpe(sr_best: float, srs: list[float], T: int, skew: float, kurt: float) -> float:
    """Bailey & López de Prado：按试验次数折算后的夏普显著性（日频夏普）。"""
    n = len(srs)
    v = np.var(srs) if n > 1 else 0.0
    emc = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * math.e))) if n > 1 else 0.0
    z = (sr_best - sr0) * math.sqrt(T - 1) / math.sqrt(1 - skew * sr_best + (kurt - 1) / 4 * sr_best ** 2)
    return float(norm.cdf(z))


# ---------------- 信号工具 ----------------

def ens8(C: pd.DataFrame) -> pd.DataFrame:
    parts = [np.sign(C - C.rolling(n).mean()) for n in (20, 50, 100, 200)] + [np.sign(C / C.shift(n) - 1) for n in (20, 60, 120, 250)]
    return sum(p.clip(lower=0) for p in parts) / 8


def atr(H, L, C, n=14):
    tr = np.maximum(np.maximum(H - L, (H - C.shift()).abs()), (L - C.shift()).abs())
    return tr.rolling(n).mean()


def state_machine(enter: pd.DataFrame, exit_: pd.DataFrame) -> pd.DataFrame:
    e, x = enter.fillna(False).to_numpy(), exit_.fillna(False).to_numpy()
    out = np.zeros(e.shape)
    st = np.zeros(e.shape[1])
    for t in range(len(e)):
        st = np.where(e[t], 1.0, np.where(x[t], 0.0, st))
        out[t] = st
    return pd.DataFrame(out, index=enter.index, columns=enter.columns)


def adx(H, L, C, n=14):
    up, dn = H.diff(), -L.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(H, L, C, n)
    pdi, mdi = 100 * pdm.rolling(n).mean() / a, 100 * mdm.rolling(n).mean() / a
    return (100 * (pdi - mdi).abs() / (pdi + mdi)).rolling(n).mean()


# ---------------- 主流程 ----------------

class Lab:
    def __init__(self):
        P = load_prices()
        self.C, self.H, self.L, self.Q = P["close"], P["high"], P["low"], P["quote_volume"]
        self.R = self.C / self.C.shift() - 1
        self.F = load_funding(self.C.columns)
        self.res, self.all_sr = {}, []
        c, r = self.C[MAJ], self.R[MAJ]
        self.B0W = size(ens8(c), r, 0.5)
        self.B0 = execute(self.B0W, r)
        self.B0d = execute(self.B0W, r, lag=2)
        self.bh = r.mean(1)

    def ev(self, code: str, name: str, variants: dict, cols=None, perp=False, extra_short=0.0, unavailable=None, rel_band=False):
        """variants: {变体名: W}。开发段夏普最高的变体进入检验段。"""
        if unavailable:
            self.res[code] = {"name": name, "status": "数据不可得", "reason": unavailable}
            print(code, "数据不可得", unavailable, flush=True)
            return None
        R = self.R
        fund = self.F if perp else None
        best, best_sr, rows = None, -9, {}
        for vn, W in variants.items():
            p = execute(W, R[W.columns], fund, extra_short, rel_band=rel_band)
            p = p[(p.index >= START)]
            dev = stats(p[p.index < DEV_END])
            rows[vn] = {"dev": dev, "p": p, "W": W}
            self.all_sr.append(dev.get("sharpe", 0))
            trials.record(EXPERIMENT, variant=f"{code}_{vn}_dev", sample="in_sample", params={"code": code, "variant": vn},
                          metrics=dev, note="OPP-007 第 2 轮开发段")
            if dev.get("sharpe", -9) > best_sr:
                best, best_sr = vn, dev["sharpe"]
        p, W = rows[best]["p"], rows[best]["W"]
        pd_ = execute(W, R[W.columns], fund, extra_short, lag=2, rel_band=rel_band)
        pd_ = pd_[pd_.index >= START]
        b0 = self.B0[self.B0.index >= START]
        te, b0te = p[p.index >= DEV_END], b0[b0.index >= DEV_END]
        r = {
            "name": name, "status": "ok", "chosen": best, "variants": {k: v["dev"] for k, v in rows.items()},
            "full": stats(p), "dev": rows[best]["dev"], "test": stats(te), "test_delay": stats(pd_[pd_.index >= DEV_END]),
            "corr_b0": round(float(p.corr(b0)), 3), "p_vs_b0": boot_p(te, b0te),
            "avg_gross": round(float(W.abs().sum(1)[W.index >= START].mean()), 3),
            "years": {int(y): round(float((1 + g).prod() - 1), 4) for y, g in p.groupby(p.index.year)},
        }
        self.res[code] = r
        self.series = getattr(self, "series", {})
        self.series[code] = p
        trials.record(EXPERIMENT, variant=f"{code}_test", sample="in_sample", params={"code": code, "chosen": best},
                      metrics={k: r[k] for k in ("full", "dev", "test", "test_delay", "corr_b0", "p_vs_b0")},
                      note="OPP-007 第 2 轮检验段（冻结后一次性，样本内，封存段未动）")
        print(code, name, best, "dev", r["dev"].get("sharpe"), "test", r["test"], "p", round(r["p_vs_b0"], 3), flush=True)
        return r

    # ---- A 指标变体 ----
    def family_a(self):
        C, H, L, R = self.C[MAJ], self.H[MAJ], self.L[MAJ], self.R[MAJ]
        S = lambda s: size(s, R, 0.5)  # noqa: E731
        ema = lambda n: C.ewm(span=n, adjust=False).mean()  # noqa: E731
        self.ev("A1", "EMA 交叉组合", {"3pairs": S(sum((ema(a) > ema(b)).astype(float) for a, b in ((10, 50), (20, 100), (50, 200))) / 3)})
        macd = ema(12) - ema(26)
        self.ev("A2", "MACD", {"hist": S((macd - macd.ewm(span=9, adjust=False).mean() > 0).astype(float)),
                               "line": S((macd > 0).astype(float))})
        ax, e8 = adx(H, L, C), ens8(C)
        self.ev("A3", "ADX 过滤", {f"adx{k}": S(e8 * (ax > k)) for k in (15, 20, 25)})
        er = (C - C.shift(30)).abs() / C.diff().abs().rolling(30).sum()
        self.ev("A4", "效率系数加权", {"er30": S(e8 * er.clip(0, 1) * 2)})
        mid, sd = C.rolling(20).mean(), C.rolling(20).std()
        self.ev("A5", "布林带突破", {"bb20_2": S(state_machine(C > mid + 2 * sd, C < mid)),
                                     "bb20_1.5": S(state_machine(C > mid + 1.5 * sd, C < mid))})
        a14 = atr(H, L, C, 10)
        def supertrend(m):
            hl = (H + L) / 2
            ub, lb = (hl + m * a14).to_numpy(), (hl - m * a14).to_numpy()
            c = C.to_numpy()
            st = np.zeros(c.shape)
            fu, fl, d = ub[0].copy(), lb[0].copy(), np.ones(c.shape[1])
            for t in range(1, len(c)):
                fu = np.where(np.isnan(fu) | (ub[t] < fu) | (c[t - 1] > fu), ub[t], fu)
                fl = np.where(np.isnan(fl) | (lb[t] > fl) | (c[t - 1] < fl), lb[t], fl)
                d = np.where(c[t] > fu, 1, np.where(c[t] < fl, -1, d))
                st[t] = d > 0
            return pd.DataFrame(st, index=C.index, columns=C.columns)
        self.ev("A6", "ATR 跟踪止损（Supertrend）", {f"m{m}": S(supertrend(m)) for m in (2, 3, 4)})
        a20 = atr(H, L, C, 20)
        turtle = (state_machine(C > H.rolling(20).max().shift(), C < L.rolling(10).min().shift())
                  + state_machine(C > H.rolling(55).max().shift(), C < L.rolling(20).min().shift())) / 2
        tw = 0.5 * (turtle * 0.02 / (a20 / C)).clip(0, 1).fillna(0)  # 每份资金按 2% ATR 风险定仓
        self.ev("A7", "海龟 20/55 + ATR 定仓", {"turtle": tw})
        wk = C.resample("W-SUN").last()
        wtr = (wk > wk.rolling(10).mean()).astype(float).reindex(C.index, method="ffill").shift(1)
        self.ev("A8", "周线 + 日线确认", {"w10": S(e8 * wtr)})
        lc = np.log(C)
        x = np.arange(60) - 29.5
        slope = lc.rolling(60).apply(lambda y: np.polyfit(x, y, 1)[0], raw=True)
        resid = lc.rolling(60).std()
        self.ev("A9", "回归斜率 t 值", {"t60": S(np.tanh(slope * 60 / resid / 2).clip(lower=0))})
        v = vol(R)
        tsm = sum(np.tanh((C / C.shift(n) - 1) / (v * math.sqrt(n / 365))) for n in (30, 90, 180)) / 3
        self.ev("A10", "连续动量", {"tanh": S(tsm.clip(lower=0))})
        conv, base = (H.rolling(9).max() + L.rolling(9).min()) / 2, (H.rolling(26).max() + L.rolling(26).min()) / 2
        sa, sb = ((conv + base) / 2).shift(26), ((H.rolling(52).max() + L.rolling(52).min()) / 2).shift(26)
        self.ev("A11", "一目均衡表", {"cloud": S((C > np.maximum(sa, sb)).astype(float))})
        obv = (np.sign(R) * self.Q[MAJ]).cumsum()
        self.ev("A12", "量价确认（OBV）", {"obv50": S(e8 * (obv > obv.rolling(50).mean()))})
        self.ev("A13", "波动突破", {f"k{k}": S(state_machine(C > C.shift() + k * atr(H, L, C), C < C.rolling(20).mean()))
                                    for k in (0.5, 1.0, 1.5)})

    # ---- B 风险叠加 ----
    def overlay(self, m: pd.Series | pd.DataFrame, boost=False) -> pd.DataFrame:
        W = self.B0W.mul(m, axis=0) if isinstance(m, pd.Series) else self.B0W * m
        return W.clip(0, 0.5)

    def family_b(self):
        C, R = self.C[MAJ], self.R[MAJ]
        b0 = self.B0
        def ddctl(th):
            eq = (1 + b0.fillna(0)).cumprod()
            dd = 1 - eq / eq.cummax()
            return self.overlay(np.where(dd > th, 0.5, 1.0) * pd.Series(1.0, index=dd.index))
        self.ev("B1", "回撤控制", {f"dd{int(t*100)}": ddctl(t) for t in (0.15, 0.2, 0.25)})
        v20 = vol(R, 20).mean(1)
        q90 = v20[(v20.index >= START) & (v20.index < DEV_END)].quantile(0.9)
        self.ev("B2", "高波动减仓", {"q90": self.overlay((v20 > q90).map({True: 0.5, False: 1.0}))})
        crash = (R.min(1) < -0.10).astype(float).rolling(5).max().fillna(0)
        self.ev("B3", "暴跌冷却 5 天", {"cool5": self.overlay(1 - crash)})
        iv = 1 / vol(R)
        wts = iv.div(iv.sum(1), axis=0)
        e8 = ens8(C)
        self.ev("B4", "按波动倒数分配", {"invvol": (wts * (e8 * TV / vol(R)).clip(0, 1)).fillna(0)})
        mom = C / C.shift(60) - 1
        pick = mom.eq(mom.max(1), axis=0).astype(float)
        self.ev("B5", "二选一轮动", {"top1": (pick * (e8 * TV / vol(R)).clip(0, 1)).fillna(0)})

    # ---- C 截面 ----
    def universe(self, n=30, hist=120):
        q30 = self.Q.rolling(30, min_periods=20).sum()
        age = self.C.notna().cumsum()
        ok = (age >= hist) & self.C.notna()
        rank = q30.where(ok).rank(axis=1, ascending=False)
        return rank <= n

    def weekly(self, W: pd.DataFrame) -> pd.DataFrame:
        W = W.copy()
        W.loc[W.index.dayofweek != 0, :] = np.nan  # 每周一收盘调仓，其余日子沿用
        return W.ffill().fillna(0)

    def topk(self, score: pd.DataFrame, k: int, uni: pd.DataFrame, gate: pd.Series | None = None, short=False):
        s = score.where(uni)
        hi = s.rank(axis=1, ascending=False) <= k
        W = hi.astype(float).div(hi.sum(1).replace(0, np.nan), axis=0).fillna(0)
        if short:
            lo = s.rank(axis=1, ascending=True) <= k
            W = 0.5 * W - 0.5 * lo.astype(float).div(lo.sum(1).replace(0, np.nan), axis=0).fillna(0)
        if gate is not None:
            W = W.mul(gate, axis=0)
        return self.weekly(W)

    def family_c(self):
        C, R = self.C, self.R
        uni = self.universe()
        btc = C["BTCUSDT"]
        gate = (btc > btc.rolling(200).mean()).astype(float)
        v = vol(R)
        self.ev("C1", "截面动量 28 日前 5", {"k5": self.topk(C / C.shift(28) - 1, 5, uni, gate)})
        self.ev("C2", "截面动量 90 日风险调整前 10", {"k10": self.topk((C / C.shift(90) - 1) / v, 10, uni, gate)})
        e = ens8(C).where(uni)
        raw = (e * TV / v).clip(0, 1)
        n = (e > 0).sum(1).replace(0, np.nan)
        W3 = raw.div(np.maximum(n, 5), axis=0).fillna(0)  # 至少按 5 份分，单币不超过 20%
        self.ev("C3", "逐币趋势等风险", {"ens8": W3}, rel_band=True)
        self.ev("C4", "多空截面动量（永续）", {"ls5": self.topk(C / C.shift(28) - 1, 5, uni, None, short=True)},
                perp=True, extra_short=0.001)
        self.ev("C5", "短期反转（上周跌最多）", {"rev5": self.topk(-(C / C.shift(7) - 1), 5, uni, gate)})
        lowv = (-v).where(uni & (e > 0))
        self.ev("C6", "低波动 + 趋势", {"k10": self.topk(lowv, 10, uni & (e > 0), gate)})
        self.uni, self.gate = uni, gate

    # ---- D 机器学习 ----
    def single_feats(self, extra: dict[str, pd.Series]) -> pd.DataFrame:
        rows = []
        for a in MAJ:
            c, r, q = self.C[a], self.R[a], self.Q[a]
            f = pd.DataFrame({f"ret{n}": c / c.shift(n) - 1 for n in (7, 14, 30, 60, 120, 250)})
            f["vol20"], f["vol60"] = vol(self.R[[a]], 20)[a], vol(self.R[[a]], 60)[a]
            for n in (20, 50, 200):
                f[f"dma{n}"] = c / c.rolling(n).mean() - 1
            f["qz"] = np.log(q / q.rolling(90).mean())
            f["ens8"] = ens8(self.C[[a]])[a]
            f["is_eth"] = float(a == "ETHUSDT")
            for k, s in extra.items():
                f[k] = s.reindex(f.index) if not callable(s) else s(a).reindex(f.index)
            f["y"] = (c.shift(-8) / c.shift(-1) - 1 > 0).astype(float).where(c.shift(-8).notna())
            f["asset"] = a
            rows.append(f)
        return pd.concat(rows)

    def walk_forward(self, X: pd.DataFrame, feats: list[str], model: str, target="y", reg=False) -> pd.Series:
        """每季度扩展窗口重训：只用标签在季度起点前已结束的样本（标签 8 天，隔 9 天）。"""
        out = pd.Series(np.nan, index=X.index)
        qs = pd.date_range(pd.Timestamp("2019-07-01", tz="UTC"), END, freq="QS")
        idx = X.index.get_level_values(0) if isinstance(X.index, pd.MultiIndex) else X.index
        for q0, q1 in zip(qs[:-1], qs[1:]):
            tr = X[(idx < q0 - pd.Timedelta(days=9)) & X[target].notna() & (idx >= START - pd.Timedelta(days=200))]
            te_m = (idx >= q0) & (idx < q1)
            if len(tr) < 200 or te_m.sum() == 0:
                continue
            Xt, yt = tr[feats].to_numpy(float), tr[target].to_numpy(float)
            Xp = X.loc[te_m, feats].to_numpy(float)
            Xt[~np.isfinite(Xt)], Xp[~np.isfinite(Xp)] = np.nan, np.nan
            if model == "lgb":
                m = (lgb.LGBMRegressor if reg else lgb.LGBMClassifier)(
                    n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=50, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbose=-1)
                m.fit(np.nan_to_num(Xt, nan=np.nan), yt)
                pr = m.predict(Xp) if reg else m.predict_proba(Xp)[:, 1]
            else:
                med = np.nan_to_num(np.nanmedian(Xt, axis=0), nan=0.0)  # 整列缺失（如 2020 年前的资金费）补 0
                Xt2, Xp2 = np.where(np.isnan(Xt), med, Xt), np.where(np.isnan(Xp), med, Xp)
                sc = StandardScaler().fit(Xt2)
                if reg:
                    pr = Ridge(alpha=10.0).fit(sc.transform(Xt2), yt).predict(sc.transform(Xp2))
                else:
                    pr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(Xt2), yt).predict_proba(sc.transform(Xp2))[:, 1]
            out[te_m] = pr
        return out

    def ml_single(self, extra) -> dict[str, pd.DataFrame]:
        X = self.single_feats(extra)
        feats = [c for c in X.columns if c not in ("y", "asset")]
        out = {}
        for model in ("lgb", "lr"):
            p = self.walk_forward(X, feats, model)
            P = pd.DataFrame({a: p[X.asset == a] for a in MAJ})
            s = (2 * P - 1).clip(0, 1)
            # 第一个模型（2019-07）之前用 B0 的强度，所有机器学习方向这一段相同
            s = s.where(P.notna(), ens8(self.C[MAJ]))
            out[model] = size(s, self.R[MAJ], 0.5)
        return out

    def family_d(self, ext_feats: dict, deriv: dict):
        self.ev("D1", "单资产机器学习（价格）", self.ml_single({}))
        self.ev("D2", "单资产机器学习（+衍生品）", self.ml_single(deriv))
        self.ev("D3", "单资产机器学习（+外部数据）", self.ml_single({**deriv, **ext_feats}),
                unavailable=None if ext_feats else "外部数据全部不可得")
        # 面板
        C, R, Q, v = self.C, self.R, self.Q, vol(self.R)
        fr7 = self.F.rolling(7, min_periods=1).mean()
        feats = {f"ret{n}": C / C.shift(n) - 1 for n in (7, 14, 28, 90)}
        feats.update({"vol30": vol(R, 30), "madj90": (C / C.shift(90) - 1) / v, "dma200": C / C.rolling(200).mean() - 1,
                      "qz": np.log(Q / Q.rolling(90).mean()), "fund7": fr7.reindex(C.index)})
        fwd = C.shift(-8) / C.shift(-1) - 1
        long = pd.concat({k: x.where(self.uni).stack(future_stack=True) for k, x in feats.items()}, axis=1)
        long["btc28"] = (C["BTCUSDT"] / C["BTCUSDT"].shift(28) - 1).reindex(long.index.get_level_values(0)).to_numpy()
        long["y"] = fwd.where(self.uni).rank(axis=1, pct=True).stack(future_stack=True).reindex(long.index)
        long = long[long.index.get_level_values(0) >= START - pd.Timedelta(days=200)]
        long = long[self.uni.stack(future_stack=True).reindex(long.index).fillna(False)]
        fs = [c for c in long.columns if c != "y"]
        for code, model in (("D4", "lgb"), ("D5", "ridge")):
            pr = self.walk_forward(long, fs, model, reg=True)
            score = pr.unstack().reindex(index=C.index, columns=C.columns)
            self.ev(code, f"面板排序（{'LightGBM' if model == 'lgb' else '岭回归'}）", {"k10": self.topk(score, 10, self.uni, self.gate)})
        # D6 市场状态（高斯混合，开发段拟合）
        btc = C["BTCUSDT"]
        Z = pd.DataFrame({"r20": btc / btc.shift(20) - 1, "v20": vol(R[["BTCUSDT"]], 20)["BTCUSDT"]})
        devm = (Z.index >= START) & (Z.index < DEV_END) & Z.notna().all(1)
        sc = StandardScaler().fit(Z[devm])
        gm = GaussianMixture(3, random_state=0).fit(sc.transform(Z[devm]))
        lab = pd.Series(gm.predict(sc.transform(Z[devm])), index=Z.index[devm])
        f7 = (btc.shift(-8) / btc.shift(-1) - 1)[devm]
        expo = np.array([1.0 if f7[lab == k].mean() > 0 else 0.0 for k in range(3)])
        ok = Z.notna().all(1)
        prob = pd.DataFrame(gm.predict_proba(sc.transform(Z[ok].fillna(0))), index=Z.index[ok]).reindex(Z.index).fillna(0)
        m = pd.Series(prob.to_numpy() @ expo, index=Z.index)
        self.ev("D6", "市场状态模型（高斯混合）", {"gmm3": size(pd.concat({a: m for a in MAJ}, axis=1), R[MAJ], 0.5)})
        # D7 元标签：预测 B0 未来 7 日盈亏
        X = self.single_feats({})
        b0w = self.B0W
        rows = []
        for a in MAJ:
            ra = self.R[a]
            pnl7 = sum(ra.shift(-k) for k in range(2, 9))
            rows.append((pnl7 > 0).astype(float).where(pnl7.notna() & (b0w[a] > 0)))
        X["y"] = pd.concat(rows).to_numpy()
        feats = [c for c in X.columns if c not in ("y", "asset")]
        out = {}
        for model in ("lgb", "lr"):
            p = self.walk_forward(X, feats, model)
            P = pd.DataFrame({a: p[X.asset == a] for a in MAJ})
            mult = (P / 0.5).clip(0, 1.5).fillna(1.0)
            out[model] = (b0w * mult).clip(0, 0.5)
        self.ev("D7", "元标签缩放 B0", out)

    # ---- E/F/G 叠加 ----
    def filt(self, code, name, m, unavailable=None, variants=None):
        if unavailable:
            return self.ev(code, name, {}, unavailable=unavailable)
        v = variants or {"main": m}
        return self.ev(code, name, {k: self.overlay(x.reindex(self.B0W.index).fillna(1.0)) for k, x in v.items()})

    def family_efg(self):
        fng = ext("ext_sentiment", "fng")
        if fng is None:
            for c in ("E1", "E2"):
                self.filt(c, "恐惧贪婪", None, "fng 不可得")
        else:
            self.filt("E1", "恐惧贪婪反向", np.where(fng < 20, 1.5, np.where(fng > 80, 0.5, 1.0)) * pd.Series(1.0, fng.index))
            ma = fng.rolling(30).mean()
            self.filt("E2", "恐惧贪婪趋势", (ma > ma.shift(5)).map({True: 1.0, False: 0.5}))
        wiki = ext("ext_sentiment", "wiki_bitcoin")
        if wiki is None:
            self.filt("E3", "注意力过热", None, "维基浏览量不可得")
        else:
            z = (np.log(wiki) - np.log(wiki).rolling(90).mean()) / np.log(wiki).rolling(90).std()
            self.filt("E3", "注意力过热（维基）", (z > 2).map({True: 0.5, False: 1.0}))
        tone, nvol = ext("ext_news", "gdelt_bitcoin_tone"), ext("ext_news", "gdelt_bitcoin_vol")
        if tone is None:
            self.filt("E4", "新闻情绪动量", None, "GDELT tone 不可得")
        else:
            self.filt("E4", "新闻情绪动量（GDELT）", (tone.rolling(30).mean() > tone.rolling(90).mean()).map({True: 1.0, False: 0.5}))
        if tone is None or nvol is None:
            self.filt("E5", "负面新闻冲击", None, "GDELT 新闻量或 tone 不可得")
        else:
            z = (nvol - nvol.rolling(90).mean()) / nvol.rolling(90).std()
            shock = ((z > 2) & (tone < 0)).astype(float).rolling(5).max()
            self.filt("E5", "负面新闻冲击（GDELT）", 1 - shock)
        # F 宏观
        spx, vix, dxy, y10 = (ext("ext_macro", s) for s in ("sp500", "vix", "usd_broad", "us10y"))
        stab = ext("ext_stable", "stable_total_usd")
        conds = {}
        if spx is not None:
            conds["F1"] = spx > spx.rolling(200).mean()
        if vix is not None:
            conds["F2"] = vix <= 30
        if dxy is not None:
            conds["F3"] = dxy / dxy.shift(60) - 1 < 0
        if y10 is not None:
            conds["F4"] = y10 - y10.shift(60) < 0
        if stab is not None:
            conds["F5"] = stab / stab.shift(30) - 1 > 0
        names = {"F1": "美股趋势", "F2": "VIX", "F3": "美元趋势", "F4": "美债利率", "F5": "稳定币流动性"}
        for c, nm in names.items():
            if c in conds:
                self.filt(c, nm, conds[c].map({True: 1.0, False: 0.5}))
            else:
                self.filt(c, nm, None, f"{nm} 数据不可得")
        if conds:
            self.filt("F6", "宏观综合", sum(x.astype(float) for x in conds.values()) / len(conds))
        else:
            self.filt("F6", "宏观综合", None, "宏观数据不可得")
        # G 链上与衍生品
        hr = ext("ext_onchain", "bc_hash_rate")
        if hr is None:
            self.filt("G1", "哈希带", None, "算力不可得")
        else:
            cap = (hr.rolling(30).mean() < hr.rolling(60).mean()).map({True: 0.5, False: 1.0})
            self.filt("G1", "哈希带", cap)
        mv = ext("ext_onchain", "cm_btc_CapMVRVCur")
        if mv is None:
            self.filt("G2", "MVRV", None, "MVRV 不可得")
        else:
            self.filt("G2", "MVRV", pd.Series(np.where(mv > 3, 0.5, np.where(mv < 1, 1.5, 1.0)), index=mv.index))
        aa = ext("ext_onchain", "cm_btc_AdrActCnt")
        if aa is None:
            aa = ext("ext_onchain", "bc_n_unique_addresses")
        if aa is None:
            self.filt("G3", "活跃地址动量", None, "活跃地址不可得")
        else:
            self.filt("G3", "活跃地址动量", (aa.rolling(30).mean() > aa.rolling(90).mean()).map({True: 1.0, False: 0.5}))
        f7 = self.F[MAJ].rolling(7, min_periods=1).mean() / 3  # 每天三次 → 每 8 小时平均
        self.filt("G4", "资金费拥挤", None, variants={"0.05%": (f7 <= 0.0005).astype(float).replace(0, 0.5).reindex(self.B0W.index).fillna(1.0)})
        cb = ext("ext_coinbase", "cb_btc_close")
        if cb is None:
            self.filt("G5", "Coinbase 溢价", None, "Coinbase 价格不可得")
        else:
            prem = (cb / self.C["BTCUSDT"].shift(1) - 1).rolling(7).mean()
            self.filt("G5", "Coinbase 溢价", (prem > 0).map({True: 1.0, False: 0.5}))
        tv = ext("ext_onchain", "cm_btc_TxTfrValAdjUSD")
        mc = ext("ext_onchain", "cm_btc_CapMrktCurUSD")
        if tv is None or mc is None:
            self.filt("G6", "NVT", None, "链上转账额或市值不可得")
        else:
            nvt = mc / tv.rolling(28).mean()
            pct = nvt.rolling(365, min_periods=180).rank(pct=True)
            self.filt("G6", "NVT", (pct > 0.9).map({True: 0.5, False: 1.0}))
        self.ext_feats = {k: s for k, s in {"fng": fng, "wiki": None if wiki is None else np.log(wiki).diff(7),
                                            "tone30": None if tone is None else tone.rolling(30).mean(),
                                            "spx_dma": None if spx is None else spx / spx.rolling(200).mean() - 1,
                                            "vix": vix, "dxy60": None if dxy is None else dxy / dxy.shift(60) - 1,
                                            "y10_60": None if y10 is None else y10 - y10.shift(60),
                                            "stab30": None if stab is None else stab / stab.shift(30) - 1,
                                            "mvrv": mv}.items() if s is not None}

    # ---- H 组合 ----
    def family_h(self):
        R, C = self.R, self.C
        best = {}
        for code, r in self.res.items():
            if r.get("status") == "ok" and code not in ("B0", "BH"):
                fam = code[0]
                if fam not in best or r["dev"]["sharpe"] > self.res[best[fam]]["dev"]["sharpe"]:
                    best[fam] = code
        s = pd.concat({c: self.series[c] for c in best.values()}, axis=1)
        self.ev_series("H1", f"各大类开发段最优等权（{','.join(best.values())}）", s.mean(1))
        self.ev_series("H2", "B0 + C3 各半", (self.B0 + self.series.get("C3", self.B0)) / 2)
        f = self.F[MAJ].reindex(R.index)
        idle = (0.5 - self.B0W).clip(lower=0) * 0.75  # 现货多 1 + 永续空保证金约 1/3，套利名义 ≈ 闲置资金的 75%
        carry = (idle.shift(2) * f.fillna(0)).sum(1) - (idle.diff().abs().sum(1).shift(1) * 2 * COST).fillna(0)
        carry = carry.where(f.notna().any(axis=1), 0.0)  # 没有资金费数据的日子不做
        self.ev_series("H3", "B0 + 空仓资金做期现套利", self.B0 + carry)
        self.ev_series("H4", "B0 + 闲置资金 4% 年息", self.B0 + (1 - self.B0W.sum(1).shift(2).fillna(0)) * 0.04 / 365)
        good = [c for c in self.res if c[0] in "EFG" and self.res[c].get("status") == "ok"
                and self.res[c]["dev"]["sharpe"] > self.res["B0"]["dev"]["sharpe"]]
        if good:
            self.ev_series("H5", f"过滤器开发段有改进者等权（{','.join(good)}）", pd.concat({c: self.series[c] for c in good}, axis=1).mean(1))
        else:
            self.res["H5"] = {"name": "过滤器投票", "status": "无开发段改进的过滤器，未构建"}
        self.ev("H6", "B0 加倍（目标波动 60%，上限 2 倍）", {"tv60": size(ens8(C[MAJ]), R[MAJ], 0.5, cap=2, tv=0.6)})
        self.ev_series("H7", "B0 + C2 各半", (self.B0 + self.series.get("C2", self.B0)) / 2)
        ok = [c for c, r in self.res.items() if r.get("status") == "ok" and c[0] != "H" and c not in ("B0", "BH")]
        self.ev_series("H8", f"全部方向等权（{len(ok)} 个）", pd.concat({c: self.series[c] for c in ok}, axis=1).mean(1))

    def ev_series(self, code, name, p: pd.Series):
        """组合方向：直接给日收益（成分已扣成本），延迟口径用同样组合的 lag=2 近似为 B0 延迟差。"""
        p = p[p.index >= START]
        b0 = self.B0[self.B0.index >= START]
        te = p[p.index >= DEV_END]
        r = {"name": name, "status": "ok", "chosen": "—", "full": stats(p), "dev": stats(p[p.index < DEV_END]),
             "test": stats(te), "test_delay": None, "corr_b0": round(float(p.corr(b0)), 3),
             "p_vs_b0": boot_p(te, b0[b0.index >= DEV_END]),
             "years": {int(y): round(float((1 + g).prod() - 1), 4) for y, g in p.groupby(p.index.year)}}
        self.res[code] = r
        self.series[code] = p
        self.all_sr.append(r["dev"].get("sharpe", 0))
        trials.record(EXPERIMENT, variant=f"{code}_test", sample="in_sample", params={"code": code},
                      metrics={k: r[k] for k in ("full", "dev", "test", "corr_b0", "p_vs_b0")}, note="OPP-007 第 2 轮组合方向")
        print(code, name, "dev", r["dev"].get("sharpe"), "test", r["test"], "p", round(r["p_vs_b0"], 3), flush=True)


def holm(ps: dict[str, float]) -> dict[str, float]:
    items = sorted(ps.items(), key=lambda x: x[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lab_ = Lab()
    b0 = lab_.B0[lab_.B0.index >= START]
    lab_.series = {"B0": b0}
    lab_.res["B0"] = {"name": "第 1 轮主策略（现货执行）", "status": "ok", "full": stats(b0), "dev": stats(b0[b0.index < DEV_END]),
                      "test": stats(b0[b0.index >= DEV_END]),
                      "test_delay": stats(lab_.B0d[lab_.B0d.index >= DEV_END]),
                      "years": {int(y): round(float((1 + g).prod() - 1), 4) for y, g in b0.groupby(b0.index.year)}}
    bh = lab_.bh[lab_.bh.index >= START]
    lab_.res["BH"] = {"name": "一直持有 BTC/ETH 各半", "status": "ok", "full": stats(bh), "dev": stats(bh[bh.index < DEV_END]),
                      "test": stats(bh[bh.index >= DEV_END])}
    trials.record(EXPERIMENT, variant="B0_BH", sample="in_sample", params={}, metrics={"B0": lab_.res["B0"], "BH": lab_.res["BH"]},
                  note="OPP-007 第 2 轮基准")
    print("B0", lab_.res["B0"]["dev"], lab_.res["B0"]["test"], flush=True)
    lab_.family_a()
    lab_.family_b()
    lab_.family_c()
    lab_.family_efg()
    deriv = {"fund7": lambda a: lab_.F[a].rolling(7, min_periods=1).mean() if a in lab_.F else pd.Series(dtype=float),
             "fund30": lambda a: lab_.F[a].rolling(30, min_periods=1).mean() if a in lab_.F else pd.Series(dtype=float)}
    lab_.family_d(lab_.ext_feats, deriv)
    lab_.family_h()

    # 判定
    b0t, b0d = lab_.res["B0"]["test"]["sharpe"], lab_.res["B0"]["dev"]["sharpe"]
    ok = {c: r for c, r in lab_.res.items() if r.get("status") == "ok" and c not in ("B0", "BH")}
    adj = holm({c: r["p_vs_b0"] for c, r in ok.items()})
    for c, r in ok.items():
        r["p_holm"] = round(adj[c], 4)
        delay_ok = r["test_delay"] is None or r["test_delay"].get("sharpe", -9) > b0t
        r["beats_b0"] = bool(r["test"]["sharpe"] > b0t and adj[c] < 0.05 and r["dev"]["sharpe"] >= b0d and delay_ok)
    best = max(ok, key=lambda c: ok[c]["test"]["sharpe"])
    tp = lab_.series[best]
    tp = tp[tp.index >= DEV_END]
    tsr = [r["test"]["sharpe"] / math.sqrt(365) for r in ok.values()]
    dsr = deflated_sharpe(ok[best]["test"]["sharpe"] / math.sqrt(365), tsr, len(tp), float(tp.skew()), float(tp.kurt() + 3))
    summary = {"best_test": best, "best_test_sharpe": ok[best]["test"]["sharpe"], "dsr_prob": round(dsr, 4),
               "n_directions_ok": len(ok), "n_dev_variants": len(lab_.all_sr),
               "beats_b0": [c for c, r in ok.items() if r["beats_b0"]],
               "unavailable": {c: r.get("reason") for c, r in lab_.res.items() if r.get("status") != "ok"}}
    lab_.res["_summary"] = summary
    trials.record(EXPERIMENT, variant="summary", sample="in_sample", params={}, metrics=summary, note="OPP-007 第 2 轮汇总判定")
    pd.concat(lab_.series, axis=1).to_parquet(OUT / "series.parquet")
    (OUT / "results.json").write_text(json.dumps(lab_.res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
