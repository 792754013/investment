"""C1 双低轮动回测（CB-P0-plan.md 第 5 节，口径按 CB-G1-criteria.md）。

规则：
  双低值 = 收盘价 + 转股溢价率（百分数）。每 FREQ 个交易日在 T 日收盘后排序，
  已持有且排名在前 K = N × BUFFER 名内的继续持有，其余卖出，空位按排名从前往后补满 N 只。
  新买入按 T 日权益 / N 定额，按 10 张取整；已持有的不再平衡，以减少换手。
可投池：T 日有收盘价和溢价率，上市满 5 个交易日，过去 20 日平均成交额 ≥ 100 万元；
  v3 起加两条信用过滤（只用 T 日数据）：转债收盘价 ≥ PX_FLOOR，正股收盘价 ≥ 1.5 元（防面值退市）。
  v2 没有这两条，结果在退市整理期买进了蓝盾、岭南等即将退市的券。
成交：T+1 日收盘价；当天没有行情（停牌等）就顺延，买单顺延超过 5 个交易日则作废。
成本：佣金 万2（可选每笔最低 5 元）+ 经手费 0.001% + 冲击 0.10%；价格 < 90 或 20 日均成交额 < 500 万时冲击 0.30%。
退出：数据在样本末之前结束的券，最后收盘价 ≥ 95 视为强赎 / 到期 / 回售，按最后收盘价扣成本卖出；
     < 95 视为退市违约，按 RECOVERY × 最后收盘价回收（主口径 0，即全损）。
只读样本内，结果写入 data/research/cb01/。

用法：uv run python research/analysis/cb01_double_low.py
"""

import itertools
import math
from datetime import datetime, UTC

import polars as pl

from lab import lake, trials
from lab.paths import data_root

CAPITAL = 75_000
START = datetime(2018, 1, 1, tzinfo=UTC)
CREDIT_SHOCK = (datetime(2023, 11, 1, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC))
MIN_AMT20 = 1_000_000
PX_FLOOR, STOCK_FLOOR = 80, 1.5
RATE, EXCH = 0.0002, 0.00001
SLIP, SLIP_ILLIQ = 0.001, 0.003
DEFAULT_PX = 95
ALIVE_DAYS = 20  # 最后一条行情距样本末不到 20 天的视为仍在存续（可能只是临时停牌），不做退出处理
GRID = dict(freq=[5, 10, 20], n=[10, 20, 30], buffer=[1.0, 1.5, 2.0])
SENS = [(20, 20, 1.5), (10, 20, 1.5), (20, 10, 1.0)]  # 在这几组上做成本、回收率、价格下限的敏感性
OUT = data_root() / "research" / "cb01"


# ---------- 数据 ----------

def panel() -> pl.DataFrame:
    daily = lake.scan("cb_daily").select("bond_code", "ts", "close", "volume").collect()
    value = lake.scan("cb_value").select("bond_code", "ts", "conv_premium").collect()
    stock = lake.scan("stock_daily").select("stock_code", "ts", pl.col("close").alias("stock_close")).collect()
    codes = lake.scan("cb_list").select("bond_code", "stock_code").collect()
    p = (
        daily.join(value, on=["bond_code", "ts"], how="left")
        .join(codes, on="bond_code", how="left")
        .join(stock, on=["stock_code", "ts"], how="left")
        .sort("bond_code", "ts")
        .with_columns((pl.col("volume") * pl.col("close")).alias("amount"))
        .with_columns(
            pl.col("amount").rolling_mean(20, min_samples=5).over("bond_code").alias("amt20"),
            pl.int_range(pl.len()).over("bond_code").alias("age"),
            (pl.col("close") + pl.col("conv_premium")).alias("score"),
        )
    )
    last = p.group_by("bond_code").agg(pl.col("ts").max().alias("last_ts"), pl.col("close").last().alias("last_close"))
    return p.join(last, on="bond_code")


def indexes() -> dict[str, pl.DataFrame]:
    i = lake.scan("cb_index").filter(pl.col("ts") >= START).select("index_code", "ts", "close").collect()
    return {c: g.sort("ts").select("ts", "close") for (c,), g in i.group_by(["index_code"])}


# ---------- 回测 ----------

class Book:
    def __init__(self, min_fee: float, recovery: float):
        self.cash, self.pos, self.min_fee, self.recovery = float(CAPITAL), {}, min_fee, recovery
        self.bought, self.defaults, self.events = 0.0, 0, []

    def fee(self, amt: float, px: float, amt20: float | None) -> float:
        slip = SLIP_ILLIQ if px < 90 or (amt20 or 0) < 5_000_000 else SLIP
        return max(amt * RATE, self.min_fee) + amt * (EXCH + slip)

    def sell(self, code, px, amt20, day, why):
        z = self.pos.pop(code)
        amt = z * px
        self.cash += amt - self.fee(amt, px, amt20)
        self.events.append((day, code, "sell", why, amt))

    def buy(self, code, px, amt20, budget, day):
        z = math.floor(budget / (px * 10)) * 10
        while z > 0 and z * px + self.fee(z * px, px, amt20) > self.cash:
            z -= 10
        if z <= 0:
            return
        amt = z * px
        self.cash -= amt + self.fee(amt, px, amt20)
        self.pos[code] = self.pos.get(code, 0) + z
        self.bought += amt
        self.events.append((day, code, "buy", "", amt))


def run(days, by_day, ranks, info, freq, n, buffer, min_fee=0.0, recovery=0.0, px_floor=PX_FLOOR):
    b = Book(min_fee, recovery)
    last_px: dict[str, float] = {}
    pend_sell: set[str] = set()
    pend_buy: dict[str, tuple[float, int]] = {}
    curve = []
    k = math.ceil(n * buffer)
    for di, day in enumerate(days):
        today = by_day[day]
        for code, (px, _) in today.items():
            if code in b.pos:
                last_px[code] = px
        # 数据结束的持仓：强赎 / 到期，或退市违约
        for code in [c for c in b.pos if info[c][2] and info[c][0] < day]:
            lp = info[code][1]
            if lp >= DEFAULT_PX:
                b.sell(code, lp, None, day, "end")
            else:
                z = b.pos.pop(code)
                b.cash += z * lp * b.recovery
                b.defaults += 1
                b.events.append((day, code, "sell", "default", z * lp * b.recovery))
            pend_sell.discard(code)
        # 执行挂单（T+1 收盘）
        for code in [c for c in pend_sell if c in today]:
            b.sell(code, *today[code], day, "rotate")
            pend_sell.discard(code)
        for code in list(pend_buy):
            budget, waited = pend_buy[code]
            if code in today and code not in pend_sell:
                b.buy(code, *today[code], budget, day)
                last_px[code] = today[code][0]
                del pend_buy[code]
            elif waited >= 5:
                del pend_buy[code]
            else:
                pend_buy[code] = (budget, waited + 1)
        equity = b.cash + sum(z * last_px.get(c, info[c][1]) for c, z in b.pos.items())
        curve.append((day, equity, len(b.pos), b.cash))
        # T 日收盘后出信号
        if di % freq == 0 and di + 1 < len(days):
            order = ranks[px_floor][day]
            rank = {c: i for i, c in enumerate(order)}
            keep = {c for c in b.pos if rank.get(c, 10**9) < k}
            for c in b.pos:
                if c not in keep:
                    pend_sell.add(c)
            pend_buy.clear()
            slots = n - len(keep)
            for c in order:
                if slots <= 0:
                    break
                if c not in b.pos:
                    pend_buy[c] = (equity / n, 0)
                    slots -= 1
    eq = pl.DataFrame(curve, schema=["ts", "equity", "n_hold", "cash"], orient="row")
    ev = pl.DataFrame(b.events, schema=["ts", "bond_code", "side", "why", "amount"], orient="row")
    return eq, ev, b


# ---------- 评估 ----------

def ann(s: pl.Series, ts: pl.Series) -> float:
    years = (ts[-1] - ts[0]).days / 365.25
    return (s[-1] / s[0]) ** (1 / years) - 1


def mdd(s: pl.Series) -> float:
    return float((1 - s / s.cum_max()).max())


def evaluate(eq: pl.DataFrame, b: Book, idx: dict) -> dict:
    cb, mf = (idx[c].filter(pl.col("ts").is_between(eq["ts"][0], eq["ts"][-1])) for c in ("000832", "H11025"))
    net = ann(eq["equity"], eq["ts"])
    b1, b2 = ann(cb["close"], cb["ts"]), ann(mf["close"], mf["ts"])
    # 年度收益用上一年最后一天作起点
    def year_ret(d, col):
        d = d.sort("ts").with_columns(pl.col("ts").dt.year().alias("y"))
        ends = d.group_by("y").agg(pl.col(col).last().alias("e")).sort("y")
        return dict(zip(ends["y"], (ends["e"] / ends["e"].shift(1, fill_value=d[col][0]) - 1)))
    ys, yi = year_ret(eq, "equity"), year_ret(cb, "close")
    lo, hi = CREDIT_SHOCK
    seg = lambda d, col: (lambda x: x[col][-1] / x[col][0] - 1)(d.filter(pl.col("ts").is_between(lo, hi)))
    years = (eq["ts"][-1] - eq["ts"][0]).days / 365.25
    return dict(
        net_apr=net, b1_apr=b1, b2_apr=b2, excess_b1=net - b1, excess_b2=net - b2,
        mdd=mdd(eq["equity"]), b1_mdd=mdd(cb["close"]),
        turnover=b.bought / float(eq["equity"].mean()) / years,
        defaults=b.defaults, avg_hold=float(eq["n_hold"].mean()),
        shock_excess=seg(eq, "equity") - seg(cb, "close"),
        years={str(y): round(ys[y] - yi.get(y, 0), 4) for y in ys},
    )


def main() -> None:
    p = panel()
    idx = indexes()
    end = p["ts"].max()
    info = {r["bond_code"]: (r["last_ts"], r["last_close"], (end - r["last_ts"]).days > ALIVE_DAYS)
            for r in p.select("bond_code", "last_ts", "last_close").unique().to_dicts()}
    p = p.filter(pl.col("ts") >= START)
    days = p["ts"].unique().sort().to_list()
    by_day = {d: {r[0]: (r[1], r[2]) for r in g.select("bond_code", "close", "amt20").iter_rows()}
              for (d,), g in p.group_by(["ts"])}
    elig = p.filter(pl.col("close").is_not_null() & pl.col("score").is_not_null()
                    & (pl.col("age") >= 5) & (pl.col("amt20") >= MIN_AMT20) & (pl.col("stock_close") >= STOCK_FLOOR))
    ranks = {}
    for floor in (70, PX_FLOOR, 90):
        r = {d: g.sort("score")["bond_code"].to_list() for (d,), g in elig.filter(pl.col("close") >= floor).group_by(["ts"])}
        ranks[floor] = {d: r.get(d, []) for d in days}
    print(f"交易日 {len(days)}，{days[0]:%Y-%m-%d} ～ {days[-1]:%Y-%m-%d}")

    OUT.mkdir(parents=True, exist_ok=True)
    runs = [dict(freq=f, n=n, buffer=bf, min_fee=0.0, recovery=0.0) for f, n, bf in itertools.product(*GRID.values())]
    runs += [dict(freq=f, n=n, buffer=bf, min_fee=5.0, recovery=0.0) for f, n, bf in SENS]
    runs += [dict(freq=f, n=n, buffer=bf, min_fee=0.0, recovery=0.5) for f, n, bf in SENS]
    runs += [dict(freq=f, n=n, buffer=bf, px_floor=fl) for f, n, bf in SENS for fl in (70, 90)]
    res = []
    for r in runs:
        eq, ev, b = run(days, by_day, ranks, info, **r)
        m = evaluate(eq, b, idx)
        r = {"min_fee": 0.0, "recovery": 0.0, "px_floor": PX_FLOOR} | r
        tag = (f"f{r['freq']}_n{r['n']}_b{int(r['buffer'] * 10)}_m{int(r['min_fee'])}"
               f"_r{int(r['recovery'] * 10)}_p{r['px_floor']}")
        eq.write_parquet(OUT / f"equity_{tag}.parquet")
        ev.write_parquet(OUT / f"events_{tag}.parquet")
        trials.record("cb01_double_low", params=r | dict(capital=CAPITAL, slip=SLIP, min_amt20=MIN_AMT20,
                                                         stock_floor=STOCK_FLOOR),
                      metrics={k: m[k] for k in ("net_apr", "excess_b1", "excess_b2", "mdd", "turnover", "defaults")}
                      | {"years": m["years"]}, sample="in_sample",
                      note="v3：加入转债价格下限与正股 1.5 元下限的信用过滤")
        res.append(r | {k: v for k, v in m.items() if k != "years"} | {f"y{k}": v for k, v in m["years"].items()})
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "years"}, flush=True)
    t = pl.DataFrame(res)
    t.write_parquet(OUT / "grid.parquet")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(3)
    print(t.drop("b1_apr", "b2_apr"))


if __name__ == "__main__":
    main()
