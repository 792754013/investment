"""OPP-006：把 r04_wick 的逐笔成交放进一个有限资金的账户里跑（只用样本内）。

r04_wick 的金额不限制同时持仓的总资金（全市场同时持仓峰值约 200 万 USDT 名义本金），对 30 万元以内的账户没有意义。
这里按时间顺序逐笔开仓：
- 每笔保证金 = 当前已实现权益 × f，且不超过空闲资金；低于 5 USDT 不开仓；
- 名义本金 = 保证金 × 杠杆，且 ≤ 成交那一分钟成交额的 1%（超出时保证金按比例缩小）；
- 逐仓：持有期内最不利价格 ≥ 1/k − 0.5% 视为强平，亏掉这笔保证金；否则按 r04 的悲观净收益结算，亏损不超过保证金；
- 盈亏在平仓时计入权益，复利滚动；权益低于 5 USDT 视为爆仓，停止。
f = 100% 对应"小资金、每次全仓上"的设想；f = 10% 对应分散下注。

只跑看结果之前就选定的 7 个口径（做多、悲观成本、穿过 0.1%），不在这里再选参数。

用法：uv run python research/analysis/r04b_portfolio.py
"""

import heapq
import itertools

import numpy as np
import polars as pl

from lab import trials
from r04_wick import COSTS, MMR, OUT, USDT_RMB, VOL_SHARE, one_at_a_time

CONFIGS = [("majors", 0.02, 60), ("majors", 0.02, 1440), ("majors", 0.03, 60), ("majors", 0.03, 1440),
           ("all", 0.05, 60), ("all", 0.08, 60), ("all", 0.12, 60)]
START_RMB = [10_000, 300_000]
LEVERAGES = [1, 3, 5, 10, 20, 50, 100]
FRACTIONS = [0.1, 1.0]
THROUGH = 0.001
MIN_USDT = 5


def trades(uni: str, level: float, hold: int) -> pl.DataFrame:
    f = pl.read_parquet(OUT / f"fills_{uni}.parquet").filter(
        pl.col("level") == level, pl.col("side") == "long", pl.col("pen") >= THROUGH,
        pl.col(f"gross_{hold}").is_not_null(), pl.col("vol_quote").is_not_null(),
    ).sort("inst_id", "ts")
    keep = np.concatenate([
        one_at_a_time(p["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]"), hold)
        for p in f.partition_by("inst_id", maintain_order=True)
    ])
    return f.filter(pl.Series(keep)).select(
        "ts", (pl.col("ts") + pl.duration(minutes=hold)).alias("end"),
        (pl.col(f"gross_{hold}") - COSTS["pessimistic"]).alias("net"), pl.col(f"adv_{hold}").alias("adv"),
        (VOL_SHARE * pl.col("vol_quote")).alias("cap"),
    ).sort("ts")


def run(t: pl.DataFrame, start_usdt: float, k: int, frac: float) -> dict:
    equity, locked, peak, max_dd = start_usdt, 0.0, start_usdt, 0.0
    open_pos: list = []
    year_end: dict[str, float] = {}
    n, n_liq, ruined = 0, 0, None
    for ts, end, net, adv, cap in t.iter_rows():
        while open_pos and open_pos[0][0] <= ts:
            _, _, margin, pnl = heapq.heappop(open_pos)
            equity += pnl
            locked -= margin
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1)
        year_end[str(ts.year)] = equity
        if equity < MIN_USDT:
            ruined = ts.isoformat()
            break
        margin = min(frac * equity, equity - locked)
        notional = min(margin * k, cap)
        margin = notional / k
        if margin < MIN_USDT:
            continue
        liq = k > 1 and adv >= 1 / k - MMR
        pnl = -margin if liq else max(notional * net, -margin)
        n += 1
        n_liq += liq
        locked += margin
        heapq.heappush(open_pos, (end, n, margin, pnl))
    for _, _, margin, pnl in open_pos:
        equity += pnl
    peak = max(peak, equity)
    max_dd = min(max_dd, equity / peak - 1)
    return {
        "n": n, "liq": round(n_liq / n, 3) if n else None, "end_rmb": round(equity * USDT_RMB),
        "multiple": round(equity / start_usdt, 2), "max_dd": round(max_dd, 3), "ruined": ruined,
        "year_end_rmb": {y: round(v * USDT_RMB) for y, v in year_end.items()},
    }


def main() -> None:
    rows = []
    for uni, level, hold in CONFIGS:
        t = trades(uni, level, hold)
        for start, k, frac in itertools.product(START_RMB, LEVERAGES, FRACTIONS):
            m = run(t, start / USDT_RMB, k, frac)
            params = {"universe": uni, "level": level, "hold_min": hold, "start_rmb": start, "leverage": k,
                      "fraction": frac, "cost": "pessimistic", "through": THROUGH}
            trials.record("r04b_portfolio", sample="in_sample", params=params, metrics=m,
                          note="OPP-006 有限资金账户；口径在看结果之前选定")
            rows.append({**params, **{k_: v for k_, v in m.items() if k_ != "year_end_rmb"},
                         "year_end_rmb": str(m["year_end_rmb"])})
    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet(OUT / "portfolio.parquet")
    with pl.Config(tbl_rows=300, tbl_cols=20, tbl_width_chars=250, fmt_str_lengths=120):
        print(res.drop("cost", "through"))


if __name__ == "__main__":
    main()
