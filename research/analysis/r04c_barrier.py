"""OPP-006 机器学习方案的基准：BTC/ETH 1 分钟内下跌 1%/1.5%/2% 时挂单接住，不做任何筛选，
止盈、止损谁先到（1 分钟数据，同一分钟两边都碰到算止损，悲观）。只用样本内。

用法：uv run python research/analysis/r04c_barrier.py
"""

import numpy as np
import polars as pl

import r04_wick as w
from lab import lake, trials


T = 30  # 最长持有 30 分钟
BARRIERS = [(0.005, 0.01), (0.01, 0.02), (0.015, 0.03)]  # (止损, 止盈)，相对入场价
COST, STOP_SLIP = 0.002, 0.001
rows = []
for inst in w.MAJORS:
    df = lake.scan("okx_swap_1m").filter(pl.col("inst_id") == inst).select("ts", "high", "low", "close").collect().sort("ts")
    ts = df["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]")
    hi, lo, cl = (df[c].to_numpy() for c in ("high", "low", "close"))
    ref = np.r_[np.nan, cl[:-1]]
    ref[1:][np.diff(ts) != np.timedelta64(1, "m")] = np.nan
    for lv in (0.01, 0.015, 0.02):
        entry = ref * (1 - lv)
        idx = np.where(lo <= entry * (1 - 0.001))[0]
        keep = w.one_at_a_time(ts[idx], T)
        for i in idx[keep]:
            if i + T >= len(cl) or ts[i + T] - ts[i] != np.timedelta64(T, "m"):
                continue
            e = entry[i]
            for s, a in BARRIERS:
                out = None
                for j in range(i, i + T + 1):
                    # 成交那一分钟：成交后的最低价未知，按最低价算（悲观）；同一分钟两边都碰到，算止损
                    if lo[j] <= e * (1 - s):
                        out = -s - STOP_SLIP
                        break
                    if j > i and hi[j] >= e * (1 + a):
                        out = a
                        break
                if out is None:
                    out = cl[i + T] / e - 1
                rows.append((inst, ts[i], lv, s, a, out - COST, out == a))
r = pl.DataFrame(rows, schema=["inst", "ts", "level", "stop", "tp", "net", "tp_hit"], orient="row")
s = r.group_by("level", "stop", "tp").agg(
    pl.len().alias("n"), (pl.len() / (52 / 12)).round(0).alias("per_year"),
    pl.col("tp_hit").mean().round(3).alias("win"),
    (pl.col("net").mean() * 1e4).round(1).alias("mean_net_bp"),
).sort("level", "stop")
s = s.with_columns(((pl.col("stop") + STOP_SLIP + COST) / (pl.col("tp") + pl.col("stop") + STOP_SLIP)).round(3).alias("breakeven_win"))
print(s)
trials.record("r04c_barrier_baseline", sample="in_sample",
              params={"levels": [0.01, 0.015, 0.02], "barriers": BARRIERS, "max_hold_min": T, "cost": COST, "stop_slip": STOP_SLIP},
              metrics={"rows": s.to_dicts()}, note="OPP-006 机器学习方案的基准：不筛选时止盈先到的比例（1 分钟数据，悲观）")
