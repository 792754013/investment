"""OPP-006 第 2 轮：秒级机器学习（D4 小资金高杠杆接针），只用样本内。

方案和停止条件见 docs/research/OPP006-ml-plan.md；训练流程复用 r05_wick_ml_minute.run_direction。

数据：Binance 永续 1 秒 K 线 bn_swap_1s（逐笔成交聚合）、持仓量与多空比 bn_metrics（5 分钟一条，按晚 5 分钟使用）、
OKX 1 分钟特征（取决策时刻之前已收盘的那一分钟）。
候选：第 t 秒收盘价比过去 60 秒的最高价低 ≥ 1%；同一币种 10 分钟内只取第一个。
入场：第 t 秒结束时决策，按下一个有成交的秒的最高价买入，再加 0.02% 滑点（延迟敏感性：t 之后 3 秒内最高价、滑点加倍）。
标签：先到 +2s 记 1；先到 −s（入场那一秒的最低价也算，悲观）或 30 分钟到期记 0；同一秒两边都碰到算止损。
s = 0.5% / 1% / 1.5%，对应 100 / 约 66 / 约 50 倍逐仓（强平即止损），每笔保证金 500 元。

用法：uv run python research/analysis/r06_wick_ml_second.py
"""

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl

from lab import lake, trials
from lab.paths import data_root
from r04_wick import MAJORS, one_at_a_time
from r05_wick_ml_minute import COST, EXPERIMENT, minute_frame, run_direction, with_other

OUT = data_root() / "research" / "r06_wick_ml"
STOPS = [0.005, 0.01, 0.015]
HORIZON_S = 1800
ENTRY_SLIP, STOP_SLIP = 0.0002, 0.001
ENTRY_SLIP_D, STOP_SLIP_D, COST_D = 0.0004, 0.002, 0.0025
MARGIN_RMB = 500
LEVERAGE = {0.005: 100, 0.01: 66, 0.015: 50}
MIN_COLS = ["rv60", "rv1440", "ret60", "ret240", "ret1440", "volr_prev60", "dist_hi", "dist_lo", "ev24", "funding",
            "hour", "wday", "min_to_funding", "is_eth"]
SEC_FEATS = ["drop", "ret5", "ret10", "ret30", "accel", "rebound", "volr5", "volr10", "volr30", "volr60s",
             "sell10", "sell30", "ntr10", "bigr10", "oth_ret30", "oth_drop"]
MET_FEATS = ["oi_chg1h", "oi_chg24h", "ls_ratio", "top_ls_ratio", "taker_ls_vol"]


def months() -> list[datetime]:
    out, y, m = [], 2022, 1
    while (y, m) <= (2026, 4):
        out.append(datetime(y, m, 1, tzinfo=UTC))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def sec_frame(inst: str, start: datetime, end: datetime) -> pl.DataFrame:
    df = (lake.scan("bn_swap_1s").filter(pl.col("inst_id") == inst, pl.col("ts") >= start, pl.col("ts") < end)
          .select("ts", "open", "high", "low", "close", "vol_quote", "buy_quote", "n_trades", "max_trade_quote")
          .collect().sort("ts"))
    if df.is_empty():
        return df
    by = dict(by="ts")
    def back(k):  # noqa: E306  t − k 秒（或之前最后一笔）的收盘价
        return df.select((pl.col("ts") + pl.duration(seconds=k)).alias("ts"), pl.col("close").alias(f"c_{k}"))
    for k in (5, 10, 30):
        df = df.join_asof(back(k), on="ts", strategy="backward")
    vol = pl.col("vol_quote")
    mean1h = vol.rolling_sum_by(window_size="3600s", **by) / 3600
    return df.with_columns(
        pl.col("high").rolling_max_by(window_size="60s", **by).alias("max60"),
        pl.col("low").rolling_min_by(window_size="60s", **by).alias("min60"),
        *(vol.rolling_sum_by(window_size=f"{k}s", **by).alias(f"v{k}") for k in (5, 10, 30, 60)),
        pl.col("buy_quote").rolling_sum_by(window_size="10s", **by).alias("b10"),
        pl.col("buy_quote").rolling_sum_by(window_size="30s", **by).alias("b30"),
        pl.col("n_trades").rolling_sum_by(window_size="10s", **by).alias("n10"),
        pl.col("n_trades").rolling_sum_by(window_size="3600s", **by).alias("n3600"),
        pl.col("max_trade_quote").rolling_max_by(window_size="10s", **by).alias("big10"),
        mean1h.alias("vmean1h"),
    ).with_columns(
        (1 - pl.col("close") / pl.col("max60")).alias("drop"),
        *((pl.col("close") / pl.col(f"c_{k}") - 1).alias(f"ret{k}") for k in (5, 10, 30)),
        (pl.col("close") / pl.col("min60") - 1).alias("rebound"),
        *((pl.col(f"v{k}") / (k * pl.col("vmean1h"))).alias(f"volr{k}" if k < 60 else "volr60s") for k in (5, 10, 30, 60)),
        (1 - pl.col("b10") / pl.col("v10")).alias("sell10"),
        (1 - pl.col("b30") / pl.col("v30")).alias("sell30"),
        (pl.col("n10") / (pl.col("n3600") / 360)).alias("ntr10"),
        (pl.col("big10") / (pl.col("vmean1h") * 10)).alias("bigr10"),
    ).with_columns((pl.col("ret5") - (pl.col("ret10") - pl.col("ret5"))).alias("accel"))


def labels(df: pl.DataFrame, idx: np.ndarray) -> list[dict]:
    ts = df["ts"].dt.epoch("s").to_numpy()
    h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    out = []
    for i in idx:
        r = {"k": int(i)}
        ok = True
        for s in STOPS:
            for tag, delay, slip, stop_slip, cost in (("", 1, ENTRY_SLIP, STOP_SLIP, COST),
                                                      ("d", 3, ENTRY_SLIP_D, STOP_SLIP_D, COST_D)):
                j0 = i + 1
                if j0 >= len(ts):
                    ok = False
                    break
                j1 = np.searchsorted(ts, ts[i] + delay, side="right")  # (t, t+delay] 内最高价
                e = (h[j0:max(j1, j0 + 1)].max()) * (1 + slip)
                je = max(j1, j0 + 1) - 1
                end = np.searchsorted(ts, ts[je] + HORIZON_S, side="right") - 1
                if end <= je or ts[-1] < ts[je] + HORIZON_S:
                    ok = False
                    break
                stop = lo[je:end + 1] <= e * (1 - s)
                tp = h[je:end + 1] >= e * (1 + 2 * s)
                tp[0] = False  # 入场那一秒只算对自己不利的一边
                fs = int(np.argmax(stop)) if stop.any() else 10 ** 9
                ft = int(np.argmax(tp)) if tp.any() else 10 ** 9
                if fs <= ft and fs < 10 ** 9:
                    y, net, ex = 0, -s - stop_slip - cost, ts[je + fs]
                elif ft < 10 ** 9:
                    y, net, ex = 1, 2 * s - cost, ts[je + ft]
                else:
                    y, net, ex = 0, c[end] / e - 1 - cost, ts[end]
                if tag == "":
                    r.update({f"y_{s}": y, f"net_{s}": net, f"exit_{s}": int(ex)})
                else:
                    r[f"netd_{s}"] = net
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def events(inst: str, other: str, minute: pl.DataFrame, metrics: pl.DataFrame) -> pl.DataFrame:
    parts = []
    for m0 in months():
        m1 = datetime(m0.year + (m0.month == 12), m0.month % 12 + 1, 1, tzinfo=UTC)
        df = sec_frame(inst, datetime.fromtimestamp(m0.timestamp() - 3600, UTC),
                       datetime.fromtimestamp(m1.timestamp() + HORIZON_S + 60, UTC))
        if df.is_empty():
            continue
        cand = df.with_row_index("k").filter(pl.col("ts") >= m0, pl.col("ts") < m1, pl.col("drop") >= 0.01)
        if cand.is_empty():
            continue
        keep = one_at_a_time(cand["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]"), 9)
        cand = cand.filter(pl.Series(keep))
        lab = labels(df, cand["k"].to_numpy())
        if not lab:
            continue
        lab = pl.DataFrame(lab).with_columns(
            *(pl.from_epoch(pl.col(f"exit_{s}"), time_unit="s").dt.replace_time_zone("UTC").alias(f"exit_{s}")
              for s in STOPS))
        ev = cand.join(lab.with_columns(pl.col("k").cast(pl.UInt32)), on="k")
        oth = sec_frame(other, datetime.fromtimestamp(m0.timestamp() - 3600, UTC), m1)
        if not oth.is_empty():
            ev = ev.sort("ts").join_asof(oth.select("ts", pl.col("ret30").alias("oth_ret30"),
                                                    pl.col("drop").alias("oth_drop")), on="ts", strategy="backward")
        else:
            ev = ev.with_columns(pl.lit(None, pl.Float64).alias("oth_ret30"), pl.lit(None, pl.Float64).alias("oth_drop"))
        parts.append(ev)
        print(f"  {inst} {m0:%Y-%m}: {ev.height} 个候选", flush=True)
    ev = pl.concat(parts, how="diagonal_relaxed").sort("ts")
    # 分钟特征：决策秒所在分钟的上一分钟（已收盘）
    mf = minute.select((pl.col("ts") + pl.duration(minutes=1)).alias("mts"), *MIN_COLS)
    ev = ev.with_columns(pl.col("ts").dt.truncate("1m").alias("mts")).join(mf, on="mts", how="left")
    # 持仓量：5 分钟一条，按晚 5 分钟使用
    met = metrics.sort("ts").with_columns(
        (pl.col("oi_value") / pl.col("oi_value").shift(12) - 1).alias("oi_chg1h"),
        (pl.col("oi_value") / pl.col("oi_value").shift(288) - 1).alias("oi_chg24h"),
        (pl.col("ts") + pl.duration(minutes=5)).alias("ts"),
    ).select("ts", *MET_FEATS)
    return ev.join_asof(met, on="ts", strategy="backward").with_columns(pl.lit(inst).alias("inst_id"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = with_other({inst: minute_frame(inst) for inst in MAJORS})
    evs = []
    for inst in MAJORS:
        other = next(i for i in MAJORS if i != inst)
        met = lake.scan("bn_metrics").filter(pl.col("inst_id") == inst).collect()
        evs.append(events(inst, other, frames[inst], met))
    ev = pl.concat(evs, how="diagonal_relaxed").sort("ts")
    ev.write_parquet(OUT / "events_D4.parquet")
    from r05_wick_ml_minute import breakeven
    be = lambda s: breakeven(s, cost=COST + ENTRY_SLIP, stop_slip=STOP_SLIP)  # noqa: E731
    base = {s: {"n": ev.height, "win": round(float(ev[f"y_{s}"].mean()), 3),
                "mean_bp": round(float(ev[f"net_{s}"].mean() * 1e4), 1), "breakeven": round(be(s), 3)} for s in STOPS}
    print("D4 基准（全部做）", base, flush=True)
    trials.record(EXPERIMENT, variant="D4_second_long_baseline", sample="in_sample",
                  params={"stops": STOPS, "horizon_s": HORIZON_S}, metrics=base,
                  note="OPP-006 第 2 轮 D4：不筛选时的胜率（含验证、测试段）")
    import r05_wick_ml_minute as r05
    r05.breakeven = lambda s, cost=COST + ENTRY_SLIP, stop_slip=STOP_SLIP: (s + stop_slip + cost) / (3 * s + stop_slip)
    res = run_direction("D4_second_long", ev, SEC_FEATS + MIN_COLS + MET_FEATS, STOPS,
                        lambda s: MARGIN_RMB * LEVERAGE[s], {"direction": "D4_second_long", "rule_feats": ["drop", "volr10"]})
    res["baseline"] = base
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("D4 选中止损档", res["chosen_stop"], "通过" if res["pass"] else "不通过", flush=True)


if __name__ == "__main__":
    main()
