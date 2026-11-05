"""OPP-006 第 3 轮：秒级放量顺势、缩量离场（D5），只用样本内。

方案和判定条件见 docs/research/OPP006-round3-plan.md（看结果之前写定）。秒级特征复用 r06，训练流程复用 r05.run_direction。

候选：近 10 秒成交额 ≥ 近 1 小时每秒均值 × 10 × N，且价格同向移动 ≥ x（向下：比 60 秒最高价低；向上：比 60 秒最低价高）；
同一币种同一方向 10 分钟内只取第一个。N = 10、x = 0.6%；训练段某方向候选 < 500 时改用 N = 5、x = 0.5%。
进场：做多按 (t, t+1 秒] 最高价、做空按最低价，加 0.02% 滑点（延迟：3 秒窗口、滑点和成本加大）。
离场：vol50 / vol30（进场满 5 秒后近 5 秒成交额跌到这一波峰值的 50% / 30% 以下，下一秒收盘离场）、fix60、fix300；
最长 30 分钟；硬止损 s = 0.3% / 0.6% / 1.2%（进场那一秒只算不利一边；同一秒止损和离场都满足算止损）。

用法：uv run python research/analysis/r07_momentum_second.py
"""

import json
from datetime import UTC, datetime

import numpy as np
import polars as pl

import r05_wick_ml_minute as r05
from lab import lake, trials
from lab.paths import data_root
from r04_wick import MAJORS, one_at_a_time
from r05_wick_ml_minute import COST, TEST_START, TEST_YEARS, VAL_START, drop_top_days, minute_frame, run_direction, with_other
from r06_wick_ml_second import MET_FEATS, MIN_COLS, SEC_FEATS, months, sec_frame

OUT = data_root() / "research" / "r07_momentum"
EXPERIMENT = "r07_momentum"
STOPS = [0.003, 0.006, 0.012]
EXITS = ["vol50", "vol30", "fix60", "fix300"]
KMAX = {0.003: 100, 0.006: 50, 0.012: 50}
LEVS = [5, 10, 20, 50, 100]
HORIZON_S = 1800
ENTRY_SLIP, STOP_SLIP = 0.0002, 0.001
ENTRY_SLIP_D, STOP_SLIP_D, COST_D = 0.0004, 0.002, 0.0025
MARGIN_G0 = 2000
TRIGGERS = {"main": (10.0, 0.006), "fallback": (5.0, 0.005)}
CFGS = [f"{e}|{s}" for e in EXITS for s in STOPS]


def liq_ok(k: int, s: float) -> bool:
    return 1 / k - 0.005 > s + STOP_SLIP


def one_trade(ts, h, lo, c, v5, i, d, s, ex, delay, slip, stop_slip, cost):
    """返回 (净收益/名义本金, 离场时间, 进场后 5 分钟顺向最大浮盈)；数据不够返回 None。"""
    j0 = i + 1
    if j0 >= len(ts):
        return None
    j1 = max(np.searchsorted(ts, ts[i] + delay, side="right"), j0 + 1)
    e = (h[j0:j1].max() * (1 + slip)) if d == 1 else (lo[j0:j1].min() * (1 - slip))
    je = j1 - 1
    end = np.searchsorted(ts, ts[je] + HORIZON_S, side="right") - 1
    if end <= je + 1 or ts[-1] < ts[je] + HORIZON_S:
        return None
    seg = slice(je, end + 1)
    adverse = (lo[seg] <= e * (1 - s)) if d == 1 else (h[seg] >= e * (1 + s))
    k5 = np.searchsorted(ts, ts[je] + 300, side="right")
    k5 = max(k5, je + 2)
    mfe = (h[je + 1:k5].max() / e - 1) if d == 1 else (1 - lo[je + 1:k5].min() / e)
    if ex.startswith("vol"):
        r = int(ex[3:]) / 100
        peak = np.maximum.accumulate(v5[i:end + 1])[je - i:]
        cond = (v5[seg] < r * peak) & (ts[seg] >= ts[je] + 5)
        cond[-1] = False
        jx = int(np.argmax(cond)) + 1 if cond.any() else end - je  # 满足条件的下一秒收盘离场
    else:
        jx = min(int(np.searchsorted(ts, ts[je] + int(ex[3:]), side="left")), end) - je
    fs = int(np.argmax(adverse[:jx + 1])) if adverse[:jx + 1].any() else None
    if fs is not None:
        return -s - stop_slip - cost, int(ts[je + fs]), mfe
    return d * (c[je + jx] / e - 1) - cost, int(ts[je + jx]), mfe


def labels(df: pl.DataFrame, idx: np.ndarray, d: int) -> list[dict]:
    ts = df["ts"].dt.epoch("s").to_numpy()
    h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    v5 = df["v5"].to_numpy()
    out = []
    for i in idx:
        r = {"k": int(i)}
        ok = True
        for ex in EXITS:
            for s in STOPS:
                a = one_trade(ts, h, lo, c, v5, i, d, s, ex, 1, ENTRY_SLIP, STOP_SLIP, COST)
                b = one_trade(ts, h, lo, c, v5, i, d, s, ex, 3, ENTRY_SLIP_D, STOP_SLIP_D, COST_D)
                if a is None or b is None:
                    ok = False
                    break
                key = f"{ex}|{s}"
                r.update({f"net_{key}": a[0], f"y_{key}": int(a[0] > 0), f"exit_{key}": a[1], f"netd_{key}": b[0]})
                r["mfe"] = a[2]
            if not ok:
                break
        if not ok:
            continue
        # 放量峰值那一秒之后 60 秒的方向（描述量）
        k60 = np.searchsorted(ts, ts[i] + 60, side="right")
        pk = i + int(np.argmax(v5[i:k60]))
        k2 = np.searchsorted(ts, ts[pk] + 60, side="right") - 1
        r["after_peak"] = float(d * (c[k2] / c[pk] - 1)) if k2 > pk else 0.0
        out.append(r)
    return out


def events(inst: str, other: str, minute: pl.DataFrame, metrics: pl.DataFrame, side: str, trig) -> pl.DataFrame:
    n_vol, x = trig
    d = 1 if side == "up" else -1
    parts = []
    for m0 in months():
        m1 = datetime(m0.year + (m0.month == 12), m0.month % 12 + 1, 1, tzinfo=UTC)
        df = sec_frame(inst, datetime.fromtimestamp(m0.timestamp() - 3600, UTC),
                       datetime.fromtimestamp(m1.timestamp() + HORIZON_S + 60, UTC))
        if df.is_empty():
            continue
        mv = pl.col("drop") if d == -1 else pl.col("rebound")
        cand = df.with_row_index("k").with_columns(mv.alias("move")).filter(
            pl.col("ts") >= m0, pl.col("ts") < m1, pl.col("move") >= x, pl.col("volr10") >= n_vol)
        if cand.is_empty():
            continue
        keep = one_at_a_time(cand["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]"), 9)
        cand = cand.filter(pl.Series(keep))
        lab = labels(df, cand["k"].to_numpy(), d)
        if not lab:
            continue
        lab = pl.DataFrame(lab).with_columns(
            *(pl.from_epoch(pl.col(f"exit_{k}"), time_unit="s").dt.replace_time_zone("UTC").alias(f"exit_{k}") for k in CFGS))
        ev = cand.join(lab.with_columns(pl.col("k").cast(pl.UInt32)), on="k")
        oth = sec_frame(other, datetime.fromtimestamp(m0.timestamp() - 3600, UTC), m1)
        if not oth.is_empty():
            ev = ev.sort("ts").join_asof(oth.select("ts", pl.col("ret30").alias("oth_ret30"),
                                                    pl.col("drop").alias("oth_drop")), on="ts", strategy="backward")
        else:
            ev = ev.with_columns(pl.lit(None, pl.Float64).alias("oth_ret30"), pl.lit(None, pl.Float64).alias("oth_drop"))
        parts.append(ev)
        print(f"  {inst} {side} {m0:%Y-%m}: {ev.height} 个候选", flush=True)
    ev = pl.concat(parts, how="diagonal_relaxed").sort("ts")
    mf = minute.select((pl.col("ts") + pl.duration(minutes=1)).alias("mts"), *MIN_COLS)
    ev = ev.with_columns(pl.col("ts").dt.truncate("1m").alias("mts")).join(mf, on="mts", how="left")
    met = metrics.sort("ts").with_columns(
        (pl.col("oi_value") / pl.col("oi_value").shift(12) - 1).alias("oi_chg1h"),
        (pl.col("oi_value") / pl.col("oi_value").shift(288) - 1).alias("oi_chg24h"),
        (pl.col("ts") + pl.duration(minutes=5)).alias("ts"),
    ).select("ts", *MET_FEATS)
    return ev.join_asof(met, on="ts", strategy="backward").with_columns(pl.lit(inst).alias("inst_id"))


def build(side: str, trig, frames) -> pl.DataFrame:
    evs = []
    for inst in MAJORS:
        other = next(i for i in MAJORS if i != inst)
        met = lake.scan("bn_metrics").filter(pl.col("inst_id") == inst).collect()
        evs.append(events(inst, other, frames[inst], met, side, trig))
    return pl.concat(evs, how="diagonal_relaxed").sort("ts")


def streak_dd(net: np.ndarray, k: int) -> tuple[int, float]:
    """最长连续亏损笔数；按固定保证金、杠杆 k 算的最大回撤（单位：笔保证金）。"""
    run = best = 0
    for x in net:
        run = run + 1 if x <= 0 else 0
        best = max(best, run)
    eq = np.cumsum(net * k)
    dd = float((np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max()) if len(eq) else 0.0
    return best, dd


def describe(name: str, ev: pl.DataFrame) -> dict:
    out = {"n": ev.height, "after_peak_cont": float((ev["after_peak"] > 0).mean()),
           "after_peak_mean_bp": float(ev["after_peak"].mean() * 1e4), "cfg": {}}
    oracle = ev.filter(pl.col("mfe") >= 0.005)
    out["oracle_n"] = oracle.height
    for key in CFGS:
        ex, s = key.split("|")
        net = ev[f"net_{key}"].to_numpy()
        k = KMAX[float(s)]
        st, dd = streak_dd(net, k)
        out["cfg"][key] = {
            "win": round(float((net > 0).mean()), 3), "mean_bp": round(float(net.mean() * 1e4), 1),
            "delay_bp": round(float(ev[f"netd_{key}"].mean() * 1e4), 1),
            "oracle_win": round(float((oracle[f"net_{key}"] > 0).mean()), 3) if oracle.height else None,
            "oracle_mean_bp": round(float(oracle[f"net_{key}"].mean() * 1e4), 1) if oracle.height else None,
            "kmax": k, "max_lose_streak": st, "max_dd_margins": round(dd, 1),
            "margin_ret_per_trade": {lv: round(float(net.mean() * lv), 4) for lv in LEVS if liq_ok(lv, float(s))},
        }
    return out


def rule_check(res: dict, ev: pl.DataFrame) -> dict:
    """M1 规则本身（在验证段选的阈值）是否满足判定 1～4。"""
    out = {}
    test = ev.filter(pl.col("ts") >= TEST_START)
    for key, r in res["stops"].items():
        rule = r.get("m1_rule")
        if not rule:
            continue
        m = np.ones(test.height, bool)
        for col, t in zip(["move", "volr10"], rule):
            if t is not None:
                m &= test[col].fill_null(-np.inf).to_numpy() >= t
        nt, ntd = test[f"net_{key}"].to_numpy()[m], test[f"netd_{key}"].to_numpy()[m]
        n = int(m.sum())
        s = float(key.split("|")[1])
        money = float(nt.sum() * MARGIN_G0 * KMAX[s] / TEST_YEARS) if n else 0.0
        out[key] = {"n": n, "mean_bp": float(nt.mean() * 1e4) if n else None,
                    "delay_bp": float(ntd.mean() * 1e4) if n else None,
                    "drop5_bp_total": drop_top_days(test["ts"].filter(pl.Series(m)), nt) * 1e4 if n else None,
                    "annual_rmb": money}
        out[key]["pass_1_4"] = bool(n >= 20 and out[key]["mean_bp"] > 0 and out[key]["delay_bp"] > 0
                                    and out[key]["drop5_bp_total"] > 0 and money >= 3000)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = with_other({inst: minute_frame(inst) for inst in MAJORS})
    r05.breakeven = lambda s, cost=None, stop_slip=None: 0.0  # 非对称止盈止损的不亏线不适用；判定在下面重算
    results = {}
    for side, name in [("down", "D5_follow_short"), ("up", "D5_follow_long")]:
        trig_name = "main"
        ev = build(side, TRIGGERS["main"], frames)
        if ev.filter(pl.col("ts") < VAL_START).height < 500:
            trig_name = "fallback"
            ev = build(side, TRIGGERS["fallback"], frames)
        ev.write_parquet(OUT / f"events_{name}.parquet")
        desc = describe(name, ev)
        desc["trigger"] = {trig_name: TRIGGERS[trig_name]}
        print(name, json.dumps(desc, ensure_ascii=False), flush=True)
        trials.record(EXPERIMENT, variant=f"{name}_baseline", sample="in_sample",
                      params={"trigger": TRIGGERS[trig_name], "exits": EXITS, "stops": STOPS}, metrics=desc,
                      note="OPP-006 第 3 轮 D5：全部做、完美入场上界、峰值后方向（含验证、测试段）")
        res = run_direction(name, ev.with_columns(pl.col("move").alias("move")), SEC_FEATS + ["move"] + MIN_COLS + MET_FEATS,
                            CFGS, lambda key: MARGIN_G0 * KMAX[float(key.split("|")[1])],
                            {"direction": name, "rule_feats": ["move", "volr10"], "round": 3})
        # 按第 3 轮判定重算（去掉不适用的胜率条件）
        for key, r in res["stops"].items():
            r["pass"].pop("1_win", None)
            r["all_pass"] = all(r["pass"].values())
        ok = {k: r for k, r in res["stops"].items() if r["threshold"] is not None}
        res["chosen_stop"] = max(ok, key=lambda k: ok[k]["val_mean_bp"]) if ok else None
        res["pass"] = bool(res["chosen_stop"] and res["stops"][res["chosen_stop"]]["all_pass"])
        res["m1_check"] = rule_check(res, ev)
        res["describe"] = desc
        results[name] = res
        trials.record(EXPERIMENT, variant=f"{name}_verdict", sample="in_sample", params={"round": 3},
                      metrics={"chosen": res["chosen_stop"], "pass": res["pass"], "m1_check": res["m1_check"]},
                      note="OPP-006 第 3 轮：方向判定与 M1 规则检查")
        print(name, "选中", res["chosen_stop"], "通过" if res["pass"] else "不通过", flush=True)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
