"""OPP-006 快速否证：永续合约插针（只用样本内，口径见 docs/opportunities/OPP-006.yaml）。

1. 描述统计：1 分钟内相对前一分钟收盘偏离 ≥ X% 的次数、当分钟收回比例、60 分钟后仍没收回的比例。
2. 不预测的版本：每分钟在前收盘价下方（做多）/ 上方（做空）L% 挂限价单，价格穿过挂单价 THROUGH 才算成交，
   持有 H 分钟后按 taker 平仓（H=0 表示成交那一分钟的收盘平仓）。同一品种同一方向同时只持有 1 笔。
3. 完美预知上界：只保留"成交那一分钟收盘收回至少一半偏离"的成交，相当于事先知道这一分钟会出针并收回。
4. 杠杆换算：成交后到平仓（含成交那一分钟）的最不利价格 ≥ 1/k − 0.5% 视为强平，保证金全亏。

两个口径：
- all：全部永续（建卡时登记的口径），L = 3/5/8/12%，H = 0/15/60 分钟。
- majors：只看 BTC、ETH（用户 2026-11-03 要求），L = 1/1.5/2/3/5%，H 加上 240、1440 分钟
  （"方向对了就拿住"：只有强平才提前出场）。

成本：悲观 = 开平都按 taker 0.05% + 平仓滑点 0.1%（合计 0.20%）；有利 = 开仓 maker 0.02% + 平仓 taker 0.05% + 滑点 0.03%。
成交：穿过挂单价 0.1%（建卡口径）或 0.5%（更严，针底流动性薄）。
容量：每笔名义本金 ≤ 2 万 USDT，且 ≤ 成交那一分钟成交额的 1%；成交额缺失的成交不计入金额。
小资金口径：每笔固定保证金 1,000 USDT，名义本金 = 1,000 × 杠杆（同样受成交额 1% 限制）。
不计资金费；不限制同时持仓的总资金（对策略有利）。

用法：uv run python research/analysis/r04_wick.py
"""

import itertools

import numpy as np
import polars as pl

from lab import lake, trials
from lab.paths import data_root

OUT = data_root() / "research" / "r04_wick"
USDT_RMB = 7.1
YEARS = 52 / 12  # 2022-01 ～ 2026-04
MAX_NOTIONAL = 20_000
MARGIN = 1_000
VOL_SHARE = 0.01
MMR = 0.005
THROUGHS = [0.001, 0.005]
LEVERAGES = [3, 5, 10, 20, 50, 100]
COSTS = {"pessimistic": 0.0020, "favorable": 0.0010}
MAJORS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
UNIVERSES = {
    "all": dict(levels=[0.03, 0.05, 0.08, 0.12], holds=[0, 15, 60], depths=[0.03, 0.05, 0.10]),
    "majors": dict(levels=[0.01, 0.015, 0.02, 0.03, 0.05], holds=[0, 15, 60, 240, 1440], depths=[0.01, 0.02, 0.03, 0.05]),
}


def bars(inst: str, holds: list[int]) -> pl.DataFrame:
    df = (
        lake.scan("okx_swap_1m").filter(pl.col("inst_id") == inst)
        .select("ts", "open", "high", "low", "close", "vol_quote").collect().sort("ts")
    )
    # 前一分钟必须连续，否则不知道挂单价
    df = df.with_columns(
        pl.when(pl.col("ts").diff() == pl.duration(minutes=1)).then(pl.col("close").shift()).alias("ref")
    )
    later = df.select("ts", pl.col("close").alias("c60")).with_columns(pl.col("ts") - pl.duration(minutes=60))
    df = df.join(later, on="ts", how="left")
    for h in holds:
        # 平仓价：第 h 根之后的收盘；最不利价格：[t, t+h] 内的最低 / 最高；中间有缺口的不计
        df = df.with_columns(
            pl.col("close").shift(-h).alias(f"exit_{h}"),
            (pl.col("ts").shift(-h) - pl.col("ts")).alias(f"span_{h}"),
            pl.col("low").rolling_min(h + 1).shift(-h).alias(f"min_{h}"),
            pl.col("high").rolling_max(h + 1).shift(-h).alias(f"max_{h}"),
        )
    return df


def describe(df: pl.DataFrame, inst: str, depths: list[float]) -> pl.DataFrame:
    d = df.filter(pl.col("ref").is_not_null())
    rows = []
    for x, side in itertools.product(depths, ["down", "up"]):
        if side == "down":
            dev = 1 - pl.col("low") / pl.col("ref")
            rec = (pl.col("close") - pl.col("low")) / (pl.col("ref") - pl.col("low"))
            stuck = pl.col("c60") <= pl.col("ref") * (1 - x / 2)
        else:
            dev = pl.col("high") / pl.col("ref") - 1
            rec = (pl.col("high") - pl.col("close")) / (pl.col("high") - pl.col("ref"))
            stuck = pl.col("c60") >= pl.col("ref") * (1 + x / 2)
        rows.append(d.filter(dev >= x).select(
            pl.lit(inst).alias("inst_id"), pl.lit(x).alias("depth"), pl.lit(side).alias("side"),
            pl.col("ts").dt.year().alias("year"), (rec >= 0.5).alias("recovered"), stuck.fill_null(False).alias("stuck_60"),
        ))
    return pl.concat(rows)


def fills(df: pl.DataFrame, inst: str, levels: list[float], holds: list[int]) -> pl.DataFrame:
    out = []
    for lv, side in itertools.product(levels, ["long", "short"]):
        if side == "long":
            entry = pl.col("ref") * (1 - lv)
            pen = pl.col("entry") / pl.col("low") - 1  # 价格穿过挂单价的幅度
            rec = pl.col("close") >= pl.col("low") + 0.5 * (pl.col("ref") - pl.col("low"))
        else:
            entry = pl.col("ref") * (1 + lv)
            pen = pl.col("high") / pl.col("entry") - 1
            rec = pl.col("close") <= pl.col("high") - 0.5 * (pl.col("high") - pl.col("ref"))
        e = df.filter(pl.col("ref").is_not_null()).with_columns(entry.alias("entry")).filter(pen >= min(THROUGHS))
        cols = [pl.lit(inst).alias("inst_id"), "ts", pl.lit(lv).alias("level"), pl.lit(side).alias("side"),
                "entry", pen.alias("pen"), rec.alias("oracle"), "vol_quote"]
        sign = 1 if side == "long" else -1
        for h in holds:
            gross = sign * (pl.col(f"exit_{h}") / pl.col("entry") - 1)
            adverse = (1 - pl.col(f"min_{h}") / pl.col("entry")) if side == "long" else (pl.col(f"max_{h}") / pl.col("entry") - 1)
            ok = pl.col(f"span_{h}") == pl.duration(minutes=h)
            cols += [pl.when(ok).then(gross).alias(f"gross_{h}"), pl.when(ok).then(adverse).alias(f"adv_{h}")]
        out.append(e.select(cols))
    return pl.concat(out)


def one_at_a_time(ts: np.ndarray, hold: int) -> np.ndarray:
    """同一品种同一方向同时只持有 1 笔：上一笔平仓后的下一分钟才能再成交。"""
    keep = np.zeros(len(ts), dtype=bool)
    free = np.datetime64("1970-01-01T00:00", "m")
    step = np.timedelta64(hold + 1, "m")
    for i, t in enumerate(ts):
        if t >= free:
            keep[i] = True
            free = t + step
    return keep


def money(g: pl.DataFrame, pnl: pl.Expr) -> dict:
    p = g.with_columns(pnl.alias("pnl")).filter(pl.col("pnl").is_not_null())
    by_month = p.group_by(pl.col("ts").dt.strftime("%Y-%m").alias("ym")).agg(pl.col("pnl").sum()).sort("ym")
    total = p["pnl"].sum()
    top = max(by_month["pnl"].max() or 0.0, 0.0)
    cum = by_month["pnl"].cum_sum()
    return {
        "annual_rmb": round(total / YEARS * USDT_RMB),
        "annual_rmb_drop_top_month": round((total - top) / YEARS * USDT_RMB),
        "max_dd_rmb": round(float((cum - cum.cum_max().clip(lower_bound=0)).min() or 0) * USDT_RMB) if cum.len() else 0,
        "by_year_rmb": {str(y): round(v * USDT_RMB) for y, v in
                        p.group_by(pl.col("ts").dt.year().alias("y")).agg(pl.col("pnl").sum()).sort("y").iter_rows()},
    }


def simulate(f: pl.DataFrame, level: float, side: str, hold: int, oracle: bool, cost: float, through: float) -> dict:
    g = f.filter(pl.col("level") == level, pl.col("side") == side, pl.col("pen") >= through,
                 pl.col(f"gross_{hold}").is_not_null())
    if oracle:
        g = g.filter(pl.col("oracle"))
    g = g.sort("inst_id", "ts")
    if g.height == 0:
        return {"n": 0}
    keep = np.concatenate([
        one_at_a_time(p["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]"), hold)
        for p in g.partition_by("inst_id", maintain_order=True)
    ])
    g = g.filter(pl.Series(keep)).with_columns(
        (pl.col(f"gross_{hold}") - cost).alias("net"), pl.col(f"adv_{hold}").alias("adv"),
    )
    m = {
        "n": g.height,
        "n_major": g.filter(pl.col("inst_id").is_in(MAJORS)).height,
        "mean_net_bp": round(g["net"].mean() * 1e4, 1),
        "median_net_bp": round(g["net"].median() * 1e4, 1),
        "win": round((g["net"] > 0).mean(), 3),
        "worst_net": round(g["net"].min(), 3),
        "notional": money(g, pl.col("net") * pl.min_horizontal(pl.lit(MAX_NOTIONAL), VOL_SHARE * pl.col("vol_quote"))),
    }
    for k in LEVERAGES:
        liq = pl.col("adv") >= 1 / k - MMR
        mret = pl.when(liq).then(-1.0).otherwise(k * pl.col("net"))
        # 小资金：保证金 1,000 USDT；名义本金受成交额限制时，保证金按比例缩小
        notional = pl.min_horizontal(pl.lit(MARGIN * k), VOL_SHARE * pl.col("vol_quote"))
        s = g.select(liq.mean().alias("liq"), mret.mean().alias("mret"), (mret >= 1).mean().alias("p_double"))
        m[f"x{k}"] = {
            "liq": round(s["liq"][0], 3),
            "mean_margin_ret": round(s["mret"][0], 3),
            "p_double": round(s["p_double"][0], 3),
            **money(g, mret * notional / k),
        }
    return m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    every = lake.scan("okx_swap_1h").select("inst_id").unique().collect()["inst_id"].sort().to_list()
    rows = []
    for uni, cfg in UNIVERSES.items():
        insts = MAJORS if uni == "majors" else every
        desc, fl = [], []
        for i, inst in enumerate(insts, 1):
            df = bars(inst, cfg["holds"])
            if df.height < 2:
                continue
            desc.append(describe(df, inst, cfg["depths"]))
            fl.append(fills(df, inst, cfg["levels"], cfg["holds"]))
            if i % 50 == 0:
                print(f"  {uni} {i}/{len(insts)}", flush=True)
        desc, f = pl.concat(desc), pl.concat(fl)
        desc.write_parquet(OUT / f"events_{uni}.parquet")
        f.write_parquet(OUT / f"fills_{uni}.parquet")

        d = desc.group_by("depth", "side").agg(
            (pl.len() / YEARS).round(1).alias("per_year"),
            pl.col("inst_id").n_unique().alias("insts"),
            pl.col("inst_id").is_in(MAJORS).sum().alias("n_major"),
            pl.col("recovered").mean().round(3).alias("recovered"),
            pl.col("stuck_60").mean().round(3).alias("stuck_60"),
        ).sort("depth", "side")
        print(uni, d)
        d.write_csv(OUT / f"describe_{uni}.csv")
        trials.record("r04_wick", variant=f"describe_{uni}", sample="in_sample",
                      params={"universe": uni, "depths": cfg["depths"]}, metrics={"rows": d.to_dicts()},
                      note="OPP-006 描述统计")

        for lv, side, hold, oracle, (cname, cost), thr in itertools.product(
                cfg["levels"], ["long", "short"], cfg["holds"], [False, True], COSTS.items(), THROUGHS):
            m = simulate(f, lv, side, hold, oracle, cost, thr)
            params = {"universe": uni, "level": lv, "side": side, "hold_min": hold, "oracle": oracle, "cost": cname,
                      "through": thr, "max_notional": MAX_NOTIONAL, "margin": MARGIN, "vol_share": VOL_SHARE}
            trials.record("r04_wick", variant=f"{uni}_{'oracle' if oracle else 'resting'}", sample="in_sample",
                          params=params, metrics=m,
                          note="OPP-006 快速否证；oracle=完美预知上界，不可实现" if oracle else "OPP-006 快速否证")
            if m["n"] == 0:
                rows.append({**params, "n": 0})
                continue
            row = {**params, **{k: v for k, v in m.items() if not isinstance(v, dict)}}
            row.update({f"notional_{k}": v for k, v in m["notional"].items() if k != "by_year_rmb"})
            row["notional_by_year"] = str(m["notional"]["by_year_rmb"])
            for k in LEVERAGES:
                x = m[f"x{k}"]
                row.update({f"x{k}_liq": x["liq"], f"x{k}_mret": x["mean_margin_ret"], f"x{k}_p2": x["p_double"],
                            f"x{k}_rmb": x["annual_rmb"], f"x{k}_dd": x["max_dd_rmb"],
                            f"x{k}_by_year": str(x["by_year_rmb"])})
            rows.append(row)
    res = pl.DataFrame(rows, infer_schema_length=None)
    res.write_parquet(OUT / "grid.parquet")
    print(f"grid: {res.height} 组 → {OUT / 'grid.parquet'}")


if __name__ == "__main__":
    main()
