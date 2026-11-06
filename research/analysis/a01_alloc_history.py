"""配置的历史回看（plan.md 4.1 节："研究可以用免费数据回看某个配置在历史上的回撤和再平衡效果"）。

不替用户选比例：只把几组示意组合在历史上"最多亏多少、多久回本、长期赚多少"算出来，供用户写投资政策说明时对照。

口径（看结果之前定下）：
  期间：2005-03-01 ～ 2026-04-30（货基样本从 2005-02 起），只读样本内。按 A 股交易日逐日计算。
  资产：
    A 股：沪深300 全收益 H00300 与中证500 全收益 H00905 各半（组合内每年再平衡时一起回到各半）
    债券：中证全债 H11001（财富指数）
    货基：mmf_daily 每日万份收益的中位数累计成净值（已扣货基自身费用）
    美股：标普 500 价格 × 美元兑人民币，另加股息 1.8%/年 × (1 − 10% 预提税)，按自然日计提
  费用（年化，按自然日扣）：A 股指数基金 0.2%，债券指数基金 0.3%，美股 QDII 0.6%（不含溢价），货基 0
  再平衡：每年第一个交易日回到目标比例；调仓成本按换手额的 0.1%（悲观）
  三组示意组合，股票占比 0%～100%，每档 10%：
    G1 A 股 + 债券；G2 A 股 + 货基；G3 股票部分 A 股、美股各半 + 债券
  敏感性：不再平衡（买入后不动）；不扣费；两个子期间 2005-2014、2015-2026.04

用法：uv run python research/analysis/a01_alloc_history.py          # 主结果和敏感性
      uv run python research/analysis/a01_alloc_history.py starts   # 只算不同起点（2008-01、2010-01 起），写回 results.json
"""

import json
from datetime import datetime, UTC

import numpy as np
import polars as pl

from lab import lake, trials
from lab.paths import data_root

START, END = datetime(2005, 3, 1, tzinfo=UTC), datetime(2026, 4, 30, 23, tzinfo=UTC)
FEE = {"cn_eq": 0.002, "bond": 0.003, "us_eq": 0.006, "cash": 0.0}
US_DIV = 0.018 * 0.9
REB_COST = 0.001
OUT = data_root() / "research" / "a01"
SHARES = [i / 10 for i in range(11)]
GRIDS = {"G1": ("cn_eq", "bond"), "G2": ("cn_eq", "cash"), "G3": ("mix_eq", "bond")}


def load_prices() -> pl.DataFrame:
    """返回 A 股交易日上的各资产"含分红、扣费前"价格（列：date, cn300, cn500, bond, us, cash）。"""
    idx = lake.scan("idx_long").filter(pl.col("index").is_in(["H00300", "H00905"])).collect()
    cn = idx.with_columns(pl.col("ts").dt.date().alias("date")).pivot(on="index", index="date", values="close").sort("date")
    cn = cn.rename({"H00300": "cn300", "H00905": "cn500"})
    ap = lake.scan("alloc_px").collect().with_columns(pl.col("ts").dt.date().alias("date"))
    bond = ap.filter(pl.col("asset") == "cn_bond_all").select("date", pl.col("close").alias("bond"))
    spx = ap.filter(pl.col("asset") == "spx_price").select("date", pl.col("close").alias("spx"))
    fx = ap.filter(pl.col("asset") == "usdcny").select("date", pl.col("close").alias("fx"))
    us = spx.join(fx, on="date", how="left").sort("date").with_columns(pl.col("fx").forward_fill())
    us = us.select("date", (pl.col("spx") * pl.col("fx")).alias("us_px"))
    mmf = lake.scan("mmf_daily").collect().with_columns(pl.col("ts").dt.date().alias("date"))
    mmf = mmf.filter(pl.col("w_income") < 20)  # 剔除净值型（与 E3 口径一致）
    cash = mmf.group_by("date").agg(pl.col("w_income").median()).sort("date")
    cal = pl.DataFrame({"date": pl.date_range(cash["date"].min(), END.date(), "1d", eager=True)})
    cash = cal.join(cash, on="date", how="left").with_columns(pl.col("w_income").fill_null(0.0))
    cash = cash.with_columns((1 + pl.col("w_income") / 10_000).cum_prod().alias("cash")).select("date", "cash")
    d = cn.filter((pl.col("date") >= START.date()) & (pl.col("date") <= END.date()))
    for other in [bond, us, cash]:
        d = d.join_asof(other.sort("date"), on="date", strategy="backward")
    d = d.drop_nulls()
    # 美股股息按自然日计提进价格
    days = (pl.col("date") - pl.col("date").first()).dt.total_days()
    d = d.with_columns((pl.col("us_px") * (1 + US_DIV) ** (days / 365)).alias("us"))
    return d.select("date", "cn300", "cn500", "bond", "us", "cash")


def asset_returns(d: pl.DataFrame, fees: bool) -> dict:
    """各资产逐日收益（扣费）。沪深300、中证500 分开持有，组合层面每年再平衡时一起回到各半。
    美股按 QDII 的 T 日净值口径：A 股 D 日取同一日历日的美股收盘和汇率（join_asof 按日期）。"""
    gap = d["date"].diff().dt.total_days().fill_null(0).to_numpy()
    r = {}
    for k in ["cn300", "cn500", "bond", "us", "cash"]:
        x = d[k].to_numpy()
        r[k] = np.concatenate([[0.0], x[1:] / x[:-1] - 1])
    fee = (lambda a: FEE[a] * gap / 365) if fees else (lambda a: 0 * gap)
    return {"cn300": r["cn300"] - fee("cn_eq"), "cn500": r["cn500"] - fee("cn_eq"), "bond": r["bond"] - fee("bond"),
            "us": r["us"] - fee("us_eq"), "cash": r["cash"] - fee("cash")}


def weights(grid: str, s: float) -> dict:
    eq, safe = GRIDS[grid]
    w = {"cn300": 0.0, "cn500": 0.0, "bond": 0.0, "us": 0.0, "cash": 0.0}
    if eq == "cn_eq":
        w["cn300"] = w["cn500"] = s / 2
    else:
        w["cn300"] = w["cn500"] = s / 4
        w["us"] = s / 2
    w[safe] += 1 - s
    return w


def simulate(dates: np.ndarray, r: dict, w: dict, rebalance: bool) -> np.ndarray:
    keys = [k for k in w if w[k] > 0]
    years = np.array([dd.year for dd in dates])
    hold = np.array([w[k] for k in keys])
    rets = np.vstack([r[k] for k in keys])
    v = np.empty(len(dates))
    v[0] = 1.0
    for t in range(1, len(dates)):
        hold = hold * (1 + rets[:, t])
        total = hold.sum()
        if rebalance and years[t] != years[t - 1]:
            target = np.array([w[k] for k in keys]) * total
            total -= np.abs(target - hold).sum() * REB_COST
            hold = np.array([w[k] for k in keys]) * total
        v[t] = total
    return v


def metrics(dates: np.ndarray, v: np.ndarray) -> dict:
    n_years = (dates[-1] - dates[0]).days / 365.25
    cagr = v[-1] ** (1 / n_years) - 1
    dr = v[1:] / v[:-1] - 1
    vol = dr.std() * np.sqrt(252)
    peak = np.maximum.accumulate(v)
    dd = v / peak - 1
    t_trough = int(dd.argmin())
    t_peak = int(np.argmax(v[: t_trough + 1]))
    rec = np.where(v[t_trough:] >= v[t_peak])[0]
    t_rec = t_trough + int(rec[0]) if len(rec) else None
    # 最长水下时间（创新高之间的最长间隔）
    longest, start, uw = 0, 0, (None, None)
    for t in range(1, len(v)):
        if v[t] >= peak[t - 1]:
            if (dates[t] - dates[start]).days > longest:
                longest, uw = (dates[t] - dates[start]).days, (str(dates[start]), str(dates[t]))
            start = t
    if (dates[-1] - dates[start]).days > longest:
        longest, uw = (dates[-1] - dates[start]).days, (str(dates[start]), None)
    years = np.array([dd_.year for dd_ in dates])
    cal = {}
    for y in np.unique(years):
        idx = np.where(years == y)[0]
        prev = idx[0] - 1 if idx[0] > 0 else idx[0]
        cal[int(y)] = v[idx[-1]] / v[prev] - 1
    full_years = {y: x for y, x in cal.items() if 2006 <= y <= 2025}
    # 滚动持有期：每个月初买入，持有 H 年
    month_starts = [i for i in range(len(dates)) if i == 0 or dates[i].month != dates[i - 1].month]
    roll = {}
    for h in [1, 3, 5, 10]:
        outs = []
        for i in month_starts:
            target = dates[i].replace(year=dates[i].year + h)
            j = np.searchsorted(dates, target)
            if j < len(dates):
                outs.append(v[j] / v[i])
        outs = np.array(outs)
        if len(outs) == 0:
            continue
        roll[f"{h}y"] = {"n": len(outs), "loss_prob": float((outs < 1).mean()),
                         "worst_ann": float(outs.min() ** (1 / h) - 1), "median_ann": float(np.median(outs) ** (1 / h) - 1)}
    return {
        "cagr": float(cagr), "vol": float(vol), "mdd": float(dd.min()),
        "mdd_peak": str(dates[t_peak]), "mdd_trough": str(dates[t_trough]),
        "mdd_recover": str(dates[t_rec]) if t_rec is not None else None,
        "mdd_recover_months": round((dates[t_rec] - dates[t_peak]).days / 30.44, 1) if t_rec is not None else None,
        "longest_underwater_months": round(longest / 30.44, 1),
        "longest_underwater": uw,
        "worst_year": [int(min(full_years, key=full_years.get)), float(min(full_years.values()))],
        "best_year": [int(max(full_years, key=full_years.get)), float(max(full_years.values()))],
        "calendar": {y: round(float(x), 4) for y, x in cal.items()},
        "rolling": roll,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_prices()
    dates = np.array(d["date"].to_list())
    r_fee, r_raw = asset_returns(d, True), asset_returns(d, False)
    res, series = {"period": [str(dates[0]), str(dates[-1])], "main": {}, "sens": {}}, {}
    month_end = [i for i in range(len(dates)) if i == len(dates) - 1 or dates[i + 1].month != dates[i].month]
    series["dates"] = [str(dates[i]) for i in month_end]
    for g in GRIDS:
        for s in SHARES:
            key = f"{g}_{int(s * 100)}"
            w = weights(g, s)
            v = simulate(dates, r_fee, w, True)
            m = metrics(dates, v)
            res["main"][key] = m
            series[key] = [round(float(v[i]), 5) for i in month_end]
            trials.record("a01_alloc_history", sample="in_sample", variant=key,
                          params={"weights": w, "rebalance": "annual", "fees": FEE, "us_div": US_DIV, "reb_cost": REB_COST},
                          metrics={k: m[k] for k in ["cagr", "vol", "mdd", "mdd_recover_months"]},
                          note="描述性配置回看，不选参数；第 3 次运行：加最长水下区间起止日期，组合和口径不变")
    # 敏感性：只对 G1 做（不再平衡、不扣费、两个子期间）
    for s in [0.3, 0.5, 0.7, 1.0]:
        w = weights("G1", s)
        for name, v in [("no_rebal", simulate(dates, r_fee, w, False)), ("no_fee", simulate(dates, r_raw, w, True))]:
            key = f"G1_{int(s * 100)}_{name}"
            res["sens"][key] = {k: metrics(dates, v)[k] for k in ["cagr", "mdd", "mdd_recover_months"]}
            trials.record("a01_alloc_history", sample="in_sample", variant=key, params={"weights": w},
                          metrics=res["sens"][key], note="敏感性")
        for lo, hi in [("2005-03-01", "2014-12-31"), ("2015-01-01", "2026-04-30")]:
            mask = (dates >= datetime.fromisoformat(lo).date()) & (dates <= datetime.fromisoformat(hi).date())
            sub = {k: x[mask] for k, x in r_fee.items()}
            v = simulate(dates[mask], sub, w, True)
            key = f"G1_{int(s * 100)}_{lo[:4]}-{hi[:4]}"
            mm = metrics(dates[mask], v)
            res["sens"][key] = {k: mm[k] for k in ["cagr", "mdd", "mdd_recover_months"]}
            trials.record("a01_alloc_history", sample="in_sample", variant=key, params={"weights": w},
                          metrics=res["sens"][key], note="敏感性：子期间")
    # 各资产单独的年化（扣费后），给报告用
    res["assets"] = {}
    for k in ["cn300", "cn500", "bond", "us", "cash"]:
        v = np.cumprod(1 + r_fee[k])
        m = metrics(dates, v)
        res["assets"][k] = {x: m[x] for x in ["cagr", "vol", "mdd", "mdd_peak", "mdd_trough", "mdd_recover_months"]}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (OUT / "series.json").write_text(json.dumps(series, ensure_ascii=False))
    print(json.dumps({"period": res["period"], "assets": res["assets"]}, ensure_ascii=False, indent=1))
    for g in GRIDS:
        print(g)
        for s in SHARES:
            m = res["main"][f"{g}_{int(s * 100)}"]
            print(f"  股票{int(s * 100):3d}%  年化{m['cagr']:6.1%}  波动{m['vol']:6.1%}  最大回撤{m['mdd']:7.1%} "
                  f"({m['mdd_peak']}→{m['mdd_trough']}，回本{m['mdd_recover_months']}月)  最长水下{m['longest_underwater_months']}月 "
                  f"最差年{m['worst_year'][0]} {m['worst_year'][1]:6.1%}  "
                  f"持有1年亏损概率{m['rolling']['1y']['loss_prob']:.0%} 3年{m['rolling']['3y']['loss_prob']:.0%} "
                  f"5年{m['rolling']['5y']['loss_prob']:.0%} 10年{m['rolling']['10y']['loss_prob']:.0%} 5年最差年化{m['rolling']['5y']['worst_ann']:.1%}")
    print("sens", json.dumps(res["sens"], ensure_ascii=False))


def starts() -> None:
    """复核意见：全期年化依赖 2005-03 起点。对 33 个组合分别从 2008、2010 年第一个交易日起重新模拟（权重从起点开始）。"""
    import sys as _s  # noqa: F401
    d = load_prices()
    dates_all = np.array(d["date"].to_list())
    r_all = asset_returns(d, True)
    res = json.loads((OUT / "results.json").read_text())
    res["starts"] = {}
    for y in [2008, 2010]:
        mask = dates_all >= datetime(y, 1, 1).date()
        dates = dates_all[mask]
        r = {k: np.concatenate([[0.0], x[mask][1:]]) for k, x in r_all.items()}
        for g in GRIDS:
            for s in SHARES:
                key = f"{g}_{int(s * 100)}"
                v = simulate(dates, r, weights(g, s), True)
                m = metrics(dates, v)
                out = {k: m[k] for k in ["cagr", "mdd", "longest_underwater_months"]}
                res["starts"].setdefault(str(y), {})[key] = out
                trials.record("a01_alloc_history", sample="in_sample", variant=f"{key}_from{y}",
                              params={"weights": weights(g, s), "start": str(dates[0])}, metrics=out,
                              note="复核后补：不同起点的全期年化")
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for y in ["2008", "2010"]:
        print(y, {k: f"{v['cagr']:.1%}/{v['mdd']:.0%}" for k, v in res["starts"][y].items()})


if __name__ == "__main__":
    import sys
    starts() if sys.argv[1:] == ["starts"] else main()
