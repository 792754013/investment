"""E3 现金管理（plan.md 4.2 节）：货币基金、国债逆回购、同业存单指数的收益与切换规则。

问题：
  1. 长期收益：滚动 1 天期逆回购（沪 204001 / 深 131810）、场外货基、场内货基、同业存单 AAA 指数，各年化多少。
  2. 高峰切换：平时放场内货基，逆回购利率高于货基 θ 时切到逆回购，每 10 万元每年多赚多少元。
  3. 手续费：逆回购佣金按上限 0.001% / 天期比例；最低收费不确定，分 0 / 1 / 5 元三档。

口径：
  - 期间 2017-06-01 ～ 2026-04-30（2017-05 起计息改为实际占款天数、365 天；之后到封存起点）。
  - 逆回购计息天数（上交所口径）：首次交收日 = T 之后的第一个交易日；到期清算日 = 不早于 T+n 自然日的第一个交易日；
    到期交收日 = 到期清算日之后的第一个交易日；计息天数 = 到期交收日 − 首次交收日。资金在到期清算日可再用于交易。
  - 成交利率用收盘价（尾盘下单可以拿到）；另报最低价作悲观下限。
  - 货基收益用每万份收益（按自然日），类别取每天各基金的中位数。剔除 B 类（大额份额）和净值型（每万份收益 > 20）。
  - 切换判断和放弃的货基收益，都用当天场内货基 7 日年化的中位数（比单日万份收益稳定，周末收益并到周一报的不受影响），
    按计息天数扣除；场内货基买卖成本假设每边 0.001%。
只读样本内。

用法：uv run python research/analysis/e03_cash.py          # 全部
      uv run python research/analysis/e03_cash.py by-year  # 只算同业存单指数和货基的逐年对比
"""

import itertools
import sys
from datetime import datetime, timedelta, UTC

import numpy as np
import polars as pl

from lab import lake, trials
from lab.paths import data_root

START, END = datetime(2017, 6, 1, 7, tzinfo=UTC), datetime(2026, 4, 30, 7, tzinfo=UTC)
AMT = 100_000
FEE_RATE = {1: 0.00001, 2: 0.00002, 3: 0.00003, 4: 0.00004, 7: 0.00005, 14: 0.0001, 28: 0.0002, 91: 0.0003, 182: 0.0003}
MMF_SWITCH = 0.00001  # 场内货基买卖成本，每边（假设）
OUT = data_root() / "research" / "e03"
NXT: dict = {}


def calendar() -> list:
    return sorted(d.date() for d in lake.scan("trade_cal").collect()["ts"])


def repo_days(cal: list) -> callable:
    """返回 f(T, n) -> (计息天数, 到期清算日)。结果缓存，避免重复查找。"""
    arr = np.array(cal)
    cache = {}

    def nxt(d):  # 严格晚于 d 的第一个交易日
        return arr[np.searchsorted(arr, d, side="right")]

    def first_ge(d):
        return arr[np.searchsorted(arr, d, side="left")]

    def f(t, n):
        k = (t, n)
        if k not in cache:
            s0 = nxt(t)
            c1 = first_ge(t + timedelta(days=n))
            cache[k] = ((nxt(c1) - s0).days, c1)
        return cache[k]
    return f


def next_day(cal: list) -> dict:
    return {d: cal[i + 1] for i, d in enumerate(cal[:-1])}


def mmf_daily() -> pl.DataFrame:
    """每个自然日：场外 / 场内货基的中位数日收益（小数）。"""
    m = lake.scan("mmf_daily").filter(pl.col("ts").is_between(START - timedelta(days=10), END)).collect()
    nav_like = m.group_by("code").agg(pl.col("w_income").median()).filter(pl.col("w_income") > 20)["code"]
    m = m.filter(~pl.col("code").is_in(nav_like) & ~pl.col("name").fill_null("").str.ends_with("B") & pl.col("w_income").is_not_null())
    m = m.with_columns(pl.col("on_exch").any().over("code").alias("on_exch"))
    d = m.group_by(pl.col("ts").dt.date().alias("d"), "on_exch").agg((pl.col("w_income").median() / 10_000).alias("r"),
                                                                       pl.col("y7").median().alias("y7"), pl.len().alias("n"))
    return d.pivot("on_exch", index="d", values=["r", "y7", "n"]).rename(
        {"r_false": "otc", "r_true": "exch", "y7_false": "otc_y7", "y7_true": "exch_y7", "n_false": "n_otc", "n_true": "n_exch"}).sort("d")


def roll(rates: dict, cal: list, f, px: str, min_fee: float, tenor: int = 1) -> dict:
    """每次到期就再做一笔 tenor 天期，按收盘（或最低）利率。返回年化、逐年金额。"""
    t, end = cal[np.searchsorted(cal, START.date())], min(END.date(), cal[-15])  # 日历只到封存起点
    out, n_trades = {}, 0
    while t <= end:
        r = rates.get(t)
        days, c1 = f(t, tenor)
        if r is not None and r[px] > 0:
            fee = max(AMT * FEE_RATE[tenor], min_fee)
            out[t.year] = out.get(t.year, 0.0) + AMT * r[px] / 100 * days / 365 - fee
            n_trades += 1
            t = c1
        else:
            t = NXT[t]
    years = (END - START).days / 365
    return dict(ann_pct=sum(out.values()) / AMT / years * 100, rmb_per_year=sum(out.values()) / years, trades_per_year=n_trades / years,
                by_year={str(k): round(v) for k, v in sorted(out.items())})


def switch(rates: dict, mmf: dict, cal: list, f, theta: float, min_fee: float, tenor: int = 1, only: set | None = None) -> dict:
    """平时放场内货基；当天逆回购收盘利率 − 场内货基 7 日年化（中位数，%）> θ 时切换。返回比"只放场内货基"多赚的金额。"""
    t, end = cal[np.searchsorted(cal, START.date())], min(END.date(), cal[-15])
    gain, n = {}, 0
    while t <= end:
        r = rates.get(t)
        days, c1 = f(t, tenor)
        y = mmf.get(t)
        if r is not None and y is not None and (only is None or t in only) and r["close"] - y > theta:
            fee = max(AMT * FEE_RATE[tenor], min_fee) + 2 * AMT * MMF_SWITCH
            g = AMT * (r["close"] - y) / 100 * days / 365 - fee
            gain[t.year] = gain.get(t.year, 0.0) + g
            n += 1
            t = c1
        else:
            t = NXT[t]
    years = (END - START).days / 365
    return dict(rmb_per_year=sum(gain.values()) / years, switches_per_year=n / years, by_year={str(k): round(v) for k, v in sorted(gain.items())})


def calendar_days(cal: list, f) -> dict:
    """几类"高峰日"：季末最后 2 个交易日；1 天期计息 ≥ 4 天的日子（长假前倒数第 2 个交易日等；
    长假前最后 1 个交易日做 1 天期只计 1 天）；1 天期计息正好 3 天的日子（多为周四）。"""
    qend, long_, three = set(), set(), set()
    for i, d in enumerate(cal[:-1]):
        nx = cal[i + 1]
        n = f(d, 1)[0]
        if n >= 4:
            long_.add(d)
        elif n == 3:
            three.add(d)
        if d.month in (3, 6, 9, 12) and nx.month != d.month:
            qend.add(d)
            if i > 0:
                qend.add(cal[i - 1])
    return {"季末": qend, "计息≥4天": long_, "计息3天": three, "季末或计息≥4天": qend | long_, "季末或计息≥3天": qend | long_ | three}


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(3)
    OUT.mkdir(parents=True, exist_ok=True)
    cal = calendar()
    f = repo_days(cal)
    NXT.update(next_day(cal))
    rp = lake.scan("repo_daily").filter(pl.col("ts").is_between(START - timedelta(days=10), END)).collect()
    rates = {code: {r["ts"].date(): r for r in g.iter_rows(named=True)} for (code,), g in rp.group_by("code")}

    # 计息天数抽查（上交所示例：周四 1 天期 → 3 天；周五 3 天期 → 1 天）
    thu = next(d for d in cal if d >= datetime(2025, 3, 6).date() and d.weekday() == 3)
    fri = thu + timedelta(days=1)
    print("计息天数抽查：周四 1 天期", f(thu, 1)[0], "天；周五 3 天期", f(fri, 3)[0], "天")

    md = mmf_daily()
    md.write_parquet(OUT / "mmf_median.parquet")
    mm = {r["d"]: r for r in md.iter_rows(named=True)}
    days = pl.date_range(START.date(), END.date(), "1d", eager=True)
    years = (END - START).days / 365
    base = {}
    for col in ["otc", "exch"]:
        tot = sum((mm.get(d) or {}).get(col) or 0.0 for d in days)
        base[col] = tot / years * 100
    print(f"\n货基中位数（剔除 B 类、净值型）：场外 {base['otc']:.2f}%/年，场内 {base['exch']:.2f}%/年；"
          f"样本数（最后一天）场外 {md['n_otc'].drop_nulls()[-1]}、场内 {md['n_exch'].drop_nulls()[-1]}")
    yr = md.filter(pl.col("d").is_between(START.date(), END.date())).group_by(pl.col("d").dt.year().alias("y")).agg(
        (pl.col("otc").sum() * 100).alias("otc_pct"), (pl.col("exch").sum() * 100).alias("exch_pct")).sort("y")
    print(yr)
    print("对照：7 日年化中位数的期间平均 场外", round(md.filter(pl.col("d").is_between(START.date(), END.date()))["otc_y7"].mean(), 3),
          "% 场内", round(md.filter(pl.col("d").is_between(START.date(), END.date()))["exch_y7"].mean(), 3), "%")

    ix = lake.scan("cash_idx").filter(pl.col("ts").is_between(START, END)).collect().sort("ts")
    c = ix["close"]
    ann = (c[-1] / c[0]) ** (1 / years) - 1
    dd = float((c / c.cum_max() - 1).min())
    m21 = float((c / c.shift(21) - 1).min())
    print(f"\n同业存单 AAA 指数：年化 {ann:.2%}，最大回撤 {dd:.2%}，最差 21 个交易日 {m21:.2%}（指数不含费用；基金还要扣约 0.2～0.3%/年的费用，未核实）")
    trials.record("e03_cash", params=dict(item="baseline", start=str(START.date()), end=str(END.date())),
                  metrics=dict(mmf_otc_pct=base["otc"], mmf_exch_pct=base["exch"], ncd_ann=ann, ncd_dd=dd), sample="in_sample",
                  note="E3 现金管理：基准收益")

    rows = []
    for code, px, mf in itertools.product(["204001", "131810"], ["close", "low"], [0.0, 1.0, 5.0]):
        m = roll(rates[code], cal, f, px, mf)
        trials.record("e03_cash", params=dict(item="roll", code=code, px=px, min_fee=mf, tenor=1, amt=AMT), metrics=m, sample="in_sample",
                      note="E3：滚动 1 天期逆回购")
        rows.append(dict(code=code, px=px, min_fee=mf) | {k: v for k, v in m.items() if k != "by_year"} | {f"y{k}": v for k, v in m["by_year"].items()})
    print("\n滚动 1 天期逆回购（10 万元）\n", pl.DataFrame(rows))

    cd = calendar_days(cal[:-15], f)  # 交易日历只到封存起点，末尾几天算不出到期日
    exch = {d: r["exch_y7"] for d, r in mm.items() if r.get("exch_y7") is not None}
    rows = []
    for code, theta, mf, sub in itertools.product(["204001", "131810"], [0.0, 0.5, 1.0, 2.0, 3.0], [0.0, 1.0, 5.0],
                                                  ["全部交易日", "季末", "计息≥4天", "计息3天", "季末或计息≥4天", "季末或计息≥3天"]):
        only = None if sub == "全部交易日" else cd[sub]
        m = switch(rates[code], exch, cal, f, theta, mf, only=only)
        trials.record("e03_cash", params=dict(item="switch", code=code, theta=theta, min_fee=mf, days=sub, tenor=1, amt=AMT,
                                              mmf_switch_cost=MMF_SWITCH), metrics=m, sample="in_sample", note="E3：场内货基 → 逆回购高峰切换")
        rows.append(dict(code=code, theta=theta, min_fee=mf, days=sub) | {k: v for k, v in m.items() if k != "by_year"}
                    | {f"y{k}": v for k, v in m["by_year"].items()})
    g = pl.DataFrame(rows)
    g.write_parquet(OUT / "switch.parquet")
    print("\n高峰切换比只放场内货基多赚（10 万元，最低收费 0 元）\n", g.filter(pl.col("min_fee") == 0).sort("code", "days", "theta"))
    print("\n最低收费的影响（θ = 1%）\n", g.filter(pl.col("theta") == 1.0).select("code", "days", "min_fee", "rmb_per_year", "switches_per_year").sort("code", "days", "min_fee"))


def by_year() -> None:
    """同业存单 AAA 指数 vs 场内 / 场外货基（7 日年化中位数的年平均），逐年。"""
    pl.Config.set_tbl_rows(20)
    pl.Config.set_float_precision(3)
    md = mmf_daily().filter(pl.col("d").is_between(START.date(), END.date()))
    m = md.group_by(pl.col("d").dt.year().alias("y")).agg(pl.col("exch_y7").mean().alias("场内货基%"), pl.col("otc_y7").mean().alias("场外货基%"))
    ix = lake.scan("cash_idx").filter(pl.col("ts").is_between(START, END)).collect().sort("ts")
    ix = ix.with_columns(pl.col("ts").dt.year().alias("y"))
    a = ix.group_by("y").agg(pl.col("close").first().alias("c0"), pl.col("close").last().alias("c1"), pl.len().alias("n"),
                             ((pl.col("close") / pl.col("close").cum_max() - 1).min() * 100).alias("年内最大回撤%"))
    a = a.with_columns(((pl.col("c1") / pl.col("c0")) ** (244 / pl.col("n")) * 100 - 100).alias("存单指数年化%"))
    t = m.join(a.select("y", "存单指数年化%", "年内最大回撤%"), on="y").sort("y").with_columns(
        (pl.col("存单指数年化%") - pl.col("场内货基%")).alias("存单−场内货基"))
    print(t)
    trials.record("e03_cash", params=dict(item="by_year_ncd_vs_mmf", start=str(START.date()), end=str(END.date())),
                  metrics=dict(rows={str(r["y"]): {k: round(v, 3) for k, v in r.items() if k != "y"} for r in t.iter_rows(named=True)}),
                  sample="in_sample", note="E3：同业存单指数 vs 货基逐年")


if __name__ == "__main__":
    by_year() if sys.argv[1:] == ["by-year"] else main()
