"""C1 v5 候选参数的稳健性与容量检查（CB-G1-criteria.md 第 2 节敏感性、第 4 节第 5 条容量）。

候选：v5 样本内同时满足第 1、3、4 条的两组（5 日 / 30 只 / 缓冲 1.0，10 日 / 30 只 / 缓冲 1.0）。
敏感性：每笔最低 5 元；违约回收 50%；价格下限 70 / 90；事件持有 10 / 40 日。
容量：按 20 万元资金放大每笔买入额，统计超过该券过去 20 日平均成交额 10% 的买单占比和金额占比。
只读样本内。

用法：uv run python research/analysis/cb04_candidates.py
"""

import polars as pl

import cb01_double_low as c1
from lab import trials
from lab.paths import data_root

CANDS = [(5, 30, 1.0), (10, 30, 1.0)]
VARIANTS = [dict(), dict(min_fee=5.0), dict(recovery=0.5), dict(px_floor=70), dict(px_floor=90),
            dict(ev_hold=10), dict(ev_hold=40)]
CAPACITY = 200_000
OUT = data_root() / "research" / "cb04"


def capacity(ev: pl.DataFrame, eq: pl.DataFrame, amt20: pl.DataFrame) -> dict:
    buys = ev.filter(pl.col("side") == "buy").join(amt20, on=["bond_code", "ts"], how="left")
    buys = buys.with_columns((pl.col("amount") * CAPACITY / c1.CAPITAL).alias("amt_cap"))
    over = buys.filter(pl.col("amt_cap") > 0.1 * pl.col("amt20"))
    return dict(n_buys=buys.height, share_over=over.height / buys.height,
                amount_share_over=float(over["amt_cap"].sum() / buys["amt_cap"].sum()),
                median_ratio=float((buys["amt_cap"] / buys["amt20"]).median()))


def main() -> None:
    p = c1.panel()
    idx = c1.indexes()
    end = p["ts"].max()
    info = {r["bond_code"]: (r["last_ts"], r["last_close"], (end - r["last_ts"]).days > c1.ALIVE_DAYS)
            for r in p.select("bond_code", "last_ts", "last_close").unique().to_dicts()}
    p = p.filter(pl.col("ts") >= c1.START)
    days = p["ts"].unique().sort().to_list()
    by_day = {d: {r[0]: (r[1], r[2]) for r in g.select("bond_code", "close", "amt20").iter_rows()}
              for (d,), g in p.group_by(["ts"])}
    elig = p.filter(pl.col("close").is_not_null() & pl.col("score").is_not_null() & (pl.col("age") >= 5)
                    & (pl.col("amt20") >= c1.MIN_AMT20) & (pl.col("stock_close") >= c1.STOCK_FLOOR))
    ranks = {}
    for floor in (70, c1.PX_FLOOR, 90):
        r = {d: g.sort("score", "bond_code")["bond_code"].to_list()
             for (d,), g in elig.filter(pl.col("close") >= floor).group_by(["ts"])}
        ranks[floor] = {d: r.get(d, []) for d in days}
    notices = c1.load_notices()
    amt20 = p.select("bond_code", "ts", "amt20")

    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for (f, n, bf), v in [(c, dict(v)) for c in CANDS for v in VARIANTS]:
        c1.EV_HOLD = v.pop("ev_hold", 20)
        eq, ev, b = c1.run(days, by_day, ranks, info, f, n, bf, notices=notices, call_exit=True, down_buy=True, **v)
        m = c1.evaluate(eq, b, idx)
        cap = capacity(ev, eq, amt20)
        full = [m["years"][str(y)] for y in range(2018, 2026)]
        params = dict(freq=f, n=n, buffer=bf, ev_hold=c1.EV_HOLD) | v
        trials.record("cb04_candidates", params=params,
                      metrics={k: m[k] for k in ("net_apr", "excess_b1", "excess_b2", "mdd", "turnover")}
                      | {"years": m["years"], "capacity": cap}, sample="in_sample",
                      note="v5b 候选的敏感性与容量检查")
        rows.append(params | dict(excess_b1=m["excess_b1"], excess_b2=m["excess_b2"], mdd=m["mdd"],
                                  b1_mdd=m["b1_mdd"], worst_year=min(full), y2025=m["years"]["2025"],
                                  shock=m["shock_excess"], turnover=m["turnover"]) | cap)
        eq.write_parquet(OUT / f"equity_f{f}_n{n}_{'_'.join(f'{k}{x}' for k, x in params.items() if k not in ('freq', 'n', 'buffer'))}.parquet")
    c1.EV_HOLD = 20
    t = pl.DataFrame(rows, infer_schema_length=None)
    t.write_parquet(OUT / "candidates.parquet")
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_float_precision(3)
    print(t)


if __name__ == "__main__":
    main()
