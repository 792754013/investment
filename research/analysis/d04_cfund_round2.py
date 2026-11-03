"""OPP-004 规则修改第 2 轮：板块限额 + 临近开放变体（预先登记见 docs/research/OPP004-round2-plan.md）。

判定标准沿用 OPP004-G1-criteria.md，不改。
相对第 1 轮（d03）：
  - 规则：每个板块（科创 / 创业板 / 其他，按名称）合计持仓 ≤ B；距开放天数增加 20～120 天的变体。
  - 口径修正（复核意见）：价格腿的分红延后 1 个交易日（场内除息比净值除息晚 1 天）；买入日 = 卖出日的零持有交易不买。
只读样本内。

用法：uv run python research/analysis/d04_cfund_round2.py
"""

import itertools

import polars as pl

import d01_cfund_discount as d1
from lab import lake, trials
from lab.paths import data_root

THETAS = [0.03, 0.04, 0.06, 0.08, 0.10]
WINDOWS = [(20, 800), (40, 600), (60, 400), (20, 120)]
LAGS = [1, 5, 10]
BOARD_CAPS = [30_000, 40_000, 50_000]
MIN_FEE = 5.0
OUT = data_root() / "research" / "d04"


def board_map() -> pl.DataFrame:
    return lake.scan("lof_list").select("code", "name").collect().with_columns(
        pl.when(pl.col("name").str.contains("科创|创新")).then(pl.lit("star"))
        .when(pl.col("name").str.contains("创业")).then(pl.lit("chinext")).otherwise(pl.lit("other")).alias("board"))


def panel() -> pl.DataFrame:
    p = d1.panel()
    # 场内价格比净值晚 1 个交易日除息：价格腿用前一天的累计分红
    return p.with_columns(pl.col("cumdiv").shift(1).over("code").fill_null(pl.col("cumdiv")).alias("cumdiv_px"))


def trades(p: pl.DataFrame, exit_lag: int) -> pl.DataFrame:
    """同 d01.trades(mtm=True)，但价格腿用 cumdiv_px。"""
    o = d1.opens(estimate=True)
    p = p.with_columns(pl.int_range(pl.len()).over("code").alias("i"))
    last_i = p.group_by("code").agg(pl.col("i").max().alias("last_i"))
    ex = o.sort("open_ts").join_asof(p.select("code", "ts", "i").sort("ts"), left_on="open_ts", right_on="ts",
                                     by="code", strategy="forward").join(last_i, on="code")
    ex = ex.with_columns(pl.col("i").fill_null(pl.col("last_i") - exit_lag))
    ex = ex.drop_nulls("i").with_columns(pl.min_horizontal(pl.col("i") + exit_lag, pl.col("last_i")).alias("xi")).join(
        p.select("code", pl.col("i").alias("xi"), pl.col("ts").alias("x_ts"), pl.col("close").alias("x_px"),
                 pl.col("nav").alias("x_nav"), pl.col("cumdiv").alias("x_div"), pl.col("cumdiv_px").alias("x_divpx")),
        on=["code", "xi"])
    ex = ex.select("code", "open_ts", "x_ts", "x_px", "x_nav", "x_div", "x_divpx").sort("open_ts")
    t = p.sort("ts").join_asof(ex, left_on="ts", right_on="open_ts", by="code", strategy="forward",
                               allow_exact_matches=False).drop_nulls("x_ts")
    dn = pl.col("x_div") - pl.col("cumdiv")
    dp = pl.col("x_divpx") - pl.col("cumdiv_px")
    return t.filter(pl.col("x_ts") > pl.col("ts")).with_columns(
        ((pl.col("open_ts") - pl.col("ts")).dt.total_days()).alias("days"),
        ((pl.col("x_px") + dp) / pl.col("close") - (pl.col("x_nav") + dn) / pl.col("nav")).alias("gross"),
        (pl.col("x_px") / pl.col("x_nav") - 1).alias("x_prem"))


def simulate(t: pl.DataFrame, boards: dict, theta: float, lo: int, hi: int, board_cap: float) -> pl.DataFrame:
    c = t.filter(pl.col("days").is_between(lo, hi) & pl.col("sig").is_not_null() & pl.col("amt20").is_not_null())
    c = c.with_columns(((-pl.col("sig") - 2 * d1.SLIP - 2 * d1.RATE) / (pl.col("days") / 365)).alias("ann")).filter(pl.col("ann") >= theta)
    by_day = c.sort("ts", "ann", "code", descending=[False, True, False]).partition_by("ts", as_dict=True, maintain_order=True)
    held, book, cash = {}, [], float(d1.CAP_TOTAL)
    for (ts,), g in sorted(by_day.items(), key=lambda kv: kv[0][0]):
        for k in [k for k, v in held.items() if v["x_ts"] <= ts]:
            cash += held.pop(k)["size"]
        used = {}
        for v in held.values():
            used[v["board"]] = used.get(v["board"], 0.0) + v["size"]
        for r in g.iter_rows(named=True):
            bd = boards.get(r["code"], "other")
            if r["code"] in held or cash < 1_000:
                continue
            size = min(cash, d1.CAP_ONE, d1.ADV_FRAC * r["amt20"], board_cap - used.get(bd, 0.0))
            if size < 1_000:
                continue
            cost = 2 * max(size * d1.RATE, MIN_FEE) / size + 2 * d1.SLIP
            held[r["code"]] = dict(x_ts=r["x_ts"], size=size, board=bd)
            used[bd] = used.get(bd, 0.0) + size
            cash -= size
            net = r["gross"] - cost
            book.append(dict(code=r["code"], board=bd, ts=ts, x_ts=r["x_ts"], size=size, gross=r["gross"], net=net, rmb=net * size))
    return pl.DataFrame(book)


def excess_curve(p: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    days = p.select("ts").unique().sort("ts")
    px = p.select("code", "ts", "close", "nav", "cumdiv", "cumdiv_px")
    pos = b.select("code", pl.col("ts").alias("t0"), "x_ts", "size").with_row_index("pid")
    pos = pos.join(px.rename({"ts": "t0", "close": "p0", "nav": "n0", "cumdiv": "c0", "cumdiv_px": "cp0"}), on=["code", "t0"])
    g = pos.join(days, how="cross").filter((pl.col("ts") >= pl.col("t0")) & (pl.col("ts") <= pl.col("x_ts")))
    g = g.join(px, on=["code", "ts"], how="left").sort("pid", "ts").with_columns(
        pl.col("close", "nav", "cumdiv", "cumdiv_px").forward_fill().over("pid"))
    cost1 = pl.max_horizontal(pl.col("size") * d1.RATE, pl.lit(MIN_FEE)) + pl.col("size") * d1.SLIP
    g = g.with_columns((pl.col("size") * ((pl.col("close") + pl.col("cumdiv_px") - pl.col("cp0")) / pl.col("p0")
                                          - (pl.col("nav") + pl.col("cumdiv") - pl.col("c0")) / pl.col("n0"))
                        - cost1 - pl.when(pl.col("ts") == pl.col("x_ts")).then(cost1).otherwise(0.0)).alias("pnl"))
    final = g.filter(pl.col("ts") == pl.col("x_ts")).select("x_ts", pl.col("pnl").alias("fin"))
    open_ = g.filter(pl.col("ts") < pl.col("x_ts")).group_by("ts").agg(pl.col("pnl").sum().alias("u"))
    real = final.group_by("x_ts").agg(pl.col("fin").sum().alias("r")).rename({"x_ts": "ts"})
    c = days.join(open_, on="ts", how="left").join(real, on="ts", how="left").fill_null(0.0).sort("ts")
    c = c.with_columns((pl.col("r").cum_sum() + pl.col("u")).alias("cum"))
    return c.with_columns((pl.col("cum") - pl.col("cum").cum_max().clip(lower_bound=0.0)).alias("dd"))


def by_year(b: pl.DataFrame) -> dict:
    spread = []
    for r in b.iter_rows(named=True):
        n = max((r["x_ts"] - r["ts"]).days, 1)
        for y in range(r["ts"].year, r["x_ts"].year + 1):
            lo = max(r["ts"], r["ts"].replace(year=y, month=1, day=1))
            hi = min(r["x_ts"], r["ts"].replace(year=y + 1, month=1, day=1))
            spread.append(dict(year=y, rmb=r["rmb"] * max((hi - lo).days, 0) / n))
    y = pl.DataFrame(spread).group_by("year").agg(pl.col("rmb").sum()).sort("year")
    return {str(k): round(v) for k, v in zip(y["year"], y["rmb"])}


def main() -> None:
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(25)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_float_precision(4)
    OUT.mkdir(parents=True, exist_ok=True)
    p = panel()
    boards = dict(board_map().select("code", "board").iter_rows())
    ts = {lag: trades(p, lag) for lag in LAGS}
    rows = []
    for theta, (lo, hi), lag, bc in itertools.product(THETAS, WINDOWS, LAGS, BOARD_CAPS):
        b = simulate(ts[lag], boards, theta, lo, hi, bc)
        if b.is_empty():
            m = dict(trades=0, avg=0.0, pos_years=0, excess_dd=0.0, top5=None, util=0.0)
        else:
            years = (b["x_ts"].max() - b["ts"].min()).days / 365
            yy = by_year(b)
            top = b.group_by("code").agg(pl.col("rmb").sum()).sort("rmb", descending=True)
            c = excess_curve(p, b)
            m = dict(trades=b.height, avg=b["rmb"].sum() / years,
                     pos_years=sum(v > 0 for k, v in yy.items() if "2021" <= k <= "2025"),
                     excess_dd=float(c["dd"].min()), top5=float(top["rmb"].head(5).sum() / top["rmb"].sum()),
                     util=float((b["size"] * (b["x_ts"] - b["ts"]).dt.total_days()).sum() / 365 / years / d1.CAP_TOTAL),
                     by_year=yy)
        trials.record("d04_cfund_round2", params=dict(theta=theta, min_days=lo, max_days=hi, exit_lag=lag, board_cap=bc,
                                                      min_fee=MIN_FEE, cap_total=d1.CAP_TOTAL, cap_one=d1.CAP_ONE,
                                                      adv_frac=d1.ADV_FRAC, slip=d1.SLIP, div_px_lag=1),
                      metrics=m, sample="in_sample", note="OPP-004 第 2 轮：板块限额 + 临近开放变体 + 复核口径修正")
        rows.append(dict(theta=theta, min_days=lo, max_days=hi, lag=lag, board_cap=bc)
                    | {k: v for k, v in m.items() if k != "by_year"} | {f"y{k}": v for k, v in m.get("by_year", {}).items()})
    g = pl.DataFrame(rows).with_columns(
        ((pl.col("avg") >= 3000) & (pl.col("pos_years") >= 4)).alias("c1"),
        (pl.col("excess_dd") >= -3000).alias("c2"),
        (pl.col("top5") <= 0.7).alias("c5a"))
    g.write_parquet(OUT / "grid.parquet")
    print("表 1 第 2 轮网格\n", g.sort("excess_dd", descending=True).head(40))
    ok = g.filter(pl.col("c1") & pl.col("c2") & pl.col("c5a"))
    print(f"\n满足 c1、c2、c5a：{ok.height} / {g.height}\n", ok)
    for k in ["board_cap", "min_days", "theta", "lag"]:
        print(g.group_by(k).agg(pl.col("avg").mean(), pl.col("excess_dd").mean().alias("dd_mean"), pl.col("excess_dd").max().alias("dd_best"),
                                pl.col("c1").mean().alias("c1_share"), pl.col("c2").mean().alias("c2_share")).sort(k))


if __name__ == "__main__":
    main()
