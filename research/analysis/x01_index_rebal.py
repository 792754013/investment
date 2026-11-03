"""OPP-005 快速否证：中小盘指数定期调样（中证 500、中证 1000、国证 2000）的事件收益。

问题：公告后买入调入股、持有到生效日收盘（或生效后买入调出股等反弹），扣掉悲观成本后，10 万元每年能赚多少？
  - 超额 = 个股后复权收益 − 所进指数收益（中证 500 / 中证 1000 / 国证 2000）。没有对冲工具（不融券、不做期货），
    所以绝对盈亏里含指数涨跌，两种都报。
  - 公告日：中证在周五收盘后发布，买入取公告后第 1 个交易日收盘（+d_in）。国证 2000 没有公告日，只看锚定生效日的窗口。
  - 生效日：最后一个旧样本交易日；"生效后"指它之后的交易日。
  - 成本：佣金万 2.5、每笔最低 5 元；印花税卖出 0.1%（2023-08-28 起 0.05%）；冲击每边 0.2%（敏感性 0.3%）。
  - 资金：
    v1：每期 10 万元按期内全部调入（或调出）股等权，单只 ≤ 20 日均成交额的 10%。股数多时每只只有 1,000～2,000 元，最低 5 元佣金占比很高。
    v2：按 plan.md 3.3 节，单个机会 ≤ 10 万元、单只 ≤ 10%：每期买 10 只、每只 1 万元。期望值 = 10 万 × 全部股票净收益的平均
        （相当于随机挑 10 只的期望），另用自助抽样给出"随机挑 10 只"的分布。
  - 国证 2000 只保留每期 ≥ 50 只的定期调整（其余是退市等临时调整）。
只读样本内。

用法：uv run python research/analysis/x01_index_rebal.py
"""

import itertools
from datetime import datetime, UTC

import polars as pl

from lab import lake, trials
from lab.paths import data_root

RATE, MIN_FEE = 0.00025, 5.0
STAMP_CUT = datetime(2023, 8, 28, tzinfo=UTC)
CAP_TOTAL, ADV_FRAC = 100_000, 0.1
SIZE_V2, K_V2 = 10_000, 10
BENCH = {"中证500": "中证500", "中证1000": "中证1000", "国证2000": "国证2000"}
OUT = data_root() / "research" / "x01"


def calendar() -> pl.DataFrame:
    return (lake.scan("idx_daily").filter(pl.col("index") == "中证500").select("ts").collect()
            .unique().sort("ts").with_row_index("i").with_columns(pl.col("i").cast(pl.Int64)))


def events(cal: pl.DataFrame) -> pl.DataFrame:
    r = lake.scan("idx_rebal").collect()
    r = r.filter(pl.len().over("index", "ts", "side") >= 50)  # 只要定期调整
    # 生效日 → 不晚于它的最后一个交易日；公告日 → 严格晚于它的第一个交易日（公告在收盘后发布）
    r = r.sort("ts").join_asof(cal.rename({"i": "eff_i"}), on="ts", strategy="backward")
    a = r.filter(pl.col("ann_ts").is_not_null()).sort("ann_ts").join_asof(
        cal.rename({"ts": "ann_ts", "i": "ann_i"}), on="ann_ts", strategy="forward", allow_exact_matches=False)
    return r.join(a.select("index", "code", "side", "ts", "ann_i"), on=["index", "code", "side", "ts"], how="left")


def prices(cal: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    e = lake.scan("eq_daily").select("code", "ts", "close", "amount").collect().join(cal, on="ts")
    e = e.sort("code", "i").with_columns(pl.col("amount").rolling_mean(20, min_samples=10).over("code").alias("amt20"))
    b = lake.scan("idx_daily").collect().join(cal, on="ts").select("index", "i", pl.col("close").alias("bclose"))
    return e, b


def legs(ev: pl.DataFrame, e: pl.DataFrame, b: pl.DataFrame, i0: pl.Expr, i1: pl.Expr) -> pl.DataFrame:
    """每只股票一条：i0 收盘买、i1 收盘卖。买入日必须有成交（停牌不买）；卖出日停牌则用之前最后一个收盘。"""
    t = ev.with_columns(i0.alias("i0"), i1.alias("i1")).drop_nulls(["i0", "i1"]).filter(pl.col("i1") > pl.col("i0"))
    buy = e.filter(pl.col("amount") > 0).select("code", pl.col("i").alias("i0"), pl.col("close").alias("p0"), "amt20")
    t = t.join(buy, on=["code", "i0"])
    sell = e.select("code", pl.col("i").alias("i1"), pl.col("close").alias("p1")).sort("i1")
    t = t.sort("i1").join_asof(sell, on="i1", by="code", strategy="backward")
    bi = b.rename({"index": "bench"})
    t = t.with_columns(pl.col("index").replace_strict(BENCH).alias("bench"))
    t = t.join(bi.rename({"i": "i0", "bclose": "b0"}), on=["bench", "i0"]).join(bi.rename({"i": "i1", "bclose": "b1"}), on=["bench", "i1"])
    return t.with_columns((pl.col("p1") / pl.col("p0") - 1).alias("ret"), (pl.col("b1") / pl.col("b0") - 1).alias("bret")).with_columns(
        (pl.col("ret") - pl.col("bret")).alias("ab"))


def book(t: pl.DataFrame, slip: float, cal: pl.DataFrame, v2: bool = False) -> pl.DataFrame:
    """每期（index, 生效日）10 万元。返回逐股的净盈亏（绝对、超额）。v2 每只按 1 万元算成本，再按 10 万 / 股数缩放。"""
    t = t.join(cal.rename({"i": "i1", "ts": "x_ts"}), on="i1")
    n = pl.len().over("index", "ts")
    if v2:
        t = t.with_columns(pl.min_horizontal(pl.lit(float(SIZE_V2)), ADV_FRAC * pl.col("amt20")).alias("size"))
    else:
        t = t.with_columns(pl.min_horizontal(pl.lit(CAP_TOTAL) / n, ADV_FRAC * pl.col("amt20")).alias("size"))
    stamp = pl.when(pl.col("x_ts") < STAMP_CUT).then(0.001).otherwise(0.0005)
    fee = 2 * pl.max_horizontal(pl.col("size") * RATE, pl.lit(MIN_FEE)) + pl.col("size") * (2 * slip + stamp)
    t = t.with_columns(fee.alias("cost")).with_columns(
        (pl.col("size") * pl.col("ret") - pl.col("cost")).alias("rmb_abs"),
        (pl.col("size") * pl.col("ab") - pl.col("cost")).alias("rmb_ex"))
    if v2:  # 10 万 / 股数：每期合计 = 10 万 × 平均净收益率
        k = pl.lit(CAP_TOTAL) / pl.col("size").sum().over("index", "ts")
        t = t.with_columns((pl.col(c) * k).alias(c) for c in ["rmb_abs", "rmb_ex", "cost", "size"])
    return t


def summarize(b: pl.DataFrame) -> dict:
    per = b.group_by("index", "ts").agg(pl.col("ab").mean().alias("ab"), pl.col("ret").mean().alias("ret"),
                                         pl.col("rmb_ex").sum(), pl.col("rmb_abs").sum(), pl.col("cost").sum(), pl.len().alias("n"))
    yy = per.group_by(pl.col("ts").dt.year().alias("y")).agg(pl.col("rmb_ex").sum(), pl.col("rmb_abs").sum()).sort("y")
    years = yy.height
    se = per["ab"].std() / per.height ** 0.5 if per.height > 1 else None
    return dict(events=per.height, stocks=b.height, ab_mean=float(b["ab"].mean()), ab_event_mean=float(per["ab"].mean()),
                ab_t=float(per["ab"].mean() / se) if se else None, hit=float((b["ab"] > 0).mean()),
                worst_event_ab=float(per["ab"].min()), cost_bp=float(b["cost"].sum() / b["size"].sum() * 1e4),
                ex_per_year=float(per["rmb_ex"].sum() / years), abs_per_year=float(per["rmb_abs"].sum() / years),
                worst_event_rmb_abs=float(per["rmb_abs"].min()),
                by_year={str(y): round(v) for y, v in zip(yy["y"], yy["rmb_ex"])})


def pick10(b: pl.DataFrame, reps: int = 2000, seed: int = 0) -> dict:
    """每期随机挑 10 只、每只 1 万：年均超额净额的分布（自助抽样）。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    net = (b["ab"] - b["cost"] / b["size"]).to_numpy()
    grp = b.select(pl.struct("index", "ts").rank("dense").alias("g"))["g"].to_numpy()
    years = b["ts"].dt.year().n_unique()
    idx = [np.flatnonzero(grp == g) for g in np.unique(grp)]
    tot = np.zeros(reps)
    for ix in idx:
        tot += rng.choice(net[ix], size=(reps, K_V2), replace=True).sum(axis=1) * SIZE_V2
    a = tot / years
    return dict(p10=float(np.percentile(a, 10)), p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)),
                share_ge_3000=float((a >= 3000).mean()))


def main() -> None:
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_float_precision(4)
    OUT.mkdir(parents=True, exist_ok=True)
    cal = calendar()
    ev = events(cal)
    e, b = prices(cal)
    print(ev.group_by("index", "side").agg(pl.col("ts").n_unique().alias("periods"), pl.len(), pl.col("ann_i").is_null().sum().alias("no_ann"),
                                           (pl.col("eff_i") - pl.col("ann_i")).median().alias("ann_to_eff_days")).sort("index", "side"))

    E, A = pl.col("eff_i"), pl.col("ann_i")
    # (名称, 方向, 买入, 卖出, 是否可交易)。可交易 = 买入时点已知（公告后 / 生效后）
    W = []
    for d_in, d_out in itertools.product([0, 2], [-1, 0, 1]):
        W.append((f"调入 公告+{d_in}→生效{d_out:+d}", "in", A + d_in, E + d_out, True))
    for k in [5, 10, 20]:
        W.append((f"调入 生效→+{k}", "in", E, E + k, True))
    for d0, k in itertools.product([0, 1], [5, 10, 20]):
        W.append((f"调出 生效+{d0}→+{d0 + k}", "out", E + d0, E + d0 + k, True))
    W += [("调入 公告−20→公告", "in", A - 20, A - 1, False), ("调出 公告+0→生效", "out", A, E, False),
          ("调入 生效−10→生效（国证用）", "in", E - 10, E, False), ("调出 生效−10→生效（国证用）", "out", E - 10, E, False)]

    rows = []
    IDX = {"中证500": ["中证500"], "中证1000": ["中证1000"], "国证2000": ["国证2000"], "中证500+1000": ["中证500", "中证1000"]}
    for (name, side, i0, i1, tradable), slip, idx in itertools.product(W, [0.002, 0.003], IDX):
        sub = ev.filter((pl.col("side") == side) & pl.col("index").is_in(IDX[idx]))
        t = legs(sub, e, b, i0, i1)
        if t.is_empty():
            continue
        if len(IDX[idx]) > 1:  # 同一生效日的两条指数合成一期，共用 10 万
            t = t.with_columns(pl.lit(idx).alias("index"))
        bk = book(t, slip, cal, v2=True)
        m = summarize(bk) | (pick10(bk) if tradable else {})
        trials.record("x01_index_rebal", params=dict(window=name, side=side, index=idx, slip=slip, rate=RATE, min_fee=MIN_FEE,
                                                     cap_total=CAP_TOTAL, size=SIZE_V2, k=K_V2, tradable=tradable, version="v2"),
                      metrics=m, sample="in_sample", note="OPP-005 快速否证 v2：单只 1 万（plan 3.3），国证只留定期调整")
        rows.append(dict(index=idx, window=name, slip=slip, tradable=tradable) | {k: v for k, v in m.items() if k != "by_year"}
                    | {f"y{k}": v for k, v in m["by_year"].items()})
        if slip == 0.002:
            bk.write_parquet(OUT / f"book_{idx}_{name.replace(' ', '_').replace('→', '-')}.parquet")
    g = pl.DataFrame(rows)
    g.write_parquet(OUT / "grid.parquet")
    g = g.drop([c for c in g.columns if c in ("stocks", "ab_mean", "hit")])
    for idx in IDX:
        print(f"\n== {idx}（冲击 0.2%）")
        print(g.filter((pl.col("index") == idx) & (pl.col("slip") == 0.002)).drop("index", "slip"))
    print("\n冲击 0.3% 时可交易配置里超额最好的 10 个")
    print(g.filter(pl.col("tradable") & (pl.col("slip") == 0.003)).sort("ex_per_year", descending=True).head(10))


if __name__ == "__main__":
    main()
