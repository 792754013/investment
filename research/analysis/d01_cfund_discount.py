"""OPP-004 快速否证：封闭 / 定开基金折价收敛，每年比"直接持有同一份净值"多赚多少元。

思路：封闭期内场内价格常常低于净值，到开放期可以按净值赎回，折价收敛到接近 0。
超额 = 场内持有收益 − 同期净值收益（都含分红）。股票仓位的涨跌两边相同，抵消掉；剩下的就是折价收敛。
这笔钱没有对冲，持有期间仍要承担基金本身的涨跌，所以它的含义是"用折价基金替代一部分本来就要持有的权益仓位"。

口径（悲观）：
  - 信号用前一个交易日的折价（T 日净值晚上才公布），T 日收盘买入。
  - 开放日：cfund_events 里"开放申购赎回 / 封闭期届满"公告的日期（同一基金 30 天内多条只取第一条）。
    开放日是基金合同事先写好的，所以买入时已知距开放还有多久，不算前视。
  - 退出：公告日之后第 EXIT_LAG 个交易日收盘，在场内卖出（不用赎回，避免赎回费口径不确定）；
    敏感性另算"按净值赎回，扣 0.5% 赎回费"。
  - 成本：佣金万 2.5（最低 5 元情形另算），冲击每边 0.2%；仓位不超过买入前 20 日均成交额的 10%。
  - 资金：本机会上限 10 万元，单只不超过 3 万元（plan.md 3.3 节）。
只读样本内；开放日在封存起点之后的封闭期，买入了也无法在样本内结算，不计入。

用法：uv run python research/analysis/d01_cfund_discount.py
"""

import itertools
import re

import polars as pl

from lab import lake, trials
from lab.paths import data_root

RATE = 0.00025
SLIP = 0.002
CAP_TOTAL = 100_000
CAP_ONE = 30_000
ADV_FRAC = 0.1
REDEEM_FEE = 0.005
OUT = data_root() / "research" / "d01"


def panel() -> pl.DataFrame:
    pr = lake.scan("lof_price").collect()
    nv = lake.scan("lof_nav").collect()
    ac = lake.scan("cfund_accnav").collect()
    codes = lake.scan("cfund_events").select("code").unique().collect()["code"].to_list()
    p = (pr.filter(pl.col("code").is_in(codes) & (pl.col("close") > 0))
         .join(nv, on=["code", "ts"]).join(ac, on=["code", "ts"], how="left").sort("code", "ts"))
    p = p.with_columns(pl.col("accnav").forward_fill().over("code"))
    return p.with_columns(
        (pl.col("accnav") - pl.col("nav")).alias("cumdiv"),  # 累计分红（每份）
        (pl.col("close") / pl.col("nav") - 1).alias("prem"),
        pl.col("amount").shift(1).rolling_mean(20, min_samples=10).over("code").alias("amt20"),
    ).with_columns((pl.col("close") / pl.col("nav") - 1).shift(1).over("code").alias("sig"))


def period_months(title: str) -> int | None:
    if re.search(r"封闭运作.*转(为|型)", title) or "转为上市开放式" in title:
        return None  # 封闭期届满后转为普通 LOF，不再封闭
    for pat, m in [(r"(两|2)年", 24), (r"(三|3)年", 36), (r"18个月", 18), (r"(一|1)年", 12)]:
        if re.search(pat, title):
            return m
    return None


def opens(estimate: bool = False) -> pl.DataFrame:
    """开放公告日。estimate=True 时，对仍定期开放的基金补一个"下一次开放日"估计值
    （最近一次开放公告 + 合同周期，周期从标题解析），只补落在样本末尾之后的那一个。"""
    raw = lake.scan("cfund_events").select("code", "ts", "title").collect()
    e = raw.select("code", "ts").unique().sort("code", "ts")
    e = e.with_columns((pl.col("ts").diff().over("code").dt.total_days()).alias("gap"))
    e = e.filter(pl.col("gap").is_null() | (pl.col("gap") > 30)).select("code", pl.col("ts").alias("open_ts"))
    if not estimate:
        return e
    per = raw.with_columns(pl.col("title").map_elements(period_months, return_dtype=pl.Int64).alias("m"))
    conv = per.filter(pl.col("title").str.contains("转为上市开放式|转型为"))["code"].unique().to_list()
    last = (per.filter(~pl.col("code").is_in(conv)).group_by("code")
            .agg(pl.col("ts").max().alias("last"), pl.col("m").drop_nulls().first().alias("m")).drop_nulls("m"))
    end = lake.scan("lof_price").select(pl.col("ts").max()).collect().item()
    est = last.select("code", pl.col("last").dt.offset_by(pl.format("{}mo", pl.col("m"))).alias("open_ts")).filter(
        pl.col("open_ts") > end - pl.duration(days=30))
    return pl.concat([e, est]).unique().sort("code", "open_ts")


def trades(p: pl.DataFrame, exit_lag: int, mtm: bool = False) -> pl.DataFrame:
    """每个基金日，假设当天买入、持有到下一个开放公告后第 exit_lag 个交易日卖出。
    mtm=True：下一次开放日在样本末尾之后的（用合同周期估计），按样本最后一天收盘盯市结算，
    避免"越接近封存起点、能结算的交易越少"造成的低估。"""
    o = opens(estimate=mtm)
    p = p.with_columns(pl.int_range(pl.len()).over("code").alias("i"))
    last_i = p.group_by("code").agg(pl.col("i").max().alias("last_i"))
    # 退出日：开放公告日之后（含当日）的第一个交易日，再往后 exit_lag 个交易日；超出样本的取最后一天
    ex = o.sort("open_ts").join_asof(p.select("code", "ts", "i").sort("ts"), left_on="open_ts", right_on="ts",
                                     by="code", strategy="forward").join(last_i, on="code")
    if mtm:
        ex = ex.with_columns(pl.col("i").fill_null(pl.col("last_i") - exit_lag))
    ex = ex.drop_nulls("i").with_columns(pl.min_horizontal(pl.col("i") + exit_lag, pl.col("last_i")).alias("xi")).join(
        p.select("code", pl.col("i").alias("xi"), pl.col("ts").alias("x_ts"), pl.col("close").alias("x_px"),
                 pl.col("nav").alias("x_nav"), pl.col("cumdiv").alias("x_div")), on=["code", "xi"])
    ex = ex.select("code", "open_ts", "x_ts", "x_px", "x_nav", "x_div").sort("open_ts")
    t = p.sort("ts").join_asof(ex, left_on="ts", right_on="open_ts", by="code", strategy="forward",
                               allow_exact_matches=False).drop_nulls("x_ts")
    d = pl.col("x_div") - pl.col("cumdiv")
    t = t.with_columns(
        ((pl.col("open_ts") - pl.col("ts")).dt.total_days()).alias("days"),
        (pl.col("open_ts") > pl.col("x_ts")).alias("is_mtm"),
        ((pl.col("x_px") + d) / pl.col("close") - (pl.col("x_nav") + d) / pl.col("nav")).alias("gross"),
        ((pl.col("x_nav") * (1 - REDEEM_FEE) + d) / pl.col("close") - (pl.col("x_nav") + d) / pl.col("nav")).alias("gross_redeem"),
        (pl.col("x_px") / pl.col("x_nav") - 1).alias("x_prem"),
    )
    # 盯市的仓位在样本末尾还不能赎回，只能按场内价格计
    return t.with_columns(pl.when(pl.col("is_mtm")).then(pl.col("gross")).otherwise(pl.col("gross_redeem")).alias("gross_redeem"))


def simulate(t: pl.DataFrame, theta: float, min_days: int, max_days: int, min_fee: float, col: str) -> dict:
    """按日滚动：持仓到期释放资金；有空余资金时买年化折价最高的，规模受 3 万 / 10 万 / 成交额约束。"""
    c = t.filter(pl.col("days").is_between(min_days, max_days) & pl.col("sig").is_not_null() & pl.col("amt20").is_not_null())
    c = c.with_columns(((-pl.col("sig") - 2 * SLIP - 2 * RATE) / (pl.col("days") / 365)).alias("ann")).filter(pl.col("ann") >= theta)
    by_day = c.sort("ts", "ann", "code", descending=[False, True, False]).partition_by("ts", as_dict=True, maintain_order=True)
    held, book, cash = {}, [], float(CAP_TOTAL)
    for (ts,), g in sorted(by_day.items(), key=lambda kv: kv[0][0]):
        for k in [k for k, v in held.items() if v["x_ts"] <= ts]:
            v = held.pop(k)
            cash += v["size"]
        for r in g.iter_rows(named=True):
            if r["code"] in held or cash < 1_000:
                continue
            size = min(cash, CAP_ONE, ADV_FRAC * r["amt20"])
            if size < 1_000:
                continue
            cost = 2 * max(size * RATE, min_fee) / size + 2 * SLIP
            net = r[col] - cost
            held[r["code"]] = dict(x_ts=r["x_ts"], size=size)
            cash -= size
            book.append(dict(code=r["code"], ts=ts, x_ts=r["x_ts"], days=r["days"], sig=r["sig"], gross=r[col], net=net,
                             size=size, rmb=net * size, x_prem=r["x_prem"]))
    b = pl.DataFrame(book)
    if b.is_empty():
        return dict(trades=0, rmb_total=0.0, rmb_per_year=0.0)
    years = (b["x_ts"].max() - b["ts"].min()).days / 365
    # 按持有天数把每笔收益摊到各个日历年（盯市仓位的收益不会全部堆到 2026 年）
    spread = []
    for r in b.iter_rows(named=True):
        n = max((r["x_ts"] - r["ts"]).days, 1)
        for y in range(r["ts"].year, r["x_ts"].year + 1):
            lo = max(r["ts"], r["ts"].replace(year=y, month=1, day=1))
            hi = min(r["x_ts"], r["ts"].replace(year=y + 1, month=1, day=1))
            spread.append(dict(year=y, rmb=r["rmb"] * max((hi - lo).days, 0) / n))
    yearly = pl.DataFrame(spread).group_by("year").agg(pl.col("rmb").sum()).sort("year")
    top = b.group_by("code").agg(pl.col("rmb").sum()).sort("rmb", descending=True)
    used = (b["size"] * (b["x_ts"] - b["ts"]).dt.total_days()).sum() / 365 / years / CAP_TOTAL
    return dict(trades=b.height, funds=b["code"].n_unique(), net_mean=b["net"].mean(), net_med=b["net"].median(),
                win=(b["net"] > 0).mean(), rmb_total=b["rmb"].sum(), years=years, rmb_per_year=b["rmb"].sum() / years,
                util=used, top5_share=top["rmb"].head(5).sum() / top["rmb"].sum() if top["rmb"].sum() else None,
                by_year={str(y): round(v) for y, v in zip(yearly["year"], yearly["rmb"])}, book=b)


def main() -> None:
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_float_precision(4)
    p = panel()

    # 表 1：开放公告前后的折价路径（中位数）
    o = opens()
    p_i = p.with_columns(pl.int_range(pl.len()).over("code").alias("i"))
    anchor = o.sort("open_ts").join_asof(p_i.select("code", "ts", "i").sort("ts"), left_on="open_ts", right_on="ts",
                                         by="code", strategy="forward").drop_nulls("i").select("code", "open_ts", pl.col("i").alias("i0"))
    path = p_i.join(anchor, on="code").with_columns((pl.col("i") - pl.col("i0")).alias("k")).filter(
        pl.col("k").is_in([-250, -120, -60, -20, -10, -5, -1, 0, 1, 2, 3, 5, 10, 20]))
    print("表 1 开放公告日（k=0）前后的场内溢价（负数为折价）\n",
          path.group_by("k").agg(pl.len().alias("n"), pl.col("prem").median().alias("med"),
                                 pl.col("prem").quantile(0.25).alias("q25"), pl.col("prem").quantile(0.75).alias("q75")).sort("k"))

    rows = []
    for lag, col, mtm in [(5, "gross", True), (5, "gross", False), (1, "gross", True), (10, "gross", True),
                          (5, "gross_redeem", True)]:
        t = trades(p, lag, mtm)
        for theta, mn, (lo, hi) in itertools.product([0.03, 0.06, 0.10], [5.0], [(20, 800), (60, 400)]):
            m = simulate(t, theta, lo, hi, mn, col)
            book = m.pop("book", None)
            params = dict(exit_lag=lag, exit=col, mtm=mtm, theta=theta, min_fee=mn, min_days=lo, max_days=hi,
                          cap_total=CAP_TOTAL, cap_one=CAP_ONE, adv_frac=ADV_FRAC, slip=SLIP)
            trials.record("d01_cfund_discount", params=params, metrics=m, sample="in_sample",
                          note="OPP-004 快速否证 v3：盯市仓位不得按净值赎回；年度按持有天数分摊")
            rows.append(params | {k: v for k, v in m.items() if k != "by_year"}
                        | {f"y{k}": v for k, v in m.get("by_year", {}).items()})
            if (lag, col, mtm, theta, lo) == (5, "gross", True, 0.03, 20):
                OUT.mkdir(parents=True, exist_ok=True)
                book.write_parquet(OUT / "book_base.parquet")
    r = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.write_parquet(OUT / "scenarios.parquet")
    print("\n表 2 每年超额（元），资金 10 万，G0 = 3,000 元\n",
          r.select("exit_lag", "exit", "mtm", "theta", "min_days", "max_days", "trades", "funds", "net_med", "win",
                   "rmb_per_year", "util", "top5_share"))
    print("\n表 3 按持有天数分摊到各年（元；2021 年从 1 月起、2026 年只到 4 月）\n", r.select("exit_lag", "exit", "mtm", "theta", "min_days", *[c for c in r.columns if c.startswith("y")]))


if __name__ == "__main__":
    main()
