"""可转债双低 + 事件规则：前向记录（纸面，不下单，不接券商）。

参数和规则冻结在 configs/strategies/cb_rotation.yaml。规则代码直接调用回测的 cb01_double_low.run()，
和样本内用的是同一份代码；启动时校验代码哈希，不一致就拒绝运行。

每个交易日 15:30（北京时间）之后运行一次：
  1. 采集当天收盘快照：新浪转债行情（收盘价、成交量、成交额）+ 东方财富转债列表（转股溢价率、正股价、上市日）
  2. 采集巨潮"向下修正""提前赎回"公告
  3. 从前向起点开始用全部快照重放一遍 run()，写出净值、成交和当前持仓
没有快照的交易日（Mac 休眠、断网）记入 gaps.csv，不插值、不补采。

数据放在 data/forward/cb/，评估日（cb_rotation.yaml 的 forward.evaluate_after）之前与封存段同等对待：
只采集和重放，不做业绩分析，本脚本也不打印收益。

用法：
  uv run python research/forward/cb_forward.py bootstrap   # 一次性：取存续转债起点前 30 个交易日行情，用于 20 日均成交额和上市天数
  uv run python research/forward/cb_forward.py daily       # 每个交易日收盘后（launchd 调用）
"""

import hashlib
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd
import polars as pl
import yaml

from lab.paths import REPO_ROOT, data_root

sys.path[:0] = [str(REPO_ROOT / "research" / "analysis"), str(REPO_ROOT / "research" / "ingest")]
import cb01_double_low as c1  # noqa: E402
from cb_notices import KEYWORDS, classify  # noqa: E402

CST = ZoneInfo("Asia/Shanghai")
CFG_PATH = REPO_ROOT / "configs" / "strategies" / "cb_rotation.yaml"
GAP = 20  # 与 cb03_events 相同：同一转债同一类事件 20 个交易日内只取第一条


def cfg() -> dict:
    c = yaml.safe_load(CFG_PATH.read_text())
    for path, sha in c["code_sha256"].items():
        got = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        if got != sha:
            sys.exit(f"{path} 已被修改（{got[:12]} ≠ 冻结的 {sha[:12]}），前向记录拒绝运行")
    r, k = c["rules"], c["costs"]
    frozen = (c["capital"], r["px_floor"], r["stock_floor"], r["min_amt20"], k["commission"], k["exchange_fee"],
              k["slip"], k["slip_illiquid"], r["ev_hold"])
    code = (c1.CAPITAL, c1.PX_FLOOR, c1.STOCK_FLOOR, c1.MIN_AMT20, c1.RATE, c1.EXCH, c1.SLIP, c1.SLIP_ILLIQ, c1.EV_HOLD)
    if frozen != code:
        sys.exit(f"cb_rotation.yaml 与 cb01_double_low.py 的常量不一致：{frozen} ≠ {code}")
    return c


def root() -> Path:
    return data_root() / "forward" / "cb"


def to_ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 7, tzinfo=UTC)  # 收盘 15:00 北京时间，与回测口径相同


def trade_days() -> list[date]:
    return sorted(pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"]).dt.date)


def code_of(symbol: str) -> str:
    return symbol[2:]


# ---------- 采集 ----------

def cmd_bootstrap() -> None:
    c = cfg()
    start = date.fromisoformat(str(c["forward"]["start"]))
    spot = ak.bond_zh_hs_cov_spot()
    spot = spot[spot["symbol"].str.match(r"^(sh11|sz12)") & (spot["trade"].astype(float) > 0)]
    rows = []
    for i, sym in enumerate(spot["symbol"]):
        try:
            d = ak.bond_zh_hs_cov_daily(symbol=sym)
        except Exception as e:
            print(f"  失败 {sym}：{e!r}"[:120], flush=True)
            continue
        d["date"] = pd.to_datetime(d["date"]).dt.date
        d = d[d["date"] < start].tail(30)
        rows.append(pd.DataFrame({"bond_code": code_of(sym), "date": d["date"], "close": d["close"].astype(float),
                                  "volume": d["volume"].astype(float)}))
        time.sleep(0.3)
        if i % 100 == 0:
            print(f"bootstrap {i}/{len(spot)}", flush=True)
    out = pd.concat(rows)
    root().mkdir(parents=True, exist_ok=True)
    out.to_parquet(root() / "bootstrap.parquet")
    print("bootstrap 完成：", out["bond_code"].nunique(), "只转债")


def snapshot(today: date) -> None:
    spot = ak.bond_zh_hs_cov_spot()
    spot = spot[spot["symbol"].str.match(r"^(sh11|sz12)")].copy()
    spot["bond_code"] = spot["symbol"].map(code_of)
    em = ak.bond_zh_cov().rename(columns={
        "债券代码": "bond_code", "债券简称": "bond_name", "正股代码": "stock_code", "转股溢价率": "conv_premium",
        "正股价": "stock_close", "上市时间": "list_date"})
    em["bond_code"] = em["bond_code"].astype(str)
    s = spot[["bond_code", "trade", "volume", "amount"]].merge(
        em[["bond_code", "bond_name", "stock_code", "conv_premium", "stock_close", "list_date"]], on="bond_code", how="left")
    s = s.rename(columns={"trade": "close"})
    s["date"] = today
    (root() / "snapshots").mkdir(parents=True, exist_ok=True)
    s.to_parquet(root() / "snapshots" / f"{today}.parquet")
    print("快照", today, len(s), "只，有成交", int((s["close"].astype(float) > 0).sum()))


def notices(today: date, since: date) -> None:
    frames = []
    for tag, kw in KEYWORDS.items():
        d = ak.stock_zh_a_disclosure_report_cninfo(symbol="", market="沪深京", keyword=kw, category="",
                                                    start_date=since.strftime("%Y%m%d"), end_date=today.strftime("%Y%m%d"))
        d["tag"] = tag
        frames.append(d)
        time.sleep(1)
    n = pd.concat(frames)
    (root() / "notices").mkdir(parents=True, exist_ok=True)
    n.to_parquet(root() / "notices" / f"{today}.parquet")
    print("公告", today, len(n), "条（含重复和无关）")


def record_gaps(start: date, today: date, cal: list[date]) -> list[date]:
    have = {date.fromisoformat(p.stem) for p in (root() / "snapshots").glob("*.parquet")}
    gaps = [d for d in cal if start <= d <= today and d not in have]
    pd.DataFrame({"missing_trade_day": gaps}).to_csv(root() / "gaps.csv", index=False)
    return gaps


# ---------- 重放 ----------

def history(start: date) -> tuple[pl.DataFrame, pl.DataFrame]:
    boot = pl.read_parquet(root() / "bootstrap.parquet").with_columns(
        (pl.col("close") * pl.col("volume")).alias("amount"), pl.lit(False).alias("snap"))
    snaps = pl.concat([pl.read_parquet(p) for p in sorted((root() / "snapshots").glob("*.parquet"))],
                      how="vertical_relaxed").with_columns(
        pl.col("close").cast(pl.Float64), pl.col("volume").cast(pl.Float64), pl.col("amount").cast(pl.Float64),
        pl.col("conv_premium").cast(pl.Float64), pl.col("stock_close").cast(pl.Float64))
    snaps = snaps.filter(pl.col("close") > 0)  # 停牌或没有成交，与回测里"当天没有行情"一致
    h = pl.concat([boot.select("bond_code", "date", "close", "amount", "snap"),
                   snaps.select("bond_code", "date", "close", "amount", pl.lit(True).alias("snap"))])
    h = h.sort("bond_code", "date").with_columns(
        pl.col("amount").rolling_mean(20, min_samples=5).over("bond_code").alias("amt20"),
        pl.int_range(pl.len()).over("bond_code").alias("age"),
    ).filter(pl.col("snap") & (pl.col("date") >= start))
    return h.join(snaps.select("bond_code", "date", "conv_premium", "stock_close"), on=["bond_code", "date"]), snaps


def events(days: list[date], snaps: pl.DataFrame) -> tuple[dict, dict]:
    files = sorted((root() / "notices").glob("*.parquet"))
    if not files:
        return {}, {}
    n = pd.concat([pd.read_parquet(p) for p in files]).rename(
        columns={"代码": "stock_code", "公告标题": "title", "公告时间": "ann_time"})
    n["title"] = n["title"].str.replace(r"</?em>", "", regex=True)
    n["kind"] = [classify(t, g) for t, g in zip(n["title"], n["tag"])]
    n = n[n["kind"].isin(["call_yes", "call_reminder", "down_propose"])].drop_duplicates(["stock_code", "title", "ann_time"])
    if n.empty:
        return {}, {}
    names = snaps.select("bond_code", "bond_name", "stock_code").unique().to_pandas()
    rows = []
    for r in n.itertuples():
        cand = names[names["stock_code"] == r.stock_code]
        hit = cand[[str(b) in r.title for b in cand["bond_name"]]]
        pick = hit if len(hit) else (cand if len(cand) == 1 else cand.iloc[0:0])
        t = pd.Timestamp(str(r.ann_time) if len(str(r.ann_time)) > 10 else f"{r.ann_time} 00:00:00")
        after_close = (t.hour, t.minute, t.second) != (0, 0, 0) and t.hour >= 15
        from_day = t.date() + timedelta(days=1) if after_close else t.date()
        if from_day < days[0]:  # 前向起点之前的公告不算事件
            continue
        d0 = next((d for d in days if d >= from_day), None)
        for b in pick["bond_code"]:
            if d0 is not None:
                rows.append((b, r.kind if r.kind != "call_reminder" else "call_yes", days.index(d0)))
    rows.sort()
    calls, downs, last = {}, {}, {}
    for b, kind, i in rows:
        if (b, kind) in last and i - last[(b, kind)] <= GAP:
            continue
        last[(b, kind)] = i
        (calls if kind == "call_yes" else downs).setdefault(to_ts(days[i]), set()).add(b)
    return calls, downs


def replay(c: dict) -> None:
    start = date.fromisoformat(str(c["forward"]["start"]))
    h, snaps = history(start)
    days = sorted(h["date"].unique().to_list())
    by_day = {to_ts(d): {r[0]: (r[1], r[2]) for r in g.select("bond_code", "close", "amt20").iter_rows()}
              for (d,), g in h.group_by(["date"])}
    r = c["rules"]
    elig = h.filter((pl.col("close") >= r["px_floor"]) & (pl.col("stock_close") >= r["stock_floor"])
                    & (pl.col("amt20") >= r["min_amt20"]) & (pl.col("age") >= r["min_age"])
                    & pl.col("conv_premium").is_not_null()).with_columns(
        (pl.col("close") + pl.col("conv_premium")).alias("score"))
    rk = {to_ts(d): g.sort("score", "bond_code")["bond_code"].to_list() for (d,), g in elig.group_by(["date"])}
    ranks = {r["px_floor"]: {to_ts(d): rk.get(to_ts(d), []) for d in days}}
    # 连续 3 个快照日没出现的视为已退出（强赎、到期或退市），按最后收盘价处理，与回测的退出口径一致
    seen = h.group_by("bond_code").agg(pl.col("date").max().alias("last"), pl.col("close").last().alias("lc"))
    recent = set(days[-3:])
    info = {b: (to_ts(l), lc, l not in recent) for b, l, lc in seen.iter_rows()}
    eq, ev, book = c1.run([to_ts(d) for d in days], by_day, ranks, info, r["freq"], r["n"], r["buffer"],
                          min_fee=0.0, recovery=c["costs"]["default_recovery"], px_floor=r["px_floor"],
                          notices=events(days, snaps), call_exit=r["call_exit"], down_buy=r["down_buy"])
    out = root() / "out"
    out.mkdir(parents=True, exist_ok=True)
    eq.write_parquet(out / "nav.parquet")
    ev.write_parquet(out / "trades.parquet")
    pl.DataFrame({"bond_code": list(book.pos), "zhang": list(book.pos.values())},
                 schema={"bond_code": pl.String, "zhang": pl.Int64}).write_csv(out / "positions.csv")
    today_trades = ev.filter(pl.col("ts") == to_ts(days[-1])).height if ev.height else 0
    print(f"重放完成：{len(days)} 个快照日，持仓 {len(book.pos)} 只，今日成交 {today_trades} 笔（收益不在评估日前显示）")


def cmd_daily() -> None:
    c = cfg()
    now = datetime.now(CST)
    today = now.date()
    cal = trade_days()
    start = date.fromisoformat(str(c["forward"]["start"]))
    if today not in cal:
        print(today, "不是交易日")
        return
    if (now.hour, now.minute) < (15, 30):
        print(now.strftime("%H:%M"), "早于 15:30，行情可能不是收盘价，不采集")
        return
    if not (root() / "snapshots" / f"{today}.parquet").exists():
        snapshot(today)
        prev = [d for d in cal if d < today]
        notices(today, max(start - timedelta(days=7), prev[-1] if prev else today))
    gaps = record_gaps(start, today, cal)
    if gaps:
        print("缺口交易日：", [str(d) for d in gaps])
    replay(c)


if __name__ == "__main__":
    {"bootstrap": cmd_bootstrap, "daily": cmd_daily}[sys.argv[1]]()
