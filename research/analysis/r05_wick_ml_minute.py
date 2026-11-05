"""OPP-006 第 2 轮：分钟级机器学习（D1 低杠杆接针、D2 顺势跟随、D3 波动时段 + 深挂单），只用样本内。

方案和停止条件见 docs/research/OPP006-ml-plan.md（看结果之前写定）。这里的训练流程（run_direction）也供 r06 的 D4 复用。

数据：OKX 永续 1 分钟 K 线 okx_swap_1m、资金费 okx_funding（BTC、ETH）。
时间切分：训练 2022-01～2024-06（内部 4 折半年滚动交叉验证，清掉标签重叠并隔 1 天），
验证 2024-07～2025-06（概率校准、选阈值、筛特征、选止损档），测试 2025-07～2026-04（冻结后只跑一次）。
方案写的是 5 折；训练段从 2022-01 开始，按半年滚动只能切出 4 折，如实记录。

用法：uv run python research/analysis/r05_wick_ml_minute.py
"""

import itertools
import json
from datetime import UTC, datetime, timedelta

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lab import lake, trials
from lab.paths import data_root
from r04_wick import MAJORS, one_at_a_time

OUT = data_root() / "research" / "r05_wick_ml"
EXPERIMENT = "r05_wick_ml"
VAL_START = datetime(2024, 7, 1, tzinfo=UTC)
TEST_START = datetime(2025, 7, 1, tzinfo=UTC)
TEST_END = datetime(2026, 5, 1, tzinfo=UTC)
TEST_YEARS = 10 / 12
FOLDS = [(datetime(y, m, 1, tzinfo=UTC), datetime(y2, m2, 1, tzinfo=UTC))
         for (y, m), (y2, m2) in [((2022, 7), (2023, 1)), ((2023, 1), (2023, 7)), ((2023, 7), (2024, 1)), ((2024, 1), (2024, 7))]]
GRID = [dict(num_leaves=a, min_child_samples=b, colsample_bytree=c, reg_lambda=d)
        for a, b, c, d in itertools.product([7, 15], [30, 80], [0.6, 0.9], [1.0, 10.0])]
LR_C = [0.01, 0.1, 1.0]
SEEDS = [0, 1, 2, 3, 4]
MIN_TRADES = 20
ALPHA = 0.1 / 4  # 4 个方向，单侧显著性门槛按方向数收紧
COST, ENTRY_SLIP, STOP_SLIP = 0.002, 0.0005, 0.001
COST_D, ENTRY_SLIP_D, STOP_SLIP_D = 0.003, 0.001, 0.002  # 延迟、滑点加倍的敏感性
MINUTE_STOPS = [0.02, 0.03, 0.05]
HORIZON = 1440
CAPITAL_RMB, RISK = 300_000, 0.01


def breakeven(s: float, cost: float = COST + ENTRY_SLIP, stop_slip: float = STOP_SLIP) -> float:
    return (s + stop_slip + cost) / (3 * s + stop_slip)


# ---------------- 分钟特征 ----------------

def minute_frame(inst: str) -> pl.DataFrame:
    df = (lake.scan("okx_swap_1m").filter(pl.col("inst_id") == inst)
          .select("ts", "open", "high", "low", "close", "vol_quote").collect().sort("ts"))
    fund = (lake.scan("okx_funding").filter(pl.col("inst_id") == inst)
            .select("ts", pl.col("funding_rate").alias("funding")).collect().sort("ts"))
    ref = pl.when(pl.col("ts").diff() == pl.duration(minutes=1)).then(pl.col("close").shift())
    vol = pl.col("vol_quote").fill_null(0.0)
    med1440 = vol.rolling_median(1440).shift(1)
    df = df.with_columns(ref.alias("ref")).with_columns(
        (1 - pl.col("low") / pl.col("ref")).alias("dn"),
        (pl.col("high") / pl.col("ref") - 1).alias("up"),
        (pl.col("close") / pl.col("ref") - 1).alias("mv"),
        (pl.col("close") / pl.col("ref")).log().alias("r1"),
    ).with_columns(
        pl.col("r1").rolling_std(60).shift(1).alias("rv60"),
        pl.col("r1").rolling_std(1440).shift(1).alias("rv1440"),
        *(((pl.col("close") / pl.col("close").shift(k)) - 1).alias(f"ret{k}") for k in (5, 15, 60, 240, 1440)),
        (vol / vol.rolling_median(60).shift(1)).alias("volr60"),
        (vol / med1440).alias("volr1440"),
        (vol.rolling_sum(5).shift(1) / (5 * med1440)).alias("volr_prev5"),
        (vol.rolling_sum(60).shift(1) / (60 * med1440)).alias("volr_prev60"),
        (pl.col("close") / pl.col("high").rolling_max(1440) - 1).alias("dist_hi"),
        (pl.col("close") / pl.col("low").rolling_min(1440) - 1).alias("dist_lo"),
        (pl.max_horizontal("dn", "up") >= 0.01).cast(pl.Int32).rolling_sum(1440).shift(1).alias("ev24"),
        pl.col("ts").dt.hour().alias("hour"),
        pl.col("ts").dt.weekday().alias("wday"),
        (480 - (pl.col("ts").dt.hour() * 60 + pl.col("ts").dt.minute()) % 480).alias("min_to_funding"),
    )
    df = df.join_asof(fund, on="ts", strategy="backward")
    return df.with_columns(pl.lit(inst).alias("inst_id"), pl.lit(int(inst.startswith("ETH"))).alias("is_eth"))


def with_other(frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    out = {}
    for inst, df in frames.items():
        other = next(f for k, f in frames.items() if k != inst)
        out[inst] = df.join(other.select("ts", pl.col("mv").alias("oth_mv"), pl.col("dn").alias("oth_dn"),
                                         pl.col("up").alias("oth_up"), pl.col("ret60").alias("oth_ret60")),
                            on="ts", how="left")
    return out


BASE_FEATS = ["dev", "dev_z", "rec", "mv", "rv60", "rv1440", "ret5", "ret15", "ret60", "ret240", "ret1440",
              "volr60", "volr1440", "volr_prev5", "volr_prev60", "dist_hi", "dist_lo", "ev24", "funding",
              "hour", "wday", "min_to_funding", "is_eth", "oth_mv", "oth_dn", "oth_up", "oth_ret60"]


def barrier(o, h, lo, c, ts, i, direction, s, entry_slip, cost, stop_slip, horizon):
    """第 i 根开盘进场，先到 +2s 记 1；先到 −s 或到期记 0；同一根两边都碰到算止损。"""
    j = i + horizon
    if j >= len(c) or ts[j] - ts[i] != np.timedelta64(horizon, "m"):
        return None
    e = o[i] * (1 + direction * entry_slip)
    hi, lw = h[i:j + 1], lo[i:j + 1]
    if direction == 1:
        stop, tp = lw <= e * (1 - s), hi >= e * (1 + 2 * s)
    else:
        stop, tp = hi >= e * (1 + s), lw <= e * (1 - 2 * s)
    fs = int(np.argmax(stop)) if stop.any() else horizon + 1
    ft = int(np.argmax(tp)) if tp.any() else horizon + 1
    if fs <= ft and fs <= horizon:
        return 0, -s - stop_slip - cost, ts[i + fs]
    if ft <= horizon:
        return 1, 2 * s - cost, ts[i + ft]
    return 0, direction * (c[j] / e - 1) - cost, ts[j]


def minute_events(df: pl.DataFrame, side: str, direction: int, stops: list[float]) -> pl.DataFrame:
    """side：急动方向（down / up）；direction：交易方向（1 做多，−1 做空）。"""
    dev = pl.col("dn") if side == "down" else pl.col("up")
    rec = ((pl.col("close") - pl.col("low")) / (pl.col("ref") - pl.col("low")) if side == "down"
           else (pl.col("high") - pl.col("close")) / (pl.col("high") - pl.col("ref")))
    d = df.with_row_index("i").with_columns(dev.alias("dev"), rec.alias("rec"))
    cand = d.filter(pl.col("dev") >= 0.01, pl.col("ref").is_not_null())
    keep = one_at_a_time(cand["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]"), 9)
    cand = cand.filter(pl.Series(keep)).with_columns((pl.col("dev") / pl.col("rv60")).alias("dev_z"))
    ts = df["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]")
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    rows = []
    for i in cand["i"].to_numpy():
        r = {"i": i}
        ok = True
        for s in stops:
            a = barrier(o, h, lo, c, ts, i + 1, direction, s, ENTRY_SLIP, COST, STOP_SLIP, HORIZON)
            b = barrier(o, h, lo, c, ts, i + 2, direction, s, ENTRY_SLIP_D, COST_D, STOP_SLIP_D, HORIZON)
            if a is None or b is None:
                ok = False
                break
            r.update({f"y_{s}": a[0], f"net_{s}": a[1], f"exit_{s}": int(a[2].astype("int64")), f"netd_{s}": b[1]})
        if ok:
            rows.append(r)
    lab = pl.DataFrame(rows).with_columns(
        *(pl.from_epoch(pl.col(f"exit_{s}") * 60, time_unit="s").dt.replace_time_zone("UTC").alias(f"exit_{s}") for s in stops))
    return cand.join(lab, on="i").drop("i")


# ---------------- 训练流程（D1～D4 共用） ----------------

def day_weights(ts: pl.Series) -> np.ndarray:
    day = ts.dt.date()
    n = day.to_frame("d").group_by("d").len()
    return 1 / np.sqrt(day.to_frame("d").join(n, on="d", how="left")["len"].to_numpy())


def lgb_fit(cfg, X, y, w, Xv=None, yv=None, wv=None, n_estimators=1000, seed=0):
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, n_estimators=n_estimators,
                           subsample=0.8, subsample_freq=1, random_state=seed, deterministic=True,
                           force_col_wise=True, verbose=-1, **cfg)
    if Xv is not None:
        m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_sample_weight=[wv],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(X, y, sample_weight=w)
    return m


def lr_fit(C, X, y, w, med):
    sc = StandardScaler().fit(np.nan_to_num(X, nan=0.0) if med is None else np.where(np.isnan(X), med, X))
    Xs = sc.transform(np.where(np.isnan(X), med, X))
    m = LogisticRegression(C=C, max_iter=5000).fit(Xs, y, sample_weight=w)
    return lambda Z: m.predict_proba(sc.transform(np.where(np.isnan(Z), med, Z)))[:, 1]


def top_mean(p, net, q=0.3):
    if len(p) < 5:
        return np.nan
    k = max(1, int(len(p) * q))
    return float(np.mean(net[np.argsort(-p)[:k]]))


def cv_score(ev, feats, ycol, netcol, fit_predict, metric):
    """4 折半年滚动：用折起点之前、而且标签在折起点前 1 天结束的样本训练。"""
    scores, iters = [], []
    for start, end in FOLDS:
        tr = ev.filter(pl.col("exit") < start - timedelta(days=1))
        va = ev.filter(pl.col("ts") >= start, pl.col("ts") < end)
        if tr.height < 50 or va.height < 10 or tr[ycol].n_unique() < 2:
            continue
        p, it = fit_predict(tr, va)
        iters.append(it)
        if metric == "auc":
            scores.append(roc_auc_score(va[ycol], p) if va[ycol].n_unique() > 1 else np.nan)
        else:
            scores.append(top_mean(p, va[netcol].to_numpy()))
    return float(np.nanmean(scores)) if scores else np.nan, iters


def plateau_pick(scores: list[float]) -> int:
    """每组参数和它只差一个参数的 4 组邻居取平均，选平均最高的（选高原，不选尖峰）。"""
    keys = [tuple(c.values()) for c in GRID]
    best, best_i = -np.inf, 0
    for i, k in enumerate(keys):
        nb = [j for j, k2 in enumerate(keys) if sum(a != b for a, b in zip(k, k2)) <= 1]
        v = np.nanmean([scores[j] for j in nb])
        if v > best:
            best, best_i = v, i
    return best_i


def choose_threshold(p, net, min_n, grid=np.round(np.arange(0.05, 0.96, 0.05), 2)):
    """验证段：每笔平均净收益最高、且阈值 ±0.05 都为正、笔数 ≥ min_n 的阈值；没有就返回 None。"""
    stats = {t: (int((p >= t).sum()), float(net[p >= t].mean()) if (p >= t).any() else np.nan) for t in grid}
    best, best_t = -np.inf, None
    for t in grid:
        n, m = stats[t]
        nb = [stats[u][1] for u in grid if abs(u - t) <= 0.051]
        if n >= min_n and all(x > 0 for x in nb if not np.isnan(x)) and m > best:
            best, best_t = m, float(t)
    return best_t


def calibrate(p_val, y_val):
    if len(p_val) >= 500:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y_val)
        return iso.predict
    lr = LogisticRegression(C=1e6).fit(np.log(np.clip(p_val, 1e-6, 1 - 1e-6) / (1 - np.clip(p_val, 1e-6, 1 - 1e-6))).reshape(-1, 1), y_val)
    return lambda q: lr.predict_proba(np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6))).reshape(-1, 1))[:, 1]


def boot_p(days_a: pl.DataFrame, days_b: pl.DataFrame, n=5000, seed=0) -> float:
    """按天重抽样：模型总净收益 − 简单规则总净收益 ≤ 0 的比例（单侧 p）。"""
    d = days_a.join(days_b, on="d", how="full", coalesce=True).fill_null(0.0)
    if d.height == 0:
        return 1.0
    diff = (d["a"] - d["b"]).to_numpy()
    rng = np.random.default_rng(seed)
    sums = diff[rng.integers(0, len(diff), (n, len(diff)))].sum(axis=1)
    return float((sums <= 0).mean())


def per_day(ts: pl.Series, net: np.ndarray, col: str) -> pl.DataFrame:
    return pl.DataFrame({"d": ts.dt.date(), col: net}).group_by("d").agg(pl.col(col).sum())


def drop_top_days(ts: pl.Series, net: np.ndarray, k=5) -> float:
    d = per_day(ts, net, "x").sort("x", descending=True)
    return float(d["x"].sum() - d.head(k)["x"].sum())


def run_direction(name: str, ev: pl.DataFrame, feats: list[str], stops: list[float], notional_rmb, extra: dict,
                  label_kind: str = "barrier", strat=None) -> dict:
    """一个方向的完整流程：交叉验证选参数 → 训练 → 筛特征 → 校准 → 选阈值和止损档 → 防泄漏检查 → 测试期一次性判定。

    label_kind="barrier"：ev 里有 y_s / net_s / netd_s / exit_s（D1、D2、D4）。
    label_kind="regime"：D3，ev 只有 y（未来 4 小时出深针）和 exit；strat 负责把概率换成挂单交易并算钱。
    """
    res = {"direction": name, "n_events": ev.height, "stops": {}}
    # 除以 0 产生的无穷大当缺失处理（LightGBM 原生处理缺失，逻辑回归用训练段中位数填）
    ev = ev.with_columns(pl.col(f).cast(pl.Float64).replace([np.inf, -np.inf], None).alias(f) for f in feats)
    train = ev.filter(pl.col("ts") < VAL_START)
    val = ev.filter(pl.col("ts") >= VAL_START, pl.col("ts") < TEST_START)
    test = ev.filter(pl.col("ts") >= TEST_START, pl.col("ts") < TEST_END)
    res["n_train_val_test"] = [train.height, val.height, test.height]
    for s in stops:
        ycol, netcol = (f"y_{s}", f"net_{s}") if label_kind == "barrier" else ("y", None)
        tag = f"{name}_s{s}" if label_kind == "barrier" else name
        evs = ev.with_columns(pl.col(f"exit_{s}" if label_kind == "barrier" else "exit").alias("exit"))
        trs = evs.filter(pl.col("ts") < VAL_START)
        metric = "auc" if label_kind == "regime" else "top30"
        X = lambda f: f.select(feats).to_numpy().astype(float)  # noqa: E731

        # 1) LightGBM 网格：交叉验证
        cv = []
        for cfg in GRID:
            def fp(tr, va, cfg=cfg):
                m = lgb_fit(cfg, X(tr), tr[ycol].to_numpy(), day_weights(tr["ts"]), X(va), va[ycol].to_numpy(),
                            day_weights(va["ts"]))
                return m.predict_proba(X(va))[:, 1], m.best_iteration_ or 1000
            score, iters = cv_score(trs, feats, ycol, netcol, fp, metric)
            cv.append((score, iters))
            trials.record(EXPERIMENT, variant=f"{tag}_lgbm_cv", sample="in_sample",
                          params={**cfg, "stop": s, "feats": len(feats), **extra}, metrics={"cv": score, "iters": iters},
                          note="OPP-006 第 2 轮交叉验证（只用训练段）")
        k = plateau_pick([c[0] for c in cv])
        cfg = GRID[k]
        n_est = int(np.median(cv[k][1])) if cv[k][1] else 200

        # 2) 逻辑回归：交叉验证
        med = np.nanmedian(X(trs), axis=0)
        lr_cv = []
        for C in LR_C:
            def fp(tr, va, C=C):
                return lr_fit(C, X(tr), tr[ycol].to_numpy(), day_weights(tr["ts"]), med)(X(va)), 0
            score, _ = cv_score(trs, feats, ycol, netcol, fp, metric)
            lr_cv.append(score)
            trials.record(EXPERIMENT, variant=f"{tag}_lr_cv", sample="in_sample",
                          params={"C": C, "stop": s, **extra}, metrics={"cv": score}, note="OPP-006 第 2 轮交叉验证")
        C = LR_C[int(np.nanargmax(lr_cv))] if not all(np.isnan(lr_cv)) else 0.1

        # 3) 训练段全量训练（5 个种子取平均），验证段置换重要性筛特征，再重训
        def fit_avg(fs):
            Xt = trs.select(fs).to_numpy().astype(float)
            ms = [lgb_fit(cfg, Xt, trs[ycol].to_numpy(), day_weights(trs["ts"]), n_estimators=max(n_est, 20), seed=sd)
                  for sd in SEEDS]
            return lambda f: np.mean([m.predict_proba(f.select(fs).to_numpy().astype(float))[:, 1] for m in ms], axis=0)
        pred = fit_avg(feats)
        yv, wv = val[ycol].to_numpy(), day_weights(val["ts"])
        base = log_loss(yv, np.clip(pred(val), 1e-6, 1 - 1e-6), sample_weight=wv, labels=[0, 1])
        rng = np.random.default_rng(0)
        imp = {}
        for f in feats:
            losses = []
            for _ in range(3):
                vv = val.with_columns(pl.Series(f, rng.permutation(val[f].to_numpy())))
                losses.append(log_loss(yv, np.clip(pred(vv), 1e-6, 1 - 1e-6), sample_weight=wv, labels=[0, 1]))
            imp[f] = float(np.mean(losses) - base)
        sel = [f for f, v in sorted(imp.items(), key=lambda x: -x[1]) if v > 0][:20] or feats
        pred = fit_avg(sel)

        # 4) 校准、阈值（验证段）
        pv_raw = pred(val)
        cal = calibrate(pv_raw, yv)
        pv, pt = cal(pv_raw), cal(pred(test))
        auc_val = float(roc_auc_score(yv, pv_raw)) if len(set(yv)) > 1 else np.nan

        # 防泄漏：打乱训练段标签重训，验证段 AUC 应接近 0.5
        sh = trs.with_columns(pl.Series(ycol, rng.permutation(trs[ycol].to_numpy())))
        msh = lgb_fit(cfg, sh.select(sel).to_numpy().astype(float), sh[ycol].to_numpy(), day_weights(sh["ts"]),
                      n_estimators=max(n_est, 20))
        auc_shuffle = float(roc_auc_score(yv, msh.predict_proba(val.select(sel).to_numpy().astype(float))[:, 1])) \
            if len(set(yv)) > 1 else np.nan

        # 简单规则 M1：两个特征的分位数阈值，在验证段选
        rule_feats = extra.get("rule_feats", ["dev_z", "volr60"])
        qs = [0.3, 0.5, 0.7]
        th = {f: [float(trs[f].quantile(q)) for q in qs] for f in rule_feats}

        def rule_mask(f, a, b):
            m = np.ones(f.height, dtype=bool)
            for col, t in zip(rule_feats, (a, b)[:len(rule_feats)]):
                m &= (f[col].fill_null(-np.inf).to_numpy() >= t)
            return m

        combos = list(itertools.product(*[th[f] for f in rule_feats]))
        if len(rule_feats) == 1:
            combos = [(c[0], None) for c in combos]

        lrp = lr_fit(C, X(trs), trs[ycol].to_numpy(), day_weights(trs["ts"]), med)
        lr_cal = calibrate(lrp(X(val)), yv)

        if label_kind == "barrier":
            nv, nt, ntd = val[netcol].to_numpy(), test[netcol].to_numpy(), test[f"netd_{s}"].to_numpy()
            thr = choose_threshold(pv, nv, MIN_TRADES)
            thr_lr = choose_threshold(lr_cal(lrp(X(val))), nv, MIN_TRADES)
            best_rule, best_m = None, -np.inf
            for a, b in combos:
                msk = rule_mask(val, a, b)
                if msk.sum() >= MIN_TRADES and nv[msk].mean() > best_m:
                    best_m, best_rule = nv[msk].mean(), (a, b)
            sel_t = pt >= thr if thr is not None else np.zeros(test.height, bool)
            rule_t = rule_mask(test, *best_rule) if best_rule else np.zeros(test.height, bool)
            lr_t = lr_cal(lrp(X(test))) >= thr_lr if thr_lr is not None else np.zeros(test.height, bool)
            yt = test[ycol].to_numpy()
            n = int(sel_t.sum())
            money = float((nt[sel_t] * notional_rmb(s)).sum() / TEST_YEARS) if n else 0.0
            r = {
                "cfg": cfg, "n_estimators": n_est, "C": C, "feats_selected": sel, "auc_val": auc_val,
                "auc_shuffle": auc_shuffle, "threshold": thr, "val_n": int((pv >= thr).sum()) if thr else 0,
                "val_mean_bp": float(nv[pv >= thr].mean() * 1e4) if thr else None,
                "breakeven": breakeven(s),
                "test_auc": float(roc_auc_score(yt, pt)) if len(set(yt)) > 1 else None,
                "test_n": n, "test_win": float(yt[sel_t].mean()) if n else None,
                "test_mean_bp": float(nt[sel_t].mean() * 1e4) if n else None,
                "test_mean_delay_bp": float(ntd[sel_t].mean() * 1e4) if n else None,
                "test_drop5_bp_total": drop_top_days(test["ts"].filter(pl.Series(sel_t)), nt[sel_t]) * 1e4 if n else None,
                "test_annual_rmb": money,
                "m0_n": test.height, "m0_win": float(yt.mean()) if test.height else None,
                "m0_mean_bp": float(nt.mean() * 1e4) if test.height else None,
                "m1_rule": best_rule, "m1_n": int(rule_t.sum()),
                "m1_mean_bp": float(nt[rule_t].mean() * 1e4) if rule_t.any() else None,
                "m2_n": int(lr_t.sum()), "m2_mean_bp": float(nt[lr_t].mean() * 1e4) if lr_t.any() else None,
                "p_vs_m1": boot_p(per_day(test["ts"].filter(pl.Series(sel_t)), nt[sel_t], "a"),
                                  per_day(test["ts"].filter(pl.Series(rule_t)), nt[rule_t], "b")),
            }
            r["pass"] = {
                "1_win": n > 0 and r["test_win"] >= r["breakeven"] + 0.05,
                "2_mean_n": n >= MIN_TRADES and r["test_mean_bp"] > 0,
                "3_vs_m1": r["p_vs_m1"] < ALPHA,
                "4_drop5": n > 0 and r["test_drop5_bp_total"] > 0,
                "5_delay": n > 0 and r["test_mean_delay_bp"] > 0,
                "6_g0": money >= 3000,
            }
        else:
            r = strat(pred, cal, val, test, sel, rule_mask, combos, auc_val, auc_shuffle, cfg, n_est)
        r["all_pass"] = all(r["pass"].values())
        res["stops"][str(s)] = r
        trials.record(EXPERIMENT, variant=f"{tag}_final", sample="in_sample", params={**cfg, "stop": s, **extra},
                      metrics=r, note="OPP-006 第 2 轮：冻结后测试期一次性判定（测试期在样本内，封存段未动）")
        trials.record(EXPERIMENT, variant=f"{tag}_m1_rule", sample="in_sample",
                      params={"rule_feats": rule_feats, "quantiles": qs, "stop": s, **extra},
                      metrics={"rule": r.get("m1_rule"), "test_n": r.get("m1_n"), "test_mean_bp": r.get("m1_mean_bp")},
                      note="简单规则 M1：分位数阈值网格在验证段选，记为 1 行")
        print(json.dumps({"direction": name, "stop": s, **{k: v for k, v in r.items() if k != "feats_selected"}},
                         ensure_ascii=False, default=str), flush=True)
    # 止损档在验证段选（验证段每笔平均净收益最高、且有可行阈值的那一档），测试期只按这一档判定
    if label_kind == "barrier":
        ok = {s: r for s, r in res["stops"].items() if r["threshold"] is not None}
        res["chosen_stop"] = max(ok, key=lambda s: ok[s]["val_mean_bp"]) if ok else None
        res["pass"] = bool(res["chosen_stop"] and res["stops"][res["chosen_stop"]]["all_pass"])
    else:
        res["chosen_stop"] = next(iter(res["stops"]))
        res["pass"] = res["stops"][res["chosen_stop"]]["all_pass"]
    return res


# ---------------- D3：波动时段 + 深挂单 ----------------

def d3_dataset(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """每个整点：用上一分钟及之前的数据做特征，标签 = 未来 4 小时 BTC 或 ETH 出现 1 分钟跌幅 ≥ 2%。"""
    parts = []
    for inst, df in frames.items():
        p = "btc_" if inst.startswith("BTC") else "eth_"
        parts.append(df.select(
            "ts", *(pl.col(c).alias(p + c) for c in ("rv60", "rv1440", "ret60", "ret240", "ret1440", "volr_prev60",
                                                    "ev24", "dist_hi", "dist_lo", "funding")),
            (pl.col("dn") >= 0.02).alias(p + "deep"),
        ))
    m = parts[0].join(parts[1], on="ts", how="full", coalesce=True).sort("ts").with_columns(
        (pl.col("btc_deep").fill_null(False) | pl.col("eth_deep").fill_null(False)).cast(pl.Int32).alias("deep"))
    # 未来 240 分钟内是否有深针：反转后滚动求和
    m = m.with_columns(pl.col("deep").reverse().rolling_sum(240, min_samples=1).reverse().alias("fut240"))
    feats_cols = [c for c in m.columns if c.startswith(("btc_", "eth_")) and not c.endswith("deep")]
    # 整点 H 的特征取 H−1 分钟那一行（已收盘），标签取 [H, H+240)
    prev = m.select((pl.col("ts") + pl.duration(minutes=1)).alias("ts"), *feats_cols)
    lab = m.select("ts", (pl.col("fut240") > 0).cast(pl.Int32).alias("y"))
    h = lab.filter(pl.col("ts").dt.minute() == 0).join(prev, on="ts", how="inner")
    return h.with_columns(
        pl.col("ts").dt.hour().alias("hour"), pl.col("ts").dt.weekday().alias("wday"),
        (pl.col("ts") + pl.duration(minutes=240)).alias("exit"),
    )


def d3_fills() -> pl.DataFrame:
    """第 1 轮的深挂单（BTC/ETH 做多，挂单深度 2%，持有 60 分钟，悲观成本）。"""
    f = pl.read_parquet(data_root() / "research" / "r04_wick" / "fills_majors.parquet").filter(
        pl.col("level") == 0.02, pl.col("side") == "long", pl.col("pen") >= 0.001, pl.col("gross_60").is_not_null(),
    ).sort("inst_id", "ts")
    keep = np.concatenate([one_at_a_time(p["ts"].dt.replace_time_zone(None).to_numpy().astype("datetime64[m]"), 60)
                           for p in f.partition_by("inst_id", maintain_order=True)])
    return f.filter(pl.Series(keep)).select(
        "ts", "inst_id", (pl.col("gross_60") - 0.002).alias("net"), (pl.col("pen") >= 0.005).alias("strict"),
        pl.col("ts").dt.truncate("1h").alias("hour_ts"))


def d3_strategy(fills: pl.DataFrame, hourly: pl.DataFrame, notional_rmb: float):
    def strat(pred, cal, val, test, sel, rule_mask, combos, auc_val, auc_shuffle, cfg, n_est):
        def attach(seg):
            p = pl.DataFrame({"hour_ts": seg["ts"], "p": cal(pred(seg))})
            rules = {c: rule_mask(seg, *c) for c in combos}
            ff = fills.join(p, on="hour_ts", how="inner")
            idx = seg.select("ts").with_row_index("k").rename({"ts": "hour_ts"})
            ff = ff.join(idx, on="hour_ts", how="inner")
            return ff, rules
        fv, rv = attach(val)
        ft, rt = attach(test)
        nv = fv["net"].to_numpy()
        thr = choose_threshold(fv["p"].to_numpy(), nv, MIN_TRADES)
        best_rule, best_m = None, -np.inf
        for c in combos:
            msk = rv[c][fv["k"].to_numpy()]
            if msk.sum() >= MIN_TRADES and nv[msk].mean() > best_m:
                best_m, best_rule = nv[msk].mean(), c
        nt = ft["net"].to_numpy()
        sel_t = ft["p"].to_numpy() >= thr if thr is not None else np.zeros(ft.height, bool)
        rule_t = rt[best_rule][ft["k"].to_numpy()] if best_rule else np.zeros(ft.height, bool)
        n = int(sel_t.sum())
        strict = ft["strict"].to_numpy()
        r = {
            "cfg": cfg, "n_estimators": n_est, "feats_selected": sel, "auc_val": auc_val, "auc_shuffle": auc_shuffle,
            "test_auc": float(roc_auc_score(test["y"], cal(pred(test)))) if test["y"].n_unique() > 1 else None,
            "threshold": thr, "val_n": int((fv["p"].to_numpy() >= thr).sum()) if thr else 0,
            "val_mean_bp": float(nv[fv["p"].to_numpy() >= thr].mean() * 1e4) if thr else None,
            "m0_n": ft.height, "m0_mean_bp": float(nt.mean() * 1e4) if ft.height else None,
            "test_n": n, "test_mean_bp": float(nt[sel_t].mean() * 1e4) if n else None,
            "test_mean_strict_bp": float(nt[sel_t & strict].mean() * 1e4) if (sel_t & strict).any() else None,
            "test_drop5_bp_total": drop_top_days(ft["ts"].filter(pl.Series(sel_t)), nt[sel_t]) * 1e4 if n else None,
            "test_annual_rmb": float(nt[sel_t].sum() * notional_rmb / TEST_YEARS) if n else 0.0,
            "m1_rule": best_rule, "m1_n": int(rule_t.sum()),
            "m1_mean_bp": float(nt[rule_t].mean() * 1e4) if rule_t.any() else None,
            "p_vs_m1": boot_p(per_day(ft["ts"].filter(pl.Series(sel_t)), nt[sel_t], "a"),
                              per_day(ft["ts"].filter(pl.Series(rule_t)), nt[rule_t], "b")),
        }
        r["pass"] = {
            "1_better_than_m0": n > 0 and r["test_mean_bp"] > max(r["m0_mean_bp"] or 0, 0),
            "2_mean_n": n >= MIN_TRADES and r["test_mean_bp"] > 0,
            "3_vs_m1": r["p_vs_m1"] < ALPHA,
            "4_drop5": n > 0 and r["test_drop5_bp_total"] > 0,
            "5_strict_fill": r["test_mean_strict_bp"] is not None and r["test_mean_strict_bp"] > 0,
            "6_g0": r["test_annual_rmb"] >= 3000,
        }
        return r
    return strat


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = with_other({inst: minute_frame(inst) for inst in MAJORS})
    notional = lambda s: CAPITAL_RMB * RISK / s  # noqa: E731  单笔最多亏总资金 1%
    results = []
    for name, side, direction in [("D1_revert_long", "down", 1), ("D2_follow_short", "down", -1),
                                  ("D2_follow_long", "up", 1)]:
        ev = pl.concat([minute_events(df, side, direction, MINUTE_STOPS) for df in frames.values()]).sort("ts")
        ev.write_parquet(OUT / f"events_{name}.parquet")
        base = {s: {"n": ev.height, "win": round(float(ev[f"y_{s}"].mean()), 3),
                    "mean_bp": round(float(ev[f"net_{s}"].mean() * 1e4), 1), "breakeven": round(breakeven(s), 3)}
                for s in MINUTE_STOPS}
        print(name, "基准（全部做）", base, flush=True)
        trials.record(EXPERIMENT, variant=f"{name}_baseline", sample="in_sample",
                      params={"stops": MINUTE_STOPS, "horizon_min": HORIZON}, metrics=base,
                      note="OPP-006 第 2 轮：不筛选时的胜率（含验证、测试段）")
        res = run_direction(name, ev, BASE_FEATS, MINUTE_STOPS, notional, {"direction": name})
        res["baseline"] = base
        results.append(res)

    hourly = d3_dataset(frames)
    hourly.write_parquet(OUT / "d3_hourly.parquet")
    d3_feats = [c for c in hourly.columns if c not in ("ts", "y", "exit")]
    strat = d3_strategy(d3_fills(), hourly, notional_rmb=100_000)
    res = run_direction("D3_regime", hourly, d3_feats, [0], None,
                        {"direction": "D3_regime", "rule_feats": ["btc_rv60"]}, label_kind="regime", strat=strat)
    results.append(res)
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    for r in results:
        print(r["direction"], "选中止损档", r["chosen_stop"], "通过" if r["pass"] else "不通过", flush=True)


if __name__ == "__main__":
    main()
