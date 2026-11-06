"""E4 税费与持有规则（plan.md 4.2 节）。

不是回测，是按规则算账：在不同金额、频率、持有期下，各种费用和税每年是多少元。
规则数字集中在 RULES 里，来源和核实日期写在结论文件 docs/research/E04-tax-fees.md。
券商费率因人而异，按几档列出；拿到你的书面费率后，改 BROKER 重算。

问题：
  1. 最低佣金：单笔金额多小时，最低 5 元会把实际费率抬高多少；对定投和再平衡每年影响多少元。
  2. 场内 ETF 和场外指数基金（A 类、C 类）在不同持有期下的总成本，以及 A、C 的分界点。
  3. 红利税：直接持股和持有基金的差别；卖出时间对税负的影响。
  4. 港股高股息：港股通基金和 QDII 基金的股息预提税差别。
  5. 个人养老金：每年能省多少税。

用法：uv run python research/analysis/e04_tax_fees.py
"""

import json

from lab import trials
from lab.paths import data_root

OUT = data_root() / "research" / "e04"

# ---- 规则（核实结果见结论文件） ----
RULES = {
    "stamp_sell_stock": 0.0005,  # 股票卖出印花税（2023-08-28 起减半）；ETF/LOF 不在征税范围
    "fees_stock": 0.0000341 + 0.00002 + 0.00001,  # 股票经手费 + 证管费 + 过户费，双向
    "fees_etf": 0.00004,  # ETF 经手费，双向；ETF 过户费、证管费未核实，按 0 计（金额极小）
    "div_tax": {"le_1m": 0.20, "1m_1y": 0.10, "gt_1y": 0.0},  # 财税〔2015〕101 号，卖出时补扣，先进先出
    "hk_div_tax": {"stock_connect_h": 0.20, "qdii_h": 0.10},  # 港股通 20% 已核实；QDII 10% 未核实
    "pension_cap": 12_000,
    "pension_withdraw_tax": 0.03,
}
# 券商佣金档位（成交额比例，最低收费）。最低 5 元出自证监发〔2002〕21 号；"免五"属灰色做法，单列作对照。
BROKER = [(0.0001, 0), (0.0001, 5), (0.00025, 0), (0.00025, 5), (0.0003, 5)]
# 场外指数基金，按 2026-01-01 施行的销售费用新规：申购费上限 0.3%，销售服务费上限 0.2%、持有满 1 年免收。
# 7 天以上的赎回费由管理人定，悲观地沿用旧的常见档（A 类 1 年内 0.5%、2 年内 0.25%；C 类 30 天内 0.5%）。
OTC = {
    "A 申购0.3%": {"sub": 0.003, "redeem": [(7, 0.015), (365, 0.005), (730, 0.0025)], "service": 0.0},
    "A 申购0.1%": {"sub": 0.001, "redeem": [(7, 0.015), (365, 0.005), (730, 0.0025)], "service": 0.0},
    "A 申购0.1% 7天后免赎回": {"sub": 0.001, "redeem": [(7, 0.015)], "service": 0.0},  # 敏感性：新规允许 7 天后不收
    "C": {"sub": 0.0, "redeem": [(7, 0.015), (30, 0.005)], "service": 0.002, "service_days_cap": 365},
}
ETF_HALF_SPREAD = 0.0002  # 宽基 ETF 买卖价差的一半（假设，流动性好的宽基约 1 个最小价位）


def commission(amount: float, rate: float, minimum: float) -> float:
    return max(rate * amount, minimum) if amount > 0 else 0.0


def trade_cost(amount: float, rate: float, minimum: float, kind: str, side: str) -> float:
    """单笔交易成本（元）。kind: etf / stock。"""
    c = commission(amount, rate, minimum)
    if kind == "stock":
        c += RULES["fees_stock"] * amount
        if side == "sell":
            c += RULES["stamp_sell_stock"] * amount
    else:
        c += RULES["fees_etf"] * amount
    return c


def q1_min_commission() -> dict:
    amounts = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
    single = {}
    for rate, mn in BROKER:
        key = f"万{rate * 1e4:g}/最低{mn}"
        single[key] = {a: round(commission(a, rate, mn) / a * 1e4, 2) for a in amounts}  # 单边，万分之几
    # 定投：每月投 M 元，分到 k 只 ETF，每年的佣金（元）
    dca = {}
    for m in [500, 1_000, 2_000, 5_000, 10_000]:
        for k in [1, 2, 4]:
            row = {}
            for rate, mn in BROKER:
                row[f"万{rate * 1e4:g}/最低{mn}"] = round(12 * k * commission(m / k, rate, mn), 1)
            dca[f"每月{m}元/{k}只"] = row
    # 最低收费起作用的成交额门槛
    breakeven = {f"万{r * 1e4:g}": round(mn / r) for r, mn in BROKER if mn}
    return {"single_bp": single, "dca_yuan_per_year": dca, "min_binding_below": breakeven}


def redeem_rate(schedule, days: int) -> float:
    for limit, r in schedule:
        if days < limit:
            return r
    return 0.0


def otc_cost(f: dict, d: int) -> float:
    """场外份额持有 d 天的交易和销售成本（比例）。"""
    service_days = min(d, f.get("service_days_cap", 10**9))
    return f["sub"] / (1 + f["sub"]) + f["service"] * service_days / 365 + redeem_rate(f["redeem"], d)


def q2_holding_cost(amount: float = 100_000) -> dict:
    """持有 d 天的总成本（元），只算交易和销售环节；管理费、托管费另算（同一指数各产品不同，属于 E1）。"""
    days_list = [7, 30, 90, 180, 365, 540, 730, 1095, 1825]
    out = {}
    for d in days_list:
        row = {k: round(otc_cost(f, d) * amount) for k, f in OTC.items()}
        for rate, mn in [(0.0001, 0), (0.00025, 5)]:
            etf = trade_cost(amount, rate, mn, "etf", "buy") + trade_cost(amount, rate, mn, "etf", "sell")
            row[f"ETF 万{rate * 1e4:g}/最低{mn}"] = round(etf + 2 * ETF_HALF_SPREAD * amount)
        out[d] = row
    # 每个持有期下 C 是否比 A 便宜
    c_cheaper = {k: [d for d in range(8, 1826) if otc_cost(OTC["C"], d) < otc_cost(f, d)][:1] + [
        d for d in range(8, 1826) if otc_cost(OTC["C"], d) >= otc_cost(f, d)][:1] for k, f in OTC.items() if k != "C"}
    return {"amount": amount, "cost_yuan": out, "c_vs_a_first_cheaper_first_not": c_cheaper}


def q1c_dca_channels() -> dict:
    """每月定投 M 元、1 只基金，第一年的成本（元）：只算买入环节；C 类按每笔实际持有月数收销售服务费（满 1 年免）。"""
    out = {}
    for m in [500, 1_000, 2_000, 5_000, 10_000]:
        row = {}
        for rate, mn in [(0.0001, 0), (0.00025, 5)]:
            row[f"ETF 万{rate * 1e4:g}/最低{mn}"] = round(12 * trade_cost(m, rate, mn, "etf", "buy"), 1)
        for k, f in OTC.items():
            if k == "C":
                row["C 第一年已计提"] = round(sum(m * f["service"] * (12 - i) / 12 for i in range(12)), 1)
                row["C 稳态每年"] = round(12 * m * f["service"], 1)  # 每笔份额最终都付满 1 年
            elif "免赎回" in k:
                continue
            else:
                row[k] = round(12 * m * f["sub"] / (1 + f["sub"]), 1)
        out[f"每月{m}元"] = row
    return out


def q1b_stock_vs_etf(capital: float = 100_000) -> dict:
    """同样一笔 10 万元的买入 + 卖出，股票和 ETF 的成本（元）。"""
    out = {}
    for rate, mn in BROKER:
        s = trade_cost(capital, rate, mn, "stock", "buy") + trade_cost(capital, rate, mn, "stock", "sell")
        e = trade_cost(capital, rate, mn, "etf", "buy") + trade_cost(capital, rate, mn, "etf", "sell")
        out[f"万{rate * 1e4:g}/最低{mn}"] = {"股票": round(s, 1), "ETF": round(e, 1)}
    return {"capital": capital, "round_trip_yuan": out}


def q3_dividend_tax(capital: float = 100_000) -> dict:
    """直接持有红利股：股息率 y，持有期不同的税负（元/年）。持基金：个人分配免税，基金层面按基金持股期。"""
    out = {}
    for y in [0.03, 0.05, 0.07]:
        div = capital * y
        out[f"股息率{y:.0%}"] = {k: round(div * t) for k, t in RULES["div_tax"].items()}
    return {"capital": capital, "tax_yuan_per_year": out}


def q4_hk_div(capital: float = 100_000) -> dict:
    out = {}
    for y in [0.05, 0.06, 0.07, 0.08]:
        div = capital * y
        sc, qd = RULES["hk_div_tax"]["stock_connect_h"], RULES["hk_div_tax"]["qdii_h"]
        out[f"股息率{y:.0%}"] = {"港股通渠道": round(div * sc), "QDII渠道": round(div * qd), "差额": round(div * (sc - qd))}
    return {"capital": capital, "withholding_yuan_per_year": out}


def q5_pension() -> dict:
    """缴存时省 1.2 万 × 边际税率；领取时对全部领取额（含收益）按 3% 计税。
    账户外持有基金的收益本来免个税，所以收益部分的 3% 是净成本，按假设收益率、年数摊到每年。"""
    cap, w = RULES["pension_cap"], RULES["pension_withdraw_tax"]
    out = {}
    for marginal in [0.0, 0.03, 0.10, 0.20, 0.25, 0.30]:
        row = {"只算本金": round(cap * (marginal - w))}
        for r, n in [(0.02, 20), (0.04, 20), (0.04, 30)]:
            fv = cap * ((1 + r) ** n - 1) / r
            gain_tax_per_year = (fv - cap * n) * w / n
            row[f"收益{r:.0%}/{n}年"] = round(cap * (marginal - w) - gain_tax_per_year)
        out[f"边际税率{marginal:.0%}"] = row
    return {"cap": cap, "net_saving_yuan_per_year": out}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"q1": q1_min_commission(), "q1b": q1b_stock_vs_etf(), "q1c": q1c_dca_channels(), "q2": q2_holding_cost(), "q3": q3_dividend_tax(), "q4": q4_hk_div(),
           "q5": q5_pension(), "rules": RULES, "broker": BROKER, "otc": OTC, "etf_half_spread": ETF_HALF_SPREAD}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    trials.record("e04_tax_fees", sample="in_sample", params={"rules": RULES, "broker": BROKER, "otc": OTC,
                  "etf_half_spread": ETF_HALF_SPREAD}, metrics={"c_vs_a": res["q2"]["c_vs_a_first_cheaper_first_not"], "pension": res["q5"]["net_saving_yuan_per_year"]},
                  note="规则算账，非回测；不读行情数据（第 3 次运行：按复核意见补 C 类稳态、A 类免赎回敏感性、养老金收益 3%）")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
