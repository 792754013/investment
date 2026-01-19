"""需求扫描步骤：根据配置与新闻数据生成需求事件列表。"""

from __future__ import annotations

from typing import List

import pandas as pd

from system.config_loader import load_news, load_products
from system.models import DemandEvent, PipelineState


def apply(state: PipelineState) -> None:
    # 读取产品与主题配置，教学上强调：配置是策略逻辑的“地基”
    products = load_products()
    themes = products.get("products", {}).get(state.product, [])
    # 读取新闻数据并按日期、产品过滤到当天的样本
    news = load_news()
    day_news = news[(news["date"].dt.date == state.date) & (news["product"] == state.product)]
    events: List[DemandEvent] = []
    for theme in themes:
        # 每个主题单独统计新闻数量，构建需求事件
        theme_news = day_news[day_news["theme"] == theme]
        if not theme_news.empty:
            count = float(theme_news["news_count"].iloc[0])
            reason = f"需求事件数量={int(count)}"
        else:
            count = 0.0
            reason = "当日无新闻事件"
        # 将新闻数量缩放为 0~1 的信号强度，便于后续步骤使用
        signal_strength = min(1.0, count / 10.0)
        events.append(DemandEvent(theme=theme, signal_strength=signal_strength, reason=reason))
    # 把生成的事件写回到流程状态，供下一步读取
    state.events = events
