from __future__ import annotations

from typing import List

import pandas as pd

from system.config_loader import load_news, load_products
from system.models import DemandEvent, PipelineState


def apply(state: PipelineState) -> None:
    products = load_products()
    themes = products.get("products", {}).get(state.product, [])
    news = load_news()
    day_news = news[(news["date"].dt.date == state.date) & (news["product"] == state.product)]
    events: List[DemandEvent] = []
    for theme in themes:
        theme_news = day_news[day_news["theme"] == theme]
        if not theme_news.empty:
            count = float(theme_news["news_count"].iloc[0])
            reason = f"需求事件数量={int(count)}"
        else:
            count = 0.0
            reason = "当日无新闻事件"
        signal_strength = min(1.0, count / 10.0)
        events.append(DemandEvent(theme=theme, signal_strength=signal_strength, reason=reason))
    state.events = events
