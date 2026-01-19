from __future__ import annotations

from typing import List

from system.models import MonitorItem, MonitorPlan
from system.product.resolver import resolve_assets, resolve_theme_constraint, resolve_themes


def build_monitor_plan(product: str) -> MonitorPlan:
    assets = resolve_assets(product)
    themes = resolve_themes(product)
    items: List[MonitorItem] = []
    for theme in themes:
        constraint = resolve_theme_constraint(theme)
        theme_assets = assets[assets["theme"] == theme]
        asset_names = theme_assets["asset_id"].tolist()
        items.append(MonitorItem(theme=theme, constraint_id=constraint["constraint_id"], assets=asset_names))
    return MonitorPlan(product=product, items=items)
