"""监控计划生成：按产品输出主题-资产-约束的映射。"""

from __future__ import annotations

from typing import List

from system.models import MonitorItem, MonitorPlan
from system.product.resolver import resolve_assets, resolve_theme_constraint, resolve_themes


def build_monitor_plan(product: str) -> MonitorPlan:
    # 读取产品资产与主题列表
    assets = resolve_assets(product)
    themes = resolve_themes(product)
    items: List[MonitorItem] = []
    for theme in themes:
        # 每个主题关联一个约束，并绑定对应资产
        constraint = resolve_theme_constraint(theme)
        theme_assets = assets[assets["theme"] == theme]
        asset_names = theme_assets["asset_id"].tolist()
        items.append(MonitorItem(theme=theme, constraint_id=constraint["constraint_id"], assets=asset_names))
    # 返回可用于监控的计划结构
    return MonitorPlan(product=product, items=items)
