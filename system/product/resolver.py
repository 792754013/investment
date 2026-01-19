"""产品解析器：从配置与资产表中解析主题与约束。"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from system.config_loader import load_assets, load_constraints, load_themes


def resolve_assets(product: str) -> pd.DataFrame:
    # 读取资产配置，并筛选当前产品
    assets = load_assets()
    return assets[assets["product"] == product].copy()


def resolve_themes(product: str) -> List[str]:
    # 根据产品的资产表提取去重后的主题列表
    assets = resolve_assets(product)
    return sorted(assets["theme"].unique().tolist())


def resolve_theme_constraint(theme: str) -> Dict:
    # 将主题映射到约束配置，方便监控与风险分析
    themes = load_themes().get("themes", {})
    constraints = load_constraints().get("constraints", {})
    theme_info = themes.get(theme, {})
    constraint_id = theme_info.get("constraint_id", "")
    return {
        "theme": theme,
        "constraint_id": constraint_id,
        "constraint": constraints.get(constraint_id, {}),
    }
