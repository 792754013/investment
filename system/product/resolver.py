from __future__ import annotations

from typing import Dict, List

import pandas as pd

from system.config_loader import load_assets, load_constraints, load_themes


def resolve_assets(product: str) -> pd.DataFrame:
    assets = load_assets()
    return assets[assets["product"] == product].copy()


def resolve_themes(product: str) -> List[str]:
    assets = resolve_assets(product)
    return sorted(assets["theme"].unique().tolist())


def resolve_theme_constraint(theme: str) -> Dict:
    themes = load_themes().get("themes", {})
    constraints = load_constraints().get("constraints", {})
    theme_info = themes.get(theme, {})
    constraint_id = theme_info.get("constraint_id", "")
    return {
        "theme": theme,
        "constraint_id": constraint_id,
        "constraint": constraints.get(constraint_id, {}),
    }
