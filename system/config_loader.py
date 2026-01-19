from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "system" / "configs"
DATA_DIR = ROOT / "system" / "data"


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_products() -> Dict:
    return load_yaml(CONFIG_DIR / "products.yaml")


def load_themes() -> Dict:
    return load_yaml(CONFIG_DIR / "themes.yaml")


def load_constraints() -> Dict:
    return load_yaml(CONFIG_DIR / "constraints.yaml")


def load_thresholds() -> Dict:
    return load_yaml(CONFIG_DIR / "thresholds.yaml")


def load_assets() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "assets.csv")


def load_prices() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])


def load_macro() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "macro_stub.csv", parse_dates=["date"])


def load_news() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "news_stub.csv", parse_dates=["date"])


def load_stage_overrides(path: str | None) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    data = load_yaml(Path(path))
    return data.get("overrides", {})


def list_trading_dates(start: date, end: date) -> List[date]:
    dates = pd.date_range(start=start, end=end, freq="D")
    return [d.date() for d in dates]


def load_product_context(product: str) -> Tuple[List[str], Dict]:
    products = load_products()
    themes = products.get("products", {}).get(product, [])
    return themes, products
