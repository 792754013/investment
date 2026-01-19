from __future__ import annotations

from typing import Dict, List

from system.config_loader import load_products


def list_products() -> List[str]:
    products = load_products()
    return sorted(products.get("products", {}).keys())


def get_product_themes(product: str) -> List[str]:
    products = load_products()
    themes = products.get("products", {}).get(product)
    if themes is None:
        raise ValueError(f"未找到产品: {product}")
    return themes


def get_product_meta(product: str) -> Dict:
    products = load_products()
    meta = products.get("meta", {}).get(product, {})
    return meta
