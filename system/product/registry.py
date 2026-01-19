"""产品注册表：读取产品配置并提供查询接口。"""

from __future__ import annotations

from typing import Dict, List

from system.config_loader import load_products


def list_products() -> List[str]:
    # 返回所有已配置产品名称
    products = load_products()
    return sorted(products.get("products", {}).keys())


def get_product_themes(product: str) -> List[str]:
    # 获取单个产品的主题列表，若不存在则抛错提示
    products = load_products()
    themes = products.get("products", {}).get(product)
    if themes is None:
        raise ValueError(f"未找到产品: {product}")
    return themes


def get_product_meta(product: str) -> Dict:
    # 读取产品元数据（例如显示名称、描述等）
    products = load_products()
    meta = products.get("meta", {}).get(product, {})
    return meta
