"""配置加载器：集中读取 YAML/CSV 配置与数据。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

# 根目录与配置/数据路径统一在此定义，方便维护
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "system" / "configs"
DATA_DIR = ROOT / "system" / "data"


def load_yaml(path: Path) -> Dict:
    # 读取 YAML 文件并返回字典，空文件返回 {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_products() -> Dict:
    # 产品配置：产品与主题的映射
    return load_yaml(CONFIG_DIR / "products.yaml")


def load_themes() -> Dict:
    # 主题配置：主题到约束的映射
    return load_yaml(CONFIG_DIR / "themes.yaml")


def load_constraints() -> Dict:
    # 约束配置：约束指标与健康度
    return load_yaml(CONFIG_DIR / "constraints.yaml")


def load_thresholds() -> Dict:
    # 阈值配置：各种策略参数
    return load_yaml(CONFIG_DIR / "thresholds.yaml")


def load_assets() -> pd.DataFrame:
    # 资产表：主题与标的的对应
    return pd.read_csv(CONFIG_DIR / "assets.csv")


def load_prices() -> pd.DataFrame:
    # 历史价格数据
    return pd.read_csv(DATA_DIR / "prices.csv", parse_dates=["date"])


def load_macro() -> pd.DataFrame:
    # 宏观指标数据（示例）
    return pd.read_csv(DATA_DIR / "macro_stub.csv", parse_dates=["date"])


def load_news() -> pd.DataFrame:
    # 新闻计数数据（示例）
    return pd.read_csv(DATA_DIR / "news_stub.csv", parse_dates=["date"])


def load_stage_overrides(path: str | None) -> Dict[str, Dict[str, str]]:
    # 读取阶段覆盖文件，支持为空
    if not path:
        return {}
    data = load_yaml(Path(path))
    return data.get("overrides", {})


def list_trading_dates(start: date, end: date) -> List[date]:
    # 使用 pandas 生成日期列表，按自然日频率
    dates = pd.date_range(start=start, end=end, freq="D")
    return [d.date() for d in dates]


def load_product_context(product: str) -> Tuple[List[str], Dict]:
    # 返回产品主题列表与完整产品配置
    products = load_products()
    themes = products.get("products", {}).get(product, [])
    return themes, products
