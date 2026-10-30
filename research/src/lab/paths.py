"""数据目录约定。

data/raw/     下载的原始文件（含封存段），只给入库脚本读
data/lake/    封存起点之前的数据，分析代码只读这里
data/sealed/  封存起点之后的数据，未解封前不得读取
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_ROOT = REPO_ROOT / "research"


def data_root() -> Path:
    return Path(os.environ.get("LAB_DATA_ROOT") or REPO_ROOT / "data")


def raw_dir() -> Path:
    return data_root() / "raw"


def lake_dir() -> Path:
    return data_root() / "lake"


def sealed_dir() -> Path:
    return data_root() / "sealed"
