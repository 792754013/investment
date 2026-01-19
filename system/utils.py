"""通用工具函数：目录创建、运行编号与 JSON 读写。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    # exist_ok=True 表示目录已存在也不会报错
    os.makedirs(path, exist_ok=True)


def generate_run_id(prefix: str) -> str:
    # 生成 UTC 时间戳，避免本地时区差异带来的排序问题
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{prefix}-{timestamp}"


def save_json(path: str, payload: Dict[str, Any]) -> None:
    # 使用 UTF-8 且 ensure_ascii=False，保证中文可读
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    # 读取 JSON 文件并解析成 Python 字典
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
