"""运行审计：保存与加载策略运行结果。"""

from __future__ import annotations

from pathlib import Path
from typing import List

from system.models import DecisionResult, RunResult
from system.utils import ensure_dir, generate_run_id, save_json, load_json

# ROOT 指向系统根目录，RUN_DIR 保存运行记录
ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs"


def save_run(date_str: str, product: str, results: List[DecisionResult]) -> RunResult:
    # 确保目录存在，然后生成唯一运行 ID
    ensure_dir(str(RUN_DIR))
    run_id = generate_run_id(product.lower())
    payload = {
        "run_id": run_id,
        "date": date_str,
        "product": product,
        "results": [item.dict() for item in results],
    }
    save_json(str(RUN_DIR / f"{run_id}.json"), payload)
    # 返回结构化结果，供 CLI 或其他模块使用
    return RunResult(run_id=run_id, date=date_str, product=product, results=results)


def load_run(run_id: str) -> RunResult:
    # 从 JSON 反序列化为 Pydantic 模型
    payload = load_json(str(RUN_DIR / f"{run_id}.json"))
    results = [DecisionResult(**item) for item in payload.get("results", [])]
    return RunResult(run_id=payload["run_id"], date=payload["date"], product=payload["product"], results=results)
