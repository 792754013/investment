from __future__ import annotations

from pathlib import Path
from typing import List

from system.models import DecisionResult, RunResult
from system.utils import ensure_dir, generate_run_id, save_json, load_json

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs"


def save_run(date_str: str, product: str, results: List[DecisionResult]) -> RunResult:
    ensure_dir(str(RUN_DIR))
    run_id = generate_run_id(product.lower())
    payload = {
        "run_id": run_id,
        "date": date_str,
        "product": product,
        "results": [item.dict() for item in results],
    }
    save_json(str(RUN_DIR / f"{run_id}.json"), payload)
    return RunResult(run_id=run_id, date=date_str, product=product, results=results)


def load_run(run_id: str) -> RunResult:
    payload = load_json(str(RUN_DIR / f"{run_id}.json"))
    results = [DecisionResult(**item) for item in payload.get("results", [])]
    return RunResult(run_id=payload["run_id"], date=payload["date"], product=payload["product"], results=results)
