"""试验日志（plan.md 9.5 节：记录所有试过的参数组合和规则变体的数量）。

每跑一次回测、换一次参数或规则，都调用 record() 追加一行到 research/trials.csv。
该文件纳入 git，只追加、不删改；研究报告里的"试验次数"从这里统计。
"""

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from lab.paths import REPO_ROOT, RESEARCH_ROOT

TRIALS_FILE = RESEARCH_ROOT / "trials.csv"
FIELDS = ["ts_utc", "experiment", "variant", "sample", "params", "metrics", "git_commit", "note"]
SAMPLES = {"in_sample", "sealed"}


def record(
    experiment: str,
    *,
    params: dict,
    metrics: dict,
    sample: str,
    variant: str = "",
    note: str = "",
    path: Path | None = None,
) -> None:
    path = path or TRIALS_FILE
    if sample not in SAMPLES:
        raise ValueError(f"sample 必须是 {sorted(SAMPLES)} 之一，实际是 {sample!r}")
    row = {
        "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "experiment": experiment,
        "variant": variant,
        "sample": sample,
        "params": json.dumps(params, sort_keys=True, ensure_ascii=False, default=str),
        "metrics": json.dumps(metrics, sort_keys=True, ensure_ascii=False, default=str),
        "git_commit": _git_commit(),
        "note": note,
    }
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def count(experiment: str | None = None, *, path: Path | None = None) -> int:
    path = path or TRIALS_FILE
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for r in csv.DictReader(f) if experiment is None or r["experiment"] == experiment)


def _git_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    return f"{sha}-dirty" if git("status", "--porcelain") else sha
