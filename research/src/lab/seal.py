"""封存样本外的配置（research/seal.toml）。"""

import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lab.paths import RESEARCH_ROOT

SEAL_FILE = RESEARCH_ROOT / "seal.toml"


@dataclass(frozen=True)
class Seal:
    start: datetime
    exempt: frozenset[str]


def load_seal(path: Path = SEAL_FILE) -> Seal:
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    start = cfg["start"]
    if not isinstance(start, datetime) or start.tzinfo is None:
        raise ValueError(f"{path}: start 必须是带时区的时间，例如 2026-05-01T00:00:00Z")
    return Seal(start=start.astimezone(UTC), exempt=frozenset(cfg.get("exempt", [])))
