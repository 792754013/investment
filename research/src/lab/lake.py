"""Parquet 数据湖读写，入库时按封存起点自动切分。

写：write("funding_rate", df, ts_col="ts", part="2024-01")
    封存起点之前的行 → data/lake/funding_rate/2024-01.parquet
    封存起点之后的行 → data/sealed/funding_rate/2024-01.parquet
读：scan("funding_rate") 只返回 lake 部分。
    读取封存段必须 scan(..., unseal=True, reason=...)，且环境变量 LAB_UNSEAL=1；
    每次解封都会记入试验日志。
"""

import os
from pathlib import Path

import polars as pl

from lab import trials
from lab.paths import lake_dir, sealed_dir
from lab.seal import Seal, load_seal


class SealedDataError(RuntimeError):
    pass


def write(
    dataset: str,
    df: pl.DataFrame,
    *,
    ts_col: str,
    part: str,
    seal: Seal | None = None,
) -> None:
    """按 part 覆盖写入（重复入库是幂等的）。ts_col 必须是 UTC 的 Datetime 列。"""
    dtype = df.schema[ts_col]
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != "UTC":
        raise TypeError(f"{dataset}.{ts_col} 必须是 Datetime(time_zone='UTC')，实际是 {dtype}")

    seal = seal or load_seal()
    if dataset in seal.exempt:
        _write_part(lake_dir() / dataset, part, df)
        return

    sealed_mask = pl.col(ts_col) >= seal.start
    _write_part(lake_dir() / dataset, part, df.filter(~sealed_mask))
    _write_part(sealed_dir() / dataset, part, df.filter(sealed_mask))


def scan(dataset: str, *, unseal: bool = False, reason: str | None = None) -> pl.LazyFrame:
    files = sorted((lake_dir() / dataset).glob("*.parquet"))
    if unseal:
        if os.environ.get("LAB_UNSEAL") != "1" or not reason:
            raise SealedDataError(
                "读取封存段需要同时满足：环境变量 LAB_UNSEAL=1、传入 reason。"
                "只有参数已定、G1 门槛已提交后才允许解封。"
            )
        trials.record(
            "unseal",
            params={"dataset": dataset},
            metrics={},
            sample="sealed",
            note=reason,
        )
        files += sorted((sealed_dir() / dataset).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"数据集 {dataset} 没有数据（{lake_dir() / dataset}）")
    return pl.scan_parquet(files)


def _write_part(directory: Path, part: str, df: pl.DataFrame) -> None:
    path = directory / f"{part}.parquet"
    if df.is_empty():
        path.unlink(missing_ok=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
