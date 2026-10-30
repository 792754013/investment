from datetime import UTC, datetime

import polars as pl
import pytest

from lab import lake, trials
from lab.seal import Seal

SEAL = Seal(start=datetime(2026, 5, 1, tzinfo=UTC), exempt=frozenset({"orderbook"}))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(trials, "TRIALS_FILE", tmp_path / "trials.csv")
    monkeypatch.delenv("LAB_UNSEAL", raising=False)
    return tmp_path


def frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [datetime(2026, 4, 30, 16, tzinfo=UTC), datetime(2026, 5, 1, tzinfo=UTC)],
            "rate": [0.0001, 0.0002],
        }
    )


def test_write_splits_at_seal_start():
    lake.write("funding_rate", frame(), ts_col="ts", part="2026-04", seal=SEAL)
    assert lake.scan("funding_rate").collect()["rate"].to_list() == [0.0001]


def test_unseal_requires_env_and_reason(monkeypatch):
    lake.write("funding_rate", frame(), ts_col="ts", part="2026-04", seal=SEAL)
    with pytest.raises(lake.SealedDataError):
        lake.scan("funding_rate", unseal=True, reason="G1 最终评估")
    monkeypatch.setenv("LAB_UNSEAL", "1")
    with pytest.raises(lake.SealedDataError):
        lake.scan("funding_rate", unseal=True)
    assert lake.scan("funding_rate", unseal=True, reason="G1 最终评估").collect().height == 2
    assert trials.count("unseal") == 1


def test_exempt_dataset_is_not_split():
    lake.write("orderbook", frame(), ts_col="ts", part="2026-04", seal=SEAL)
    assert lake.scan("orderbook").collect().height == 2


def test_write_rejects_naive_timestamps():
    df = frame().with_columns(pl.col("ts").dt.replace_time_zone(None))
    with pytest.raises(TypeError):
        lake.write("funding_rate", df, ts_col="ts", part="2026-04", seal=SEAL)


def test_rewrite_is_idempotent():
    lake.write("funding_rate", frame(), ts_col="ts", part="2026-04", seal=SEAL)
    lake.write("funding_rate", frame(), ts_col="ts", part="2026-04", seal=SEAL)
    assert lake.scan("funding_rate").collect().height == 1


def test_trials_record_and_count():
    trials.record("r1_map", params={"min_oi": 1e7}, metrics={"net_apr": 0.08}, sample="in_sample")
    trials.record("r1_map", params={"min_oi": 2e7}, metrics={"net_apr": 0.07}, sample="in_sample")
    with pytest.raises(ValueError):
        trials.record("r1_map", params={}, metrics={}, sample="oos")
    assert trials.count("r1_map") == 2
    assert trials.count() == 2
