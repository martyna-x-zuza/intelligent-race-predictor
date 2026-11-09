import math

import numpy as np
import pandas as pd
import pytest

from race_predictor.exceptions import DataValidationError
from race_predictor.features import (
    AGGREGATES,
    RIEGEL_COLUMNS,
    SEQ_FEATURES,
    RaceDataset,
    build_race_dataset,
)
from tests.conftest import Days, Frame


@pytest.fixture
def cleaned(frame: Frame, days: Days) -> pd.DataFrame:
    """Two athletes: ``a`` with 10 training runs, a swim, a 10 km race; ``b`` with one race only."""
    rows = [
        *(
            {"date": days(i), "distance_m": 6000.0 + 100 * i, "avg_speed_ms": 3.0 + 0.01 * i}
            for i in range(0, 50, 5)
        ),
        {"date": days(20), "type": "Swim", "distance_m": 1500.0, "elapsed_s": 1800.0},
        {"date": days(50), "distance_m": 10_000.0, "moving_s": 2700.0, "is_race": True},
        {"date": days(60), "distance_m": 5000.0},  # after the race: must not leak in
        {"date": days(10), "athlete_id": "b", "distance_m": 5000.0, "is_race": True},
    ]
    return frame(rows).assign(
        is_race=lambda df: df["is_race"].fillna(False).astype(bool), is_triathlon=0
    )


def test_one_example_per_race_with_enough_history(cleaned: pd.DataFrame) -> None:
    ds = build_race_dataset(cleaned)
    assert len(ds) == 1  # athlete b has no earlier runs
    assert ds.meta.loc[0, "athlete_id"] == "a"
    assert ds.y[0] == pytest.approx(270.0)  # 2700 s over 10 km
    assert ds.race_time_s[0] == pytest.approx(2700.0)


def test_sequence_holds_only_runs_before_the_race(cleaned: pd.DataFrame) -> None:
    (seq,) = build_race_dataset(cleaned).seqs
    assert seq.shape == (10, len(SEQ_FEATURES))
    days_to_race = seq[:, SEQ_FEATURES.index("days_to_race")]
    assert (days_to_race > 0).all()
    assert list(days_to_race) == sorted(days_to_race, reverse=True)  # oldest first


def test_static_features(cleaned: pd.DataFrame) -> None:
    static = build_race_dataset(cleaned).static.iloc[0]
    assert set(AGGREGATES) | set(RIEGEL_COLUMNS) <= set(static.index)
    assert static["n_runs"] == 10
    assert static["swim_sessions"] == 1 and static["swim_dist_total"] == 1500.0
    assert static["bike_sessions"] == 0 and static["bike_t_total"] == 0.0
    assert static["longest_run_km"] == pytest.approx(10.5)
    # Riegel reference: the fastest run of at least 5 km (the last training run).
    assert static["riegel_ref_distance_m"] == 10_500.0
    assert static["riegel_ref_time_s"] == pytest.approx(10_500.0 / 3.45)


def test_riegel_reference_is_missing_without_a_long_run(frame: Frame, days: Days) -> None:
    rows = [{"date": days(i), "distance_m": 3000.0} for i in range(5)]
    rows.append({"date": days(9), "distance_m": 5000.0, "is_race": True})
    df = frame(rows).assign(is_race=lambda d: d["is_race"].fillna(False).astype(bool))
    static = build_race_dataset(df).static.iloc[0]
    assert math.isnan(static["riegel_ref_time_s"])


def test_no_qualifying_race_is_an_error(cleaned: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="no race"):
        build_race_dataset(cleaned, min_runs=50)


def test_subset_by_positions_and_mask(cleaned: pd.DataFrame) -> None:
    ds = build_race_dataset(cleaned, min_runs=0)
    assert len(ds) == 2
    assert ds.subset([1]).meta["athlete_id"].tolist() == ["b"]
    assert ds.subset(np.array([True, False])).meta["athlete_id"].tolist() == ["a"]
    assert len(ds.subset([])) == 0


def test_inconsistent_parts_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="inconsistent"):
        RaceDataset([], pd.DataFrame({"x": [1.0]}), np.array([1.0]), pd.DataFrame({"x": [1]}))
