import math

import pandas as pd
import pytest

from race_predictor.exceptions import DataValidationError
from race_predictor.processing import (
    DropAthletesWithoutHR,
    DropTriathlonRaces,
    EstimateHRMax,
    ImputeMissing,
    MarkTriathlonRuns,
    Pipeline,
    RemoveAnomalies,
    RemoveImplausiblyFastRuns,
    Step,
    ValidateHeartRate,
    ValidateMaxHeartRate,
)
from tests.conftest import Days, Frame

NAN = math.nan


class _AddOne(Step):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(distance_m=df["distance_m"] + 1)


def test_pipeline_runs_steps_in_order(frame: Frame, days: Days) -> None:
    pipeline = Pipeline([_AddOne(), _AddOne()])
    out = pipeline.run(frame([{"date": days(0)}]))
    assert out["distance_m"].iloc[0] == 8002.0
    assert len(pipeline) == 2
    assert [repr(step) for step in pipeline] == ["_AddOne()", "_AddOne()"]


def test_pipeline_requires_columns() -> None:
    with pytest.raises(DataValidationError, match="missing columns"):
        Pipeline([]).run(pd.DataFrame({"athlete_id": []}))


def test_steps_do_not_modify_their_input(frame: Frame, days: Days) -> None:
    df = frame([{"date": days(0), "max_hr": 250.0}, {"date": days(1), "avg_speed_ms": 9.0}])
    before = df.copy()
    for step in [RemoveAnomalies(), ValidateMaxHeartRate(), EstimateHRMax(), ImputeMissing()]:
        step.apply(df)
    pd.testing.assert_frame_equal(df, before)


def test_remove_anomalies_only_touches_runs(frame: Frame, days: Days) -> None:
    df = frame(
        [
            {"date": days(0)},
            {"date": days(1), "avg_speed_ms": 9.0},  # too fast on average
            {"date": days(2), "max_speed_ms": 16.0},  # impossible peak
            {"date": days(3), "distance_m": 500.0},  # too short
            {"date": days(4), "type": "Ride", "avg_speed_ms": 12.0},
        ]
    )
    assert RemoveAnomalies().apply(df)["type"].tolist() == ["Run", "Ride"]


def test_implausibly_fast_runs_are_judged_per_athlete(frame: Frame, days: Days) -> None:
    df = frame(
        [
            *({"date": days(i), "avg_speed_ms": 3.0} for i in range(5)),
            {"date": days(5), "avg_speed_ms": 6.3},  # 2.1x the median: a ride logged as a run
            {"date": days(6), "avg_speed_ms": 4.8},  # 1.6x: fast but possible
            {"date": days(7), "type": "Ride", "avg_speed_ms": 7.0},
            *({"date": days(i), "athlete_id": "b", "avg_speed_ms": 4.5} for i in range(3)),
        ]
    )
    out = RemoveImplausiblyFastRuns().apply(df)
    assert 6.3 not in out["avg_speed_ms"].tolist()
    assert len(out) == len(df) - 1


def test_max_hr_spikes_are_blanked_before_estimating_hr_max(frame: Frame, days: Days) -> None:
    df = frame([{"date": days(0), "max_hr": 185.0}, {"date": days(1), "max_hr": 244.0}])
    out = EstimateHRMax().apply(ValidateMaxHeartRate().apply(df))
    assert out["max_hr"].isna().sum() == 1
    assert (out["hr_max"] == 185.0).all()


def test_validate_heart_rate(frame: Frame, days: Days) -> None:
    df = frame(
        [{"date": days(0)}, {"date": days(1), "avg_hr": 30.0}, {"date": days(2), "avg_hr": NAN}]
    )
    assert len(ValidateHeartRate().apply(df)) == 1
    assert len(ValidateHeartRate(keep_missing=True).apply(df)) == 2


def test_drop_athletes_without_hr(frame: Frame, days: Days) -> None:
    df = frame(
        [
            {"date": days(0)},
            {"date": days(0), "athlete_id": "b", "avg_hr": NAN, "max_hr": NAN},
        ]
    )
    assert DropAthletesWithoutHR().apply(df)["athlete_id"].tolist() == ["a"]


def test_estimate_hr_max_is_per_athlete(frame: Frame, days: Days) -> None:
    df = frame(
        [
            {"date": days(0), "max_hr": 170.0},
            {"date": days(1), "max_hr": 185.0},
            {"date": days(1), "athlete_id": "b", "max_hr": 160.0},
        ]
    )
    assert EstimateHRMax().apply(df)["hr_max"].tolist() == [185.0, 185.0, 160.0]


def test_mark_triathlon_runs(frame: Frame, days: Days) -> None:
    df = frame(
        [
            {"date": days(0), "type": "Ride"},
            {"date": days(0) + pd.Timedelta(hours=2)},  # run right after the ride
            {"date": days(5)},
            {"date": days(6), "type": "Ride"},
            {"date": days(6) + pd.Timedelta(hours=8)},  # too long after the ride
        ]
    )
    assert MarkTriathlonRuns().apply(df)["is_triathlon"].tolist() == [0, 1, 0, 0, 0]


def test_drop_triathlon_races_keeps_the_run_as_history(frame: Frame, days: Days) -> None:
    df = frame([{"date": days(0)}, {"date": days(1)}]).assign(
        is_race=[True, True], is_triathlon=[0, 1]
    )
    out = DropTriathlonRaces().apply(df)
    assert len(out) == 2
    assert out["is_race"].tolist() == [True, False]


def test_impute_missing_uses_the_athletes_own_median_first(frame: Frame, days: Days) -> None:
    df = frame(
        [
            {"date": days(0), "avg_hr": 140.0},
            {"date": days(1), "avg_hr": 150.0},
            {"date": days(2), "avg_hr": NAN},
            {"date": days(0), "athlete_id": "b", "avg_hr": 170.0},
            {"date": days(0), "athlete_id": "c", "avg_hr": NAN},
        ]
    ).assign(hr_max=NAN)
    out = ImputeMissing().apply(df)
    assert out["avg_hr"].tolist() == [140.0, 150.0, 145.0, 170.0, 150.0]
    assert out["hr_max"].isna().all()  # never imputed
