import math

import pandas as pd
import pytest

from race_predictor.processing import Pipeline, cleaning_pipeline, training_windows
from tests.conftest import Days, Frame

NAN = math.nan


def _names(pipeline: Pipeline) -> list[str]:
    return [type(step).__name__ for step in pipeline]


def test_train_and_predict_share_the_cleaning_steps() -> None:
    train, predict = _names(cleaning_pipeline("train")), _names(cleaning_pipeline("predict"))
    shared = ["RemoveAnomalies", "RemoveImplausiblyFastRuns", "ValidateMaxHeartRate"]
    assert train[:3] == predict[:3] == shared
    assert "DropAthletesWithoutHR" in train and "DropAthletesWithoutHR" not in predict
    assert predict[-1] == "ImputeMissing"  # imputation only after race detection


def test_triathlon_races_can_be_kept() -> None:
    assert _names(cleaning_pipeline())[-1] == "DropTriathlonRaces"
    assert "DropTriathlonRaces" not in _names(cleaning_pipeline(include_triathlon_races=True))


def test_unknown_purpose_is_rejected() -> None:
    with pytest.raises(ValueError, match="purpose"):
        cleaning_pipeline("serve")  # type: ignore[arg-type]


def test_train_cleaning_adds_the_label_columns(frame: Frame, days: Days) -> None:
    out = cleaning_pipeline("train").run(frame([{"date": days(i)} for i in range(5)]))
    assert {"hr_max", "is_triathlon", "is_race"} <= set(out.columns)


def test_predict_cleaning_keeps_runs_without_hr(frame: Frame, days: Days) -> None:
    df = frame(
        [
            *({"date": days(i), "avg_hr": NAN, "max_hr": NAN} for i in range(5)),
            {"date": days(5), "avg_speed_ms": 3.1},
            {"date": days(6), "avg_speed_ms": 6.3},  # a ride logged as a run
        ]
    )
    out = cleaning_pipeline("predict").run(df)
    assert len(out) == 6
    assert out["avg_hr"].notna().all()  # kept and imputed, not dropped
    assert 6.3 not in out["avg_speed_ms"].tolist()


def test_runs_after_a_ride_are_history_not_races(frame: Frame, days: Days) -> None:
    df = frame(
        [
            *({"date": days(i), "avg_hr": 140.0} for i in range(5)),
            {"date": days(6), "type": "Ride", "distance_m": 40_000.0, "avg_speed_ms": 8.0},
            {"date": days(6) + pd.Timedelta(hours=1), "distance_m": 5000.0, "avg_hr": 140.0},
            {"date": days(8), "distance_m": 10_000.0, "avg_hr": 175.0},  # a real 10 km race
        ]
    )
    out = cleaning_pipeline("train").run(df)
    brick = out[out["is_triathlon"] == 1]
    assert len(brick) == 1 and not brick["is_race"].any()
    assert out.loc[out["distance_m"] == 10_000.0, "is_race"].all()
    thesis = cleaning_pipeline("train", include_triathlon_races=True).run(df)
    assert thesis.loc[thesis["is_triathlon"] == 1, "is_race"].all()


def test_training_windows_never_leak(frame: Frame, days: Days) -> None:
    rows = [{"date": days(i), "is_race": False} for i in range(0, 40, 2)]
    rows.append({"date": days(41), "is_race": True})
    rows.append({"date": days(45), "is_race": False})  # after the race
    rows.append({"date": days(-400), "is_race": False})  # outside the window
    windows = list(training_windows(frame(rows), days_window=365))
    assert len(windows) == 1
    race, window = windows[0]
    assert (window["date"] < race["date"]).all()
    assert (window["date"] >= race["date"] - pd.Timedelta(days=365)).all()
    assert len(window) == 20


def test_races_with_too_little_history_have_no_window(frame: Frame, days: Days) -> None:
    rows = [{"date": days(0), "is_race": False}, {"date": days(1), "is_race": True}]
    assert list(training_windows(frame(rows), min_runs=3)) == []
