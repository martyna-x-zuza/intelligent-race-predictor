import pandas as pd

from race_predictor.processing import (
    DetectRaces,
    EstimateHRMax,
    HeuristicRaceDetector,
    RaceDetector,
    StandardDistanceDetector,
)
from race_predictor.processing._races import _is_standard_distance
from tests.conftest import Days, Frame


def test_is_standard_distance() -> None:
    distances = pd.Series([5100.0, 7000.0, 21_000.0, 42_500.0, 46_500.0])
    assert _is_standard_distance(distances).tolist() == [True, False, True, True, False]


def test_standard_distance_detector(frame: Frame, days: Days) -> None:
    df = frame(
        [
            {"date": days(0), "distance_m": 10_100.0},
            {"date": days(1), "distance_m": 7000.0},
            {"date": days(2), "distance_m": 10_000.0, "type": "Ride"},
        ]
    )
    assert StandardDistanceDetector().detect(df).tolist() == [True, False, False]


def test_heuristic_detector_flags_a_hard_standard_distance_run(frame: Frame, days: Days) -> None:
    easy = [
        {"date": days(i), "distance_m": 7000.0, "avg_hr": 140.0, "avg_speed_ms": 3.0}
        for i in range(0, 60, 3)
    ]
    race = {"date": days(61), "distance_m": 10_000.0, "avg_hr": 175.0, "avg_speed_ms": 3.6}
    df = EstimateHRMax().apply(frame([*easy, race]).assign(max_hr=190.0))
    flags = HeuristicRaceDetector().detect(df)
    assert flags.iloc[-1]
    assert not flags.iloc[:-1].any()


def test_heuristic_reasons_explain_the_decision(frame: Frame, days: Days) -> None:
    df = frame(
        [
            *({"date": days(i)} for i in range(5)),
            {"date": days(6), "distance_m": 10_000.0, "avg_hr": 175.0},  # high HR + 10 km
            {"date": days(7), "distance_m": 7000.0},  # nothing special
        ]
    ).assign(hr_max=190.0, is_triathlon=0)
    detector = HeuristicRaceDetector()
    reasons = detector.reasons(df)
    assert reasons["is_race"].equals(detector.detect(df))
    race, ordinary = reasons.iloc[5], reasons.iloc[6]
    assert race["high_hr"] and race["standard_distance"] and race["score"] == 3
    assert race["is_race"]
    assert ordinary["score"] == 0 and not ordinary["is_race"]


def test_long_stops_disqualify_a_race(frame: Frame, days: Days) -> None:
    df = frame([{"date": days(0), "distance_m": 10_000.0, "avg_hr": 175.0, "elapsed_s": 4000.0}])
    assert not HeuristicRaceDetector().detect(df.assign(hr_max=190.0)).iloc[0]


def test_detect_races_accepts_any_strategy(frame: Frame, days: Days) -> None:
    class EveryRun:
        def detect(self, df: pd.DataFrame) -> pd.Series:
            return df["type"] == "Run"

    detector: RaceDetector = EveryRun()
    step = DetectRaces(detector)
    out = step.apply(frame([{"date": days(0)}, {"date": days(1), "type": "Swim"}]))
    assert out["is_race"].tolist() == [True, False]
    assert repr(step) == "DetectRaces(EveryRun)"
    assert repr(DetectRaces()) == "DetectRaces(HeuristicRaceDetector)"
