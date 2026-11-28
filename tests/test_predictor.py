import math

import pytest

from race_predictor.exceptions import DataValidationError
from race_predictor.predictor import RacePredictor, format_time
from tests.conftest import Days, Frame
from tests.models.test_baselines import make_dataset


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "0:00:00"), (6_330.0, "1:45:30"), (3_599.6, "1:00:00"), (math.nan, "-")],
)
def test_format_time(seconds: float, expected: str) -> None:
    assert format_time(seconds) == expected


def test_predict_gives_one_prediction_per_distance_with_a_range(frame: Frame, days: Days) -> None:
    activities = frame({"date": days(i), "distance_m": 8_000.0 + 100 * i} for i in range(10))
    predictor = RacePredictor().fit(make_dataset())

    predictions = predictor.predict(activities, [5_000.0, 10_000.0, 21_097.5])

    assert [p.distance_m for p in predictions] == [5_000.0, 10_000.0, 21_097.5]
    assert predictions[0].time_s < predictions[1].time_s < predictions[2].time_s
    assert all(p.low_s < p.high_s for p in predictions)


def test_predict_needs_a_fitted_predictor(frame: Frame, days: Days) -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        RacePredictor().predict(frame([{"date": days(0)}]))


def test_predict_needs_a_few_recent_runs(frame: Frame, days: Days) -> None:
    predictor = RacePredictor().fit(make_dataset())
    with pytest.raises(DataValidationError, match="3 or more runs"):
        predictor.predict(frame([{"date": days(0)}, {"date": days(1)}]))
