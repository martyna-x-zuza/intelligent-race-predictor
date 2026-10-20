from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from race_predictor.domain import Activity, ActivityType
from race_predictor.exceptions import DataValidationError
from tests.conftest import ActivityFactory


def test_optional_measurements_default_to_none() -> None:
    activity = Activity("a1", datetime(2024, 1, 1), ActivityType.RIDE)
    assert activity.distance_m is None
    assert activity.avg_hr is None
    assert activity.avg_temp_c is None


def test_activity_is_immutable(make_activity: ActivityFactory) -> None:
    activity = make_activity()
    with pytest.raises(FrozenInstanceError):
        activity.distance_m = 1.0  # type: ignore[misc]


def test_activity_type_is_a_string_enum() -> None:
    assert isinstance(ActivityType.RUN, str)
    assert ActivityType.RUN.value == "Run"
    assert ActivityType("Weight Training") is ActivityType.WEIGHT_TRAINING


def test_empty_athlete_id_is_rejected(make_activity: ActivityFactory) -> None:
    with pytest.raises(DataValidationError, match="athlete_id"):
        make_activity(athlete_id="")


@pytest.mark.parametrize(
    "field",
    ["distance_m", "elapsed_s", "moving_s", "avg_speed_ms", "max_hr", "elevation_gain_m"],
)
def test_negative_quantities_are_rejected(make_activity: ActivityFactory, field: str) -> None:
    with pytest.raises(DataValidationError, match=field):
        make_activity(**{field: -1.0})


def test_negative_temperature_is_allowed(make_activity: ActivityFactory) -> None:
    assert make_activity(avg_temp_c=-12.5).avg_temp_c == -12.5


def test_distance_km(make_activity: ActivityFactory) -> None:
    assert make_activity(distance_m=10_500.0).distance_km == pytest.approx(10.5)
    assert make_activity(distance_m=None).distance_km is None


def test_pace_is_moving_time_per_kilometre(make_activity: ActivityFactory) -> None:
    assert make_activity(distance_m=10_000.0, moving_s=3_000.0).pace_s_per_km == pytest.approx(300)


@pytest.mark.parametrize(
    ("distance_m", "moving_s"), [(None, 3_000.0), (0.0, 3_000.0), (10_000.0, None)]
)
def test_pace_is_none_when_it_cannot_be_computed(
    make_activity: ActivityFactory, distance_m: float | None, moving_s: float | None
) -> None:
    assert make_activity(distance_m=distance_m, moving_s=moving_s).pace_s_per_km is None
