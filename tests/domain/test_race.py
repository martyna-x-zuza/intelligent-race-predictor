import pytest

from race_predictor.domain import ActivityType, Race, RaceCategory
from race_predictor.exceptions import DataValidationError
from tests.conftest import ActivityFactory


@pytest.mark.parametrize(
    ("distance_m", "expected"),
    [
        (5_000.0, RaceCategory.FIVE_K),
        (5_400.0, RaceCategory.FIVE_K),
        (10_050.0, RaceCategory.TEN_K),
        (21_100.0, RaceCategory.HALF_MARATHON),
        (42_195.0, RaceCategory.MARATHON),
        (7_500.0, RaceCategory.OTHER),
        (80_000.0, RaceCategory.OTHER),
    ],
)
def test_category_from_distance(distance_m: float, expected: RaceCategory) -> None:
    assert RaceCategory.from_distance(distance_m) is expected


def test_category_tolerance_is_configurable() -> None:
    assert RaceCategory.from_distance(5_400.0, tolerance=0.05) is RaceCategory.OTHER


def test_race_derives_category_from_activity(make_activity: ActivityFactory) -> None:
    race = Race.from_activity(make_activity(distance_m=42_000.0))
    assert race.category is RaceCategory.MARATHON


def test_race_time_prefers_moving_time(make_activity: ActivityFactory) -> None:
    race = Race.from_activity(make_activity(moving_s=2_900.0, elapsed_s=3_100.0))
    assert race.time_s == 2_900.0


def test_race_time_falls_back_to_elapsed_time(make_activity: ActivityFactory) -> None:
    race = Race.from_activity(make_activity(moving_s=None, elapsed_s=3_100.0))
    assert race.time_s == 3_100.0


def test_race_time_is_none_without_any_time(make_activity: ActivityFactory) -> None:
    assert Race.from_activity(make_activity(moving_s=None)).time_s is None


@pytest.mark.parametrize("activity_type", [ActivityType.RIDE, ActivityType.SWIM])
def test_only_runs_can_be_races(
    make_activity: ActivityFactory, activity_type: ActivityType
) -> None:
    with pytest.raises(DataValidationError, match="must be a run"):
        Race.from_activity(make_activity(type=activity_type))


@pytest.mark.parametrize("distance_m", [None, 0.0])
def test_race_needs_a_positive_distance(
    make_activity: ActivityFactory, distance_m: float | None
) -> None:
    with pytest.raises(DataValidationError, match="distance"):
        Race.from_activity(make_activity(distance_m=distance_m))
