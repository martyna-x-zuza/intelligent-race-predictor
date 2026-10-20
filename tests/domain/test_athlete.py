from datetime import datetime

import pytest

from race_predictor.domain import ActivityType, Athlete
from race_predictor.exceptions import DataValidationError
from tests.conftest import ActivityFactory


def test_new_athlete_has_no_activities() -> None:
    athlete = Athlete("a1", source="strava")
    assert athlete.activities == []
    assert athlete.hr_max is None
    assert athlete.runs() == []


def test_athletes_do_not_share_the_default_activity_list() -> None:
    first, second = Athlete("a1", "strava"), Athlete("a2", "garmin")
    assert first.activities is not second.activities


def test_empty_id_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="id"):
        Athlete("", "strava")


def test_add_activity(make_activity: ActivityFactory) -> None:
    athlete = Athlete("a1", "strava")
    activity = make_activity()
    athlete.add_activity(activity)
    assert athlete.activities == [activity]


def test_foreign_activity_is_rejected(make_activity: ActivityFactory) -> None:
    athlete = Athlete("a1", "strava")
    with pytest.raises(DataValidationError, match="belongs to 'a2'"):
        athlete.add_activity(make_activity(athlete_id="a2"))


def test_constructor_validates_initial_activities(make_activity: ActivityFactory) -> None:
    with pytest.raises(DataValidationError):
        Athlete("a1", "strava", [make_activity(athlete_id="a2")])


def test_runs_are_filtered_and_sorted_oldest_first(make_activity: ActivityFactory) -> None:
    newer = make_activity(date=datetime(2024, 6, 1))
    older = make_activity(date=datetime(2024, 5, 1))
    ride = make_activity(type=ActivityType.RIDE, date=datetime(2024, 4, 1))
    athlete = Athlete("a1", "strava", [newer, ride, older])
    assert athlete.runs() == [older, newer]


def test_hr_max_is_the_highest_recorded_max_hr(make_activity: ActivityFactory) -> None:
    athlete = Athlete(
        "a1",
        "garmin",
        [make_activity(max_hr=171.0), make_activity(max_hr=188.0), make_activity(max_hr=None)],
    )
    assert athlete.hr_max == 188.0
