from datetime import datetime
from pathlib import Path

import pytest

from race_predictor.domain import ActivityType
from race_predictor.sources import StravaCsvSource
from race_predictor.sources.strava import parse_strava_date
from tests.sources.conftest import STRAVA_HEADER, WriteCsv


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Jul 17, 2021, 3:01:35 PM", datetime(2021, 7, 17, 15, 1, 35)),
        ("4 Jul 2017, 07:52:56", datetime(2017, 7, 4, 7, 52, 56)),
    ],
)
def test_parse_strava_date_handles_both_locales(text: str, expected: datetime) -> None:
    assert parse_strava_date(text) == expected


@pytest.mark.parametrize("value", ["not a date", "", None])
def test_parse_strava_date_returns_none_for_invalid_values(value: object) -> None:
    assert parse_strava_date(value) is None


def test_reads_a_run_in_canonical_units(strava_file: Path) -> None:
    run, _ = StravaCsvSource().read(strava_file, "a1")
    assert run.athlete_id == "a1"
    assert run.type is ActivityType.RUN
    assert run.distance_m == pytest.approx(10_000)
    assert run.moving_s == 2900
    assert run.elapsed_s == 3000
    assert run.avg_speed_ms == pytest.approx(3.4)
    assert run.avg_hr == 150


def test_max_heart_rate_falls_back_to_the_duplicated_column(strava_file: Path) -> None:
    run, ride = StravaCsvSource().read(strava_file, "a1")
    assert run.max_hr == 180  # first column empty, second filled
    assert ride.max_hr == 170  # first column filled


def test_missing_columns_become_none(strava_file: Path) -> None:
    run, _ = StravaCsvSource().read(strava_file, "a1")
    assert run.elevation_gain_m is None
    assert run.avg_temp_c is None


def test_unknown_activity_types_become_workouts(write_csv: WriteCsv) -> None:
    path = write_csv("s.csv", STRAVA_HEADER, ['1,"Jul 17, 2021, 3:01:35 PM",Yoga,Yoga,600,,,,,,'])
    (activity,) = StravaCsvSource().read(path, "a1")
    assert activity.type is ActivityType.WORKOUT
    assert activity.distance_m is None


def test_rows_without_a_date_are_skipped(write_csv: WriteCsv) -> None:
    path = write_csv("s.csv", STRAVA_HEADER, ["1,unknown,Run,Run,600,1.0,,600,,,"])
    assert list(StravaCsvSource().read(path, "a1")) == []


def test_negative_elevation_loss_does_not_drop_the_activity(write_csv: WriteCsv) -> None:
    header = f"{STRAVA_HEADER},Elevation Gain,Elevation Loss"
    path = write_csv(
        "s.csv", header, ['1,"Jul 17, 2021, 3:01:35 PM",Swim,Swim,600,1.0,,600,,,,0,-0.4']
    )
    (swim,) = StravaCsvSource().read(path, "a1")
    assert swim.elevation_gain_m == 0.0
    assert swim.elevation_loss_m is None
