from datetime import datetime
from pathlib import Path

import pytest

from race_predictor.domain import ActivityType
from race_predictor.sources import GarminCsvSource
from race_predictor.sources.garmin import parse_garmin_date, speed_to_ms
from tests.sources.conftest import GARMIN_HEADER, WriteCsv


@pytest.mark.parametrize(
    ("value", "activity_type", "expected"),
    [
        ("5:00", ActivityType.RUN, 1000 / 300),
        ("2:00", ActivityType.SWIM, 100 / 120),
        ("36", ActivityType.RIDE, 10.0),
        ("36,0", ActivityType.WORKOUT, 10.0),
    ],
)
def test_speed_to_ms(value: str, activity_type: ActivityType, expected: float) -> None:
    assert speed_to_ms(value, activity_type) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["--", "0:00", None])
def test_speed_to_ms_returns_none_for_missing_or_zero_pace(value: object) -> None:
    assert speed_to_ms(value, ActivityType.RUN) is None


def test_parse_garmin_date() -> None:
    assert parse_garmin_date("2025-09-19 21:33:31") == datetime(2025, 9, 19, 21, 33, 31)
    assert parse_garmin_date("--") is None


def test_reads_a_run_in_canonical_units(garmin_file: Path) -> None:
    run, _, _ = GarminCsvSource().read(garmin_file, "a1")
    assert run.type is ActivityType.RUN
    assert run.date == datetime(2025, 9, 19, 21, 33, 31)
    assert run.distance_m == pytest.approx(6440)
    assert run.elapsed_s == 2290  # no "Upłynęło czasu" column: falls back to "Czas"
    assert run.moving_s == 2277
    assert run.avg_speed_ms == pytest.approx(1000 / 356)
    assert run.max_hr == 177
    assert run.avg_temp_c == pytest.approx(22.5)


def test_ride_speed_is_in_kmh_and_one_temperature_is_enough(garmin_file: Path) -> None:
    _, ride, _ = GarminCsvSource().read(garmin_file, "a1")
    assert ride.type is ActivityType.RIDE
    assert ride.avg_speed_ms == pytest.approx(30 / 3.6)
    assert ride.avg_temp_c == 18.0


def test_swim_distance_in_metres_is_converted(garmin_file: Path) -> None:
    _, _, swim = GarminCsvSource().read(garmin_file, "a1")
    assert swim.type is ActivityType.SWIM
    assert swim.distance_m == pytest.approx(1500)
    assert swim.avg_hr is None
    assert swim.avg_temp_c is None


def test_rows_without_a_date_are_skipped(write_csv: WriteCsv) -> None:
    path = write_csv("g.csv", GARMIN_HEADER, ["Bieganie,--,5,00:25:00,,,,,,,"])
    assert list(GarminCsvSource().read(path, "a1")) == []
