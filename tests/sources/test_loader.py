import logging
from pathlib import Path

import pandas as pd
import pytest

from race_predictor.domain import Athlete
from race_predictor.exceptions import UnsupportedExportFormatError
from race_predictor.sources import (
    COLUMNS,
    GarminCsvSource,
    athletes_to_frame,
    iter_athletes,
    load_athlete,
)
from tests.sources.conftest import GARMIN_HEADER, GARMIN_RUN, STRAVA_HEADER, STRAVA_RUN, WriteCsv

DATA = Path(__file__).parents[2] / "data"


def test_load_athlete_detects_the_format(strava_file: Path) -> None:
    athlete = load_athlete(strava_file)
    assert athlete.id == "strava_strava"
    assert athlete.source == "strava"
    assert len(athlete.activities) == 2


def test_load_athlete_with_explicit_id_and_source(garmin_file: Path) -> None:
    athlete = load_athlete(garmin_file, "me", GarminCsvSource())
    assert athlete.id == "me"
    assert {a.athlete_id for a in athlete.activities} == {"me"}


def test_load_athlete_rejects_unknown_formats(write_csv: WriteCsv) -> None:
    with pytest.raises(UnsupportedExportFormatError):
        load_athlete(write_csv("x.csv", "a,b", ["1,2"]))


def test_iter_athletes_numbers_files_and_skips_unknown_ones(
    write_csv: WriteCsv, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_csv("b.csv", STRAVA_HEADER, [STRAVA_RUN])
    write_csv("a.csv", GARMIN_HEADER, [GARMIN_RUN])
    write_csv("c.csv", "a,b", ["1,2"])
    write_csv("d.csv", STRAVA_HEADER, [STRAVA_RUN])
    with caplog.at_level(logging.WARNING):
        athletes = list(iter_athletes(tmp_path))
    assert [a.id for a in athletes] == ["garmin_1", "strava_2", "strava_3"]
    assert "skipping c.csv" in caplog.text


def test_athletes_to_frame(strava_file: Path, garmin_file: Path) -> None:
    frame = athletes_to_frame([load_athlete(strava_file, "s"), load_athlete(garmin_file, "g")])
    assert tuple(frame.columns) == COLUMNS
    assert len(frame) == 5
    assert frame["type"].tolist()[:2] == ["Run", "Ride"]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert pd.api.types.is_float_dtype(frame["avg_temp_c"])
    assert frame["avg_temp_c"].isna().sum() == 3


def test_athletes_to_frame_of_nothing_has_the_canonical_columns() -> None:
    frame = athletes_to_frame([Athlete("a1", "strava")])
    assert frame.empty
    assert tuple(frame.columns) == COLUMNS


@pytest.mark.skipif(not DATA.exists(), reason="sample exports not present")
def test_sample_exports_load() -> None:
    strava = list(iter_athletes(DATA / "strava_activities"))
    garmin = list(iter_athletes(DATA / "garmin_activities"))
    assert len(strava) > 1
    assert len(garmin) == 2
    assert all(athlete.runs() for athlete in [*strava, *garmin])
