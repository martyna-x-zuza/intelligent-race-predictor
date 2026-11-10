import logging
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from race_predictor.domain import Activity, ActivityType
from race_predictor.exceptions import UnsupportedExportFormatError
from race_predictor.sources import (
    GarminCsvSource,
    StravaCsvSource,
    available_sources,
    detect_source,
    get_source,
    register_source,
)
from race_predictor.sources._base import _REGISTRY, ActivitySource, Row, _read_header
from tests.sources.conftest import WriteCsv


class _TinySource(ActivitySource):
    """Minimal adapter: column ``day`` is a date, ``km`` a distance."""

    required_columns: ClassVar = frozenset({"day", "km"})

    def read_table(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path, dtype=str)

    def parse_row(self, row: Row, athlete_id: str) -> Activity | None:
        if row["day"] == "none":
            return None
        return Activity(
            athlete_id,
            pd.Timestamp(row["day"]).to_pydatetime(),
            ActivityType.RUN,
            distance_m=float(row["km"]) * 1000,
        )


@pytest.fixture
def tiny_source() -> Iterator[type[_TinySource]]:
    registered = register_source("tiny")(_TinySource)
    yield registered
    del _REGISTRY["tiny"]


def test_builtin_sources_are_registered() -> None:
    assert {"strava", "garmin"} <= set(available_sources())
    assert isinstance(get_source("strava"), StravaCsvSource)
    assert isinstance(get_source("garmin"), GarminCsvSource)


def test_unknown_source_name_is_rejected() -> None:
    with pytest.raises(UnsupportedExportFormatError, match="unknown source"):
        get_source("polar")


def test_register_source_sets_the_name(tiny_source: type[_TinySource]) -> None:
    assert tiny_source.name == "tiny"
    assert "tiny" in available_sources()


def test_registering_a_different_class_under_a_taken_name_fails() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_source("strava")(_TinySource)


def test_detect_source(strava_file: Path, garmin_file: Path) -> None:
    assert detect_source(strava_file).name == "strava"
    assert detect_source(garmin_file).name == "garmin"


def test_detect_source_rejects_unknown_formats(write_csv: WriteCsv) -> None:
    path = write_csv("other.csv", "a,b", ["1,2"])
    with pytest.raises(UnsupportedExportFormatError, match=r"other\.csv"):
        detect_source(path)


def test_read_header_of_unreadable_files(tmp_path: Path) -> None:
    assert _read_header(tmp_path / "missing.csv") == []
    binary = tmp_path / "binary.csv"
    binary.write_bytes(b"\xff\xfe\x00\x81")
    assert _read_header(binary) == []


def test_read_rejects_a_file_in_another_format(garmin_file: Path) -> None:
    with pytest.raises(UnsupportedExportFormatError, match="strava"):
        list(StravaCsvSource().read(garmin_file, "a1"))


def test_read_is_lazy(tiny_source: type[_TinySource], write_csv: WriteCsv) -> None:
    path = write_csv("tiny.csv", "day,km", ["2024-01-01,5", "2024-01-02,10"])
    activities = tiny_source().read(path, "a1")
    assert isinstance(activities, Iterator)
    assert next(activities).distance_m == 5000.0


def test_read_skips_undated_and_invalid_rows(
    tiny_source: type[_TinySource], write_csv: WriteCsv, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_csv("tiny.csv", "day,km", ["2024-01-01,5", "none,5", "2024-01-03,-1"])
    with caplog.at_level(logging.WARNING):
        activities = list(tiny_source().read(path, "a1"))
    assert [a.distance_m for a in activities] == [5000.0]
    assert "skipped 2 unreadable row(s)" in caplog.text
