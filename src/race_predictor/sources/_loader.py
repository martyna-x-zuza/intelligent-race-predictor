"""Load export files into :class:`~race_predictor.domain.Athlete` objects and DataFrames."""

import logging
from collections.abc import Iterable, Iterator
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from race_predictor.domain import Activity, Athlete
from race_predictor.exceptions import UnsupportedExportFormatError
from race_predictor.sources._base import ActivitySource
from race_predictor.sources._garmin import GarminCsvSource
from race_predictor.sources._strava import StravaCsvSource

_logger = logging.getLogger(__name__)

_SOURCES: tuple[type[ActivitySource], ...] = (StravaCsvSource, GarminCsvSource)

COLUMNS = ("source", *(f.name for f in fields(Activity)))
_NUMERIC_COLUMNS = tuple(c for c in COLUMNS if c not in {"source", "athlete_id", "date", "type"})


def detect_source(path: str | Path) -> ActivitySource:
    """Find the source that can read a file.

    Args:
        path: The export file.

    Returns:
        A new instance of the first source that recognises the file.

    Raises:
        UnsupportedExportFormatError: If no source can read ``path``.
    """
    path = Path(path)
    for source in _SOURCES:
        if source.can_read(path):
            return source()
    known = [source.name for source in _SOURCES]
    raise UnsupportedExportFormatError(
        f"{path.name} is not a supported export (known formats: {known})"
    )


def load_athlete(
    path: str | Path, athlete_id: str | None = None, source: ActivitySource | None = None
) -> Athlete:
    """Load one export file as one athlete.

    Args:
        path: The export file.
        athlete_id: Id given to the athlete; defaults to ``"<source>_<file stem>"``.
        source: Adapter to use; detected from the file header when omitted.

    Returns:
        The athlete with all readable activities of the file.

    Raises:
        UnsupportedExportFormatError: If ``source`` is omitted and no adapter recognises the file.
    """
    path = Path(path)
    source = source or detect_source(path)
    athlete_id = athlete_id or f"{source.name}_{path.stem}"
    return Athlete(athlete_id, source.name, list(source.read(path, athlete_id)))


def iter_athletes(directory: str | Path) -> Iterator[Athlete]:
    """Lazily load every CSV export in a directory, one athlete per file.

    Files are visited in name order and athletes are numbered ``"<source>_1"``,
    ``"<source>_2"``, ... in that order, so ids do not depend on (possibly personal) file names.
    Files in an unsupported format are skipped with a warning.

    Args:
        directory: Directory with the export files.

    Yields:
        One athlete per supported file.
    """
    index = 0
    for path in sorted(Path(directory).glob("*.csv")):
        try:
            source = detect_source(path)
        except UnsupportedExportFormatError:
            _logger.warning("skipping %s: not a supported export", path.name)
            continue
        index += 1
        yield load_athlete(path, f"{source.name}_{index}", source)


def athletes_to_frame(athletes: Iterable[Athlete]) -> pd.DataFrame:
    """Flatten athletes into one activity table.

    Missing measurements become ``NaN`` and ``type`` holds the plain string value, so the
    table is ready for the cleaning and feature pipeline.

    Args:
        athletes: Athletes to flatten.

    Returns:
        One row per activity, with the canonical :data:`COLUMNS`.
    """
    records = [
        {"source": athlete.source, **asdict(activity)}
        for athlete in athletes
        for activity in athlete.activities
    ]
    frame = pd.DataFrame.from_records(records, columns=list(COLUMNS))
    frame["type"] = frame["type"].map(str)
    frame["date"] = pd.to_datetime(frame["date"])
    frame[list(_NUMERIC_COLUMNS)] = frame[list(_NUMERIC_COLUMNS)].astype(float)
    return frame
