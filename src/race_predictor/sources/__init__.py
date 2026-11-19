"""Data ingestion: adapters for Strava and Garmin exports.

This is the public interface of the package; its modules (``_base``, ``_strava``, ...) are
internal.
"""

from race_predictor.sources._base import ActivitySource, Row
from race_predictor.sources._garmin import GarminCsvSource
from race_predictor.sources._loader import (
    COLUMNS,
    athletes_to_frame,
    detect_source,
    iter_athletes,
    load_athlete,
)
from race_predictor.sources._strava import StravaCsvSource

__all__ = [
    "COLUMNS",
    "ActivitySource",
    "GarminCsvSource",
    "Row",
    "StravaCsvSource",
    "athletes_to_frame",
    "detect_source",
    "iter_athletes",
    "load_athlete",
]
