"""Data ingestion: adapters for Strava and Garmin exports.

This is the public interface of the package; its modules (``_base``, ``_strava``, ...) are
internal. Importing the package registers the built-in adapters.
"""

from race_predictor.sources import _garmin, _strava  # noqa: F401  (registers the adapters)
from race_predictor.sources._base import (
    ActivitySource,
    Row,
    available_sources,
    detect_source,
    get_source,
    register_source,
)
from race_predictor.sources._garmin import GarminCsvSource
from race_predictor.sources._loader import COLUMNS, athletes_to_frame, iter_athletes, load_athlete
from race_predictor.sources._strava import StravaCsvSource

__all__ = [
    "COLUMNS",
    "ActivitySource",
    "GarminCsvSource",
    "Row",
    "StravaCsvSource",
    "athletes_to_frame",
    "available_sources",
    "detect_source",
    "get_source",
    "iter_athletes",
    "load_athlete",
    "register_source",
]
