"""Data ingestion: adapters for Strava and Garmin exports.

Importing this package registers the built-in adapters.
"""

from race_predictor.sources import garmin, strava
from race_predictor.sources.base import (
    ActivitySource,
    available_sources,
    detect_source,
    get_source,
    register_source,
)
from race_predictor.sources.garmin import GarminCsvSource
from race_predictor.sources.loader import COLUMNS, athletes_to_frame, iter_athletes, load_athlete
from race_predictor.sources.strava import StravaCsvSource

__all__ = [
    "COLUMNS",
    "ActivitySource",
    "GarminCsvSource",
    "StravaCsvSource",
    "athletes_to_frame",
    "available_sources",
    "detect_source",
    "garmin",
    "get_source",
    "iter_athletes",
    "load_athlete",
    "register_source",
    "strava",
]
