"""Shared test helpers."""

from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from race_predictor.domain import Activity, ActivityType

ActivityFactory = Callable[..., Activity]


@pytest.fixture
def make_activity() -> ActivityFactory:
    """Build an :class:`Activity` with sensible defaults; override any field by keyword."""

    def factory(**overrides: Any) -> Activity:
        defaults: dict[str, Any] = {
            "athlete_id": "a1",
            "date": datetime(2024, 5, 1, 8, 0),
            "type": ActivityType.RUN,
            "distance_m": 10_000.0,
            "moving_s": 3_000.0,
        }
        return Activity(**{**defaults, **overrides})

    return factory


Frame = Callable[[Iterable[dict[str, Any]]], pd.DataFrame]
Days = Callable[[float], pd.Timestamp]

_DEFAULTS: dict[str, Any] = {
    "source": "strava",
    "athlete_id": "a",
    "type": "Run",
    "distance_m": 8000.0,
    "elapsed_s": 2500.0,
    "moving_s": 2400.0,
    "avg_speed_ms": 3.3,
    "max_speed_ms": 4.5,
    "avg_hr": 145.0,
    "max_hr": 170.0,
    "elevation_gain_m": 20.0,
    "elevation_loss_m": 20.0,
    "avg_temp_c": 15.0,
}


@pytest.fixture
def days() -> Days:
    """``days(n)``: midnight ``n`` days after 2024-01-01."""
    return lambda n: pd.Timestamp("2024-01-01") + pd.Timedelta(days=n)


@pytest.fixture
def frame() -> Frame:
    """Activity table from row overrides; every row needs at least a ``date``."""
    return lambda rows: pd.DataFrame([{**_DEFAULTS, **row} for row in rows])
