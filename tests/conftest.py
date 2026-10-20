"""Shared test helpers."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

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
