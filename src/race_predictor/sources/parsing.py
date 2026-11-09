"""Helpers that turn raw export cells into canonical numbers.

Exports are messy: decimal commas (``"6,44"``), placeholders (``"--"``), empty cells read by
pandas as ``NaN``. Every helper returns ``None`` for a missing or unparseable value, which is
how the domain model represents a missing measurement.
"""

from __future__ import annotations

import math
from typing import Final

#: Cell contents that mean "no value" in Strava and Garmin exports.
_MISSING: Final = frozenset({"", "--", "none", "nan"})


def _clean(value: object) -> str | None:
    """Return ``value`` as stripped text, or ``None`` if it represents a missing value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in _MISSING else text


def to_float(value: object) -> float | None:
    """Parse numbers like ``"6,44"``, ``"152"`` or ``3.4``; ``None`` if missing or invalid."""
    text = _clean(value)
    if text is None:
        return None
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return None
    return None if math.isnan(number) else number


def to_elevation(value: object) -> float | None:
    """Parse an elevation gain or loss; negative readings are treated as missing.

    Barometric altimeters produce small negative "losses" (median -0.4 m in the sample
    exports) on flat or indoor activities. They are sensor noise, not data, and dropping the
    whole activity because of them would lose real runs.
    """
    number = to_float(value)
    return None if number is None or number < 0 else number


def parse_duration(value: object) -> float | None:
    """Convert ``"HH:MM:SS(.f)"``, ``"MM:SS"`` or plain seconds to seconds.

    Returns:
        The duration in seconds, or ``None`` if ``value`` is missing or not a duration.
    """
    text = _clean(value)
    if text is None:
        return None
    seconds = 0.0
    for part in text.split(":"):
        number = to_float(part)
        if number is None:
            return None
        seconds = seconds * 60 + number
    return seconds


def scale(value: float | None, factor: float) -> float | None:
    """Multiply a possibly missing value by ``factor`` (unit conversion)."""
    return None if value is None else value * factor
