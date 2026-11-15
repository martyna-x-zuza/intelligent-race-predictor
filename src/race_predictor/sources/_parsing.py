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
    """Normalise a raw cell to text.

    Args:
        value: A cell as read by pandas: a string, a number, ``NaN`` or ``None``.

    Returns:
        The stripped text, or ``None`` if the cell represents a missing value.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in _MISSING else text


def to_float(value: object) -> float | None:
    """Parse a number such as ``"6,44"``, ``"152"`` or ``3.4``.

    Args:
        value: A raw export cell.

    Returns:
        The number, or ``None`` if the cell is missing or not a number.
    """
    text = _clean(value)
    if text is None:
        return None
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return None
    return None if math.isnan(number) else number


def to_elevation(value: object) -> float | None:
    """Parse an elevation gain or loss, treating negative readings as missing.

    Barometric altimeters produce small negative "losses" (median -0.4 m in the sample
    exports) on flat or indoor activities. They are sensor noise, not data, and dropping the
    whole activity because of them would lose real runs.

    Args:
        value: A raw export cell with an elevation in metres.

    Returns:
        The elevation in metres, or ``None`` if the cell is missing, invalid or negative.
    """
    number = to_float(value)
    return None if number is None or number < 0 else number


def parse_duration(value: object) -> float | None:
    """Convert ``"HH:MM:SS(.f)"``, ``"MM:SS"`` or plain seconds to seconds.

    Args:
        value: A raw export cell with a duration.

    Returns:
        The duration in seconds, or ``None`` if the cell is missing or not a duration.
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
    """Convert units by multiplying a possibly missing value.

    Args:
        value: The value to convert, or ``None`` if it is missing.
        factor: Conversion factor, for example 1000 for kilometres to metres.

    Returns:
        ``value * factor``, or ``None`` if ``value`` is missing.
    """
    return None if value is None else value * factor
