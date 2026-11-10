"""Adapter for the Garmin Connect activity CSV export (Polish-locale headers)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar, Final

import pandas as pd

from race_predictor.domain import Activity, ActivityType
from race_predictor.sources._base import ActivitySource, Row, register_source
from race_predictor.sources._parsing import parse_duration, scale, to_elevation, to_float

_TYPE_MAP: Final = {
    "Bieganie": ActivityType.RUN,
    "Bieg na bieżni": ActivityType.RUN,
    "Virtual Run": ActivityType.RUN,
    "Run": ActivityType.RUN,
    "Kolarstwo": ActivityType.RIDE,
    "Kolarstwo szosowe": ActivityType.RIDE,
    "Jazda na rowerze treningowym": ActivityType.RIDE,
    "Jazda po drodze szutrowej/nieutwardzonej": ActivityType.RIDE,
    "Virtual Ride": ActivityType.RIDE,
    "Ride": ActivityType.RIDE,
    "Pływanie na basenie": ActivityType.SWIM,
    "Pływanie open water": ActivityType.SWIM,
    "Swim": ActivityType.SWIM,
    "Trening siłowy": ActivityType.WEIGHT_TRAINING,
    "Weight Training": ActivityType.WEIGHT_TRAINING,
}

#: Swim distances above this value are recorded in metres rather than kilometres.
_MAX_SWIM_DISTANCE_KM: Final = 15.0
_KM: Final = 1000.0
_KMH_TO_MS: Final = 1 / 3.6


def _speed_to_ms(value: object, activity_type: ActivityType) -> float | None:
    """Convert a Garmin speed or pace cell to m/s.

    Garmin writes speed differently per sport: runs as pace in ``min/km``, swims as pace in
    ``min/100 m`` and everything else as speed in km/h.

    Args:
        value: The speed or pace cell, for example ``"5:56"`` or ``"30,5"``.
        activity_type: Sport of the activity, which decides how the cell is read.

    Returns:
        The speed in m/s, or ``None`` if the cell is missing or not a positive pace.
    """
    if activity_type not in (ActivityType.RUN, ActivityType.SWIM):
        return scale(to_float(value), _KMH_TO_MS)
    pace_s = parse_duration(value) if ":" in str(value) else scale(to_float(value), 60)
    if pace_s is None or pace_s <= 0:
        return None
    metres = 1000 if activity_type is ActivityType.RUN else 100
    return metres / pace_s


def _parse_garmin_date(value: object) -> datetime | None:
    """Parse a Garmin date such as ``"2025-09-19 21:33:31"``.

    Args:
        value: The ``Data`` cell.

    Returns:
        The parsed date, or ``None`` if the cell is not a date.
    """
    try:
        return datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _first_present(row: Row, *columns: str) -> object:
    """Pick the first of several alternative columns that the export has.

    Column sets differ between export versions (for example older exports have no
    ``"Czas ruchu"``), so a fallback column is given for some fields.

    Args:
        row: One row of the export.
        *columns: Column names in order of preference.

    Returns:
        The value of the first column present in ``row``, or ``None`` if there is none.
    """
    for column in columns:
        if column in row:
            return row[column]
    return None


def _mean_temperature(row: Row) -> float | None:
    """Average the minimum and maximum temperature of an activity.

    Args:
        row: One row of the export.

    Returns:
        The mean of the two temperatures in °C, the one that is present, or ``None``.
    """
    temps = [
        t
        for t in (
            to_float(row.get("Minimalna temperatura")),
            to_float(row.get("Maksymalna temperatura")),
        )
        if t is not None
    ]
    return sum(temps) / len(temps) if temps else None


@register_source("garmin")
class GarminCsvSource(ActivitySource):
    """Reads the Garmin Connect activity CSV export, as in the thesis data."""

    required_columns: ClassVar = frozenset({"Typ aktywności", "Data", "Dystans", "Czas"})

    def read_table(self, path: Path) -> pd.DataFrame:
        """Read the export.

        Args:
            path: The Garmin CSV file.

        Returns:
            The raw table with every cell as a string.
        """
        return pd.read_csv(path, dtype=str)

    def parse_row(self, row: Row, athlete_id: str) -> Activity | None:
        """Convert one Garmin row to canonical units.

        Args:
            row: One row of :meth:`read_table`.
            athlete_id: Id the activity is attributed to.

        Returns:
            The activity, or ``None`` if the row has no parseable date.
        """
        date = _parse_garmin_date(row["Data"])
        if date is None:
            return None
        activity_type = _TYPE_MAP.get(str(row["Typ aktywności"]), ActivityType.WORKOUT)

        distance_km = to_float(row["Dystans"])
        if (
            activity_type is ActivityType.SWIM
            and distance_km is not None
            and distance_km > _MAX_SWIM_DISTANCE_KM
        ):
            distance_km /= _KM

        return Activity(
            athlete_id=athlete_id,
            date=date,
            type=activity_type,
            distance_m=scale(distance_km, _KM),
            elapsed_s=parse_duration(_first_present(row, "Upłynęło czasu", "Czas")),
            moving_s=parse_duration(_first_present(row, "Czas ruchu", "Czas")),
            avg_speed_ms=_speed_to_ms(
                _first_present(row, "Średnie tempo", "Średnia prędkość"), activity_type
            ),
            max_speed_ms=_speed_to_ms(
                _first_present(row, "Najlepsze tempo", "Maksymalna prędkość"), activity_type
            ),
            avg_hr=to_float(row.get("Średnie tętno")),
            max_hr=to_float(row.get("Maksymalne tętno")),
            elevation_gain_m=to_elevation(row.get("Całkowity wznios")),
            elevation_loss_m=to_elevation(row.get("Całkowity spadek")),
            avg_temp_c=_mean_temperature(row),
        )
