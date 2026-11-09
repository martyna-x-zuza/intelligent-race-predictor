"""Adapter for the Garmin Connect activity CSV export (Polish-locale headers)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar, Final

import pandas as pd

from race_predictor.domain import Activity, ActivityType
from race_predictor.sources.base import ActivitySource, Row, register_source
from race_predictor.sources.parsing import parse_duration, scale, to_elevation, to_float

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
MAX_SWIM_DISTANCE_KM: Final = 15.0
_KM: Final = 1000.0
_KMH_TO_MS: Final = 1 / 3.6


def speed_to_ms(value: object, activity_type: ActivityType) -> float | None:
    """Convert a Garmin speed or pace cell to m/s.

    Garmin writes speed differently per sport: runs as pace in ``min/km``, swims as pace in
    ``min/100 m`` and everything else as speed in km/h.

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


def parse_garmin_date(value: object) -> datetime | None:
    """Parse ``"2025-09-19 21:33:31"``; ``None`` if ``value`` is not a date."""
    try:
        return datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _first_present(row: Row, *columns: str) -> object:
    """Value of the first of ``columns`` present in ``row``; ``None`` if there is none.

    Column sets differ between export versions (for example older exports have no
    ``"Czas ruchu"``), so a fallback column is given for some fields.
    """
    for column in columns:
        if column in row:
            return row[column]
    return None


def _mean_temperature(row: Row) -> float | None:
    """Mean of the minimum and maximum temperature, or whichever of them is present."""
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
        return pd.read_csv(path, dtype=str)

    def parse_row(self, row: Row, athlete_id: str) -> Activity | None:
        date = parse_garmin_date(row["Data"])
        if date is None:
            return None
        activity_type = _TYPE_MAP.get(str(row["Typ aktywności"]), ActivityType.WORKOUT)

        distance_km = to_float(row["Dystans"])
        if (
            activity_type is ActivityType.SWIM
            and distance_km is not None
            and distance_km > MAX_SWIM_DISTANCE_KM
        ):
            distance_km /= _KM

        return Activity(
            athlete_id=athlete_id,
            date=date,
            type=activity_type,
            distance_m=scale(distance_km, _KM),
            elapsed_s=parse_duration(_first_present(row, "Upłynęło czasu", "Czas")),
            moving_s=parse_duration(_first_present(row, "Czas ruchu", "Czas")),
            avg_speed_ms=speed_to_ms(
                _first_present(row, "Średnie tempo", "Średnia prędkość"), activity_type
            ),
            max_speed_ms=speed_to_ms(
                _first_present(row, "Najlepsze tempo", "Maksymalna prędkość"), activity_type
            ),
            avg_hr=to_float(row.get("Średnie tętno")),
            max_hr=to_float(row.get("Maksymalne tętno")),
            elevation_gain_m=to_elevation(row.get("Całkowity wznios")),
            elevation_loss_m=to_elevation(row.get("Całkowity spadek")),
            avg_temp_c=_mean_temperature(row),
        )
