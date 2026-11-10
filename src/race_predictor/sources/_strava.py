"""Adapter for the Strava ``activities.csv`` bulk export."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar, Final

import pandas as pd
from dateutil import parser as date_parser

from race_predictor.domain import Activity, ActivityType
from race_predictor.sources._base import ActivitySource, Row, register_source
from race_predictor.sources._parsing import scale, to_elevation, to_float

_TYPE_MAP: Final = {
    "Run": ActivityType.RUN,
    "Virtual Run": ActivityType.RUN,
    "Ride": ActivityType.RIDE,
    "Virtual Ride": ActivityType.RIDE,
    "Swim": ActivityType.SWIM,
    "Weight Training": ActivityType.WEIGHT_TRAINING,
}

_KM: Final = 1000.0


def _parse_strava_date(value: object) -> datetime | None:
    """Parse a Strava date such as ``"Jul 17, 2021, 3:01:35 PM"`` or ``"4 Jul 2017, 07:52:56"``.

    Strava formats dates according to the account's locale, so both orders occur in exports.

    Args:
        value: The ``Activity Date`` cell.

    Returns:
        The parsed date, or ``None`` if the cell is not a date.
    """
    try:
        return date_parser.parse(str(value))
    except (ValueError, OverflowError):
        return None


@register_source("strava")
class StravaCsvSource(ActivitySource):
    """Reads the Strava ``activities.csv`` export.

    Units: ``Distance`` is in kilometres, times in seconds, speeds in m/s.
    """

    required_columns: ClassVar = frozenset(
        {"Activity Date", "Activity Type", "Activity Name", "Moving Time"}
    )

    def read_table(self, path: Path) -> pd.DataFrame:
        """Read the export, merging its duplicated maximum heart rate columns.

        Args:
            path: The ``activities.csv`` file.

        Returns:
            The raw table with every cell as a string.
        """
        table = pd.read_csv(path, dtype=str)
        # The export repeats some column names; pandas suffixes the second copy with ".1".
        # The first "Max Heart Rate" is often empty while the second one is filled.
        if "Max Heart Rate.1" in table:
            table["Max Heart Rate"] = table["Max Heart Rate"].fillna(table["Max Heart Rate.1"])
        return table

    def parse_row(self, row: Row, athlete_id: str) -> Activity | None:
        """Convert one Strava row to canonical units.

        Args:
            row: One row of :meth:`read_table`.
            athlete_id: Id the activity is attributed to.

        Returns:
            The activity, or ``None`` if the row has no parseable date.
        """
        date = _parse_strava_date(row["Activity Date"])
        if date is None:
            return None
        return Activity(
            athlete_id=athlete_id,
            date=date,
            type=_TYPE_MAP.get(str(row["Activity Type"]), ActivityType.WORKOUT),
            distance_m=scale(to_float(row.get("Distance")), _KM),
            elapsed_s=to_float(row.get("Elapsed Time")),
            moving_s=to_float(row.get("Moving Time")),
            avg_speed_ms=to_float(row.get("Average Speed")),
            max_speed_ms=to_float(row.get("Max Speed")),
            avg_hr=to_float(row.get("Average Heart Rate")),
            max_hr=to_float(row.get("Max Heart Rate")),
            elevation_gain_m=to_elevation(row.get("Elevation Gain")),
            elevation_loss_m=to_elevation(row.get("Elevation Loss")),
            avg_temp_c=to_float(row.get("Average Temperature")),
        )
