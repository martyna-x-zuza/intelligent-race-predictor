"""Cleaning pipeline: a chain of small, independently testable steps.

Each :class:`Step` takes the activity table produced by
:func:`~race_predictor.sources.athletes_to_frame` and returns a new one (never modifying its
input), so steps can be reordered, switched off or tested in isolation. :class:`Pipeline` runs
them in order (Chain of Responsibility / Pipeline pattern).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Final

import pandas as pd

from race_predictor.domain import ActivityType
from race_predictor.exceptions import DataValidationError

#: Columns every pipeline input must have.
REQUIRED_COLUMNS: Final = frozenset(
    {"athlete_id", "date", "type", "distance_m", "avg_speed_ms", "max_speed_ms", "avg_hr", "max_hr"}
)


class Step(ABC):
    """One transformation of the activity table."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed table; ``df`` itself must not be modified."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Pipeline:
    """An ordered chain of :class:`Step` objects applied one after another."""

    def __init__(self, steps: Iterable[Step]) -> None:
        self.steps: tuple[Step, ...] = tuple(steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"Pipeline({list(self.steps)!r})"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check the required columns, then pass ``df`` through every step.

        Raises:
            DataValidationError: If a required column is missing.
        """
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataValidationError(f"missing columns: {sorted(missing)}")
        for step in self.steps:
            df = step.apply(df)
        return df


def _is_run(df: pd.DataFrame) -> pd.Series:
    return df["type"] == ActivityType.RUN


class RemoveAnomalies(Step):
    """Drop runs that are physically implausible or too short to be informative.

    Defaults follow the thesis: average speed above 8 m/s, peak above 15 m/s or under 1 km.
    """

    def __init__(
        self,
        max_avg_speed_ms: float = 8.0,
        max_peak_speed_ms: float = 15.0,
        min_distance_m: float = 1000.0,
    ) -> None:
        self.max_avg_speed_ms = max_avg_speed_ms
        self.max_peak_speed_ms = max_peak_speed_ms
        self.min_distance_m = min_distance_m

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        bad = _is_run(df) & (
            (df["avg_speed_ms"] > self.max_avg_speed_ms)
            | (df["max_speed_ms"] > self.max_peak_speed_ms)
            | (df["distance_m"] < self.min_distance_m)
        )
        return df[~bad].reset_index(drop=True)


class RemoveImplausiblyFastRuns(Step):
    """Drop runs faster than ``max_rel_speed`` times the athlete's median run speed.

    Genuine runs in the sample data reach at most about 1.6x the athlete's median; faster
    "runs" are rides logged with the wrong type. Left in, such a run becomes the athlete's best
    pace and distorts every race prediction that uses it.
    """

    def __init__(self, max_rel_speed: float = 1.7) -> None:
        self.max_rel_speed = max_rel_speed

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        runs = _is_run(df)
        median = df["avg_speed_ms"].where(runs).groupby(df["athlete_id"]).transform("median")
        bad = runs & (df["avg_speed_ms"] > self.max_rel_speed * median)
        return df[~bad].reset_index(drop=True)


class ValidateMaxHeartRate(Step):
    """Blank out maximum heart rate readings above ``high`` bpm (sensor spikes).

    :class:`EstimateHRMax` takes each athlete's highest reading, so a single spike would
    inflate HRmax and with it the heart-rate threshold of race detection.
    """

    def __init__(self, high: float = 220.0) -> None:
        self.high = high

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(max_hr=df["max_hr"].mask(df["max_hr"] > self.high))


class ValidateHeartRate(Step):
    """Keep rows whose average heart rate is physiologically possible.

    Rows without heart rate are dropped, unless ``keep_missing`` is set (they can then be
    imputed by :class:`ImputeMissing`).
    """

    def __init__(
        self, low: float = 40.0, high: float = 220.0, *, keep_missing: bool = False
    ) -> None:
        self.low = low
        self.high = high
        self.keep_missing = keep_missing

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ok = df["avg_hr"].between(self.low, self.high)
        if self.keep_missing:
            ok |= df["avg_hr"].isna()
        return df[ok].reset_index(drop=True)


class DropAthletesWithoutHR(Step):
    """Drop every athlete who has no heart-rate reading at all."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        has_hr = df[["avg_hr", "max_hr"]].notna().any(axis=1).groupby(df["athlete_id"]).any()
        keep = has_hr[has_hr].index
        return df[df["athlete_id"].isin(keep)].reset_index(drop=True)


class EstimateHRMax(Step):
    """Add ``hr_max``: the athlete's highest observed maximum heart rate."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(hr_max=df.groupby("athlete_id")["max_hr"].transform("max"))


class MarkTriathlonRuns(Step):
    """Add ``is_triathlon``: 1 for a run starting within ``max_gap_hours`` after a ride."""

    def __init__(self, max_gap_hours: float = 6.0) -> None:
        self.max_gap_hours = max_gap_hours

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
        by_athlete = df.groupby("athlete_id")
        previous_type = by_athlete["type"].shift()
        gap_hours = (df["date"] - by_athlete["date"].shift()).dt.total_seconds() / 3600
        after_ride = (
            _is_run(df) & (previous_type == ActivityType.RIDE) & (gap_hours <= self.max_gap_hours)
        )
        return df.assign(is_triathlon=after_ride.astype(int))


class DropTriathlonRaces(Step):
    """Keep runs after a ride as training history, but never as road races to predict.

    The race detector cannot tell brick training sessions from triathlon run legs, and run legs
    are slower than road races of the same distance, so neither belongs to the races a road-race
    model learns from. Must run after :class:`~race_predictor.processing.races.DetectRaces`.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(is_race=df["is_race"] & (df["is_triathlon"] != 1))


class ImputeMissing(Step):
    """Fill missing numbers with the athlete's median for that activity type.

    Falls back to the median of that activity type over all athletes. ``hr_max`` is never
    imputed: it is already a per-athlete aggregate.
    """

    def __init__(self, columns: Iterable[str] | None = None) -> None:
        self.columns = None if columns is None else list(columns)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric = [str(c) for c in df.select_dtypes("number").columns if c != "hr_max"]
        for column in self.columns or numeric:
            by_athlete = df.groupby(["athlete_id", "type"])[column].transform("median")
            by_type = df.groupby("type")[column].transform("median")
            df[column] = df[column].fillna(by_athlete).fillna(by_type)
        return df
