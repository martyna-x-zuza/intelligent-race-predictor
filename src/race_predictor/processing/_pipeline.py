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
_REQUIRED_COLUMNS: Final = frozenset(
    {"athlete_id", "date", "type", "distance_m", "avg_speed_ms", "max_speed_ms", "avg_hr", "max_hr"}
)


class Step(ABC):
    """One transformation of the activity table."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the activity table.

        Args:
            df: The activity table; it must not be modified.

        Returns:
            The transformed table.
        """

    def __repr__(self) -> str:
        """Show the step by its class name.

        Returns:
            For example ``"RemoveAnomalies()"``.
        """
        return f"{type(self).__name__}()"


class Pipeline:
    """An ordered chain of :class:`Step` objects applied one after another."""

    def __init__(self, steps: Iterable[Step]) -> None:
        """Create the pipeline.

        Args:
            steps: Steps in the order they are applied.
        """
        self._steps: tuple[Step, ...] = tuple(steps)

    def __iter__(self) -> Iterator[Step]:
        """Iterate over the steps.

        Returns:
            An iterator over the steps, in order.
        """
        return iter(self._steps)

    def __len__(self) -> int:
        """Count the steps.

        Returns:
            The number of steps.
        """
        return len(self._steps)

    def __repr__(self) -> str:
        """Show the pipeline with its steps.

        Returns:
            For example ``"Pipeline([RemoveAnomalies(), EstimateHRMax()])"``.
        """
        return f"Pipeline({list(self._steps)!r})"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check the required columns, then pass the table through every step.

        Args:
            df: The activity table, as built by ``athletes_to_frame``.

        Returns:
            The table after the last step.

        Raises:
            DataValidationError: If a required column is missing.
        """
        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataValidationError(f"missing columns: {sorted(missing)}")
        for step in self._steps:
            df = step.apply(df)
        return df


def _is_run(df: pd.DataFrame) -> pd.Series:
    """Flag the runs of an activity table.

    Args:
        df: The activity table.

    Returns:
        Boolean Series aligned to ``df.index``, ``True`` for runs.
    """
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
        """Create the step.

        Args:
            max_avg_speed_ms: Highest plausible average speed of a run, in m/s.
            max_peak_speed_ms: Highest plausible peak speed of a run, in m/s.
            min_distance_m: Shortest run that is kept, in metres.
        """
        self._max_avg_speed_ms = max_avg_speed_ms
        self._max_peak_speed_ms = max_peak_speed_ms
        self._min_distance_m = min_distance_m

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop implausible runs; other activities are kept.

        Args:
            df: The activity table.

        Returns:
            The table without implausible runs.
        """
        bad = _is_run(df) & (
            (df["avg_speed_ms"] > self._max_avg_speed_ms)
            | (df["max_speed_ms"] > self._max_peak_speed_ms)
            | (df["distance_m"] < self._min_distance_m)
        )
        return df[~bad].reset_index(drop=True)


class RemoveImplausiblyFastRuns(Step):
    """Drop runs much faster than the athlete's median run speed.

    Genuine runs in the sample data reach at most about 1.6x the athlete's median; faster
    "runs" are rides logged with the wrong type. Left in, such a run becomes the athlete's best
    pace and distorts every race prediction that uses it.
    """

    def __init__(self, max_rel_speed: float = 1.7) -> None:
        """Create the step.

        Args:
            max_rel_speed: Fastest kept run, as a multiple of the athlete's median run speed.
        """
        self._max_rel_speed = max_rel_speed

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop runs above ``max_rel_speed`` times the athlete's median.

        Args:
            df: The activity table.

        Returns:
            The table without implausibly fast runs.
        """
        runs = _is_run(df)
        median = df["avg_speed_ms"].where(runs).groupby(df["athlete_id"]).transform("median")
        bad = runs & (df["avg_speed_ms"] > self._max_rel_speed * median)
        return df[~bad].reset_index(drop=True)


class ValidateMaxHeartRate(Step):
    """Blank out maximum heart rate readings that are sensor spikes.

    :class:`EstimateHRMax` takes each athlete's highest reading, so a single spike would
    inflate HRmax and with it the heart-rate threshold of race detection.
    """

    def __init__(self, high: float = 220.0) -> None:
        """Create the step.

        Args:
            high: Readings above this value, in bpm, are treated as spikes.
        """
        self._high = high

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace maximum heart rates above ``high`` with ``NaN``.

        Args:
            df: The activity table.

        Returns:
            The table with spikes removed from ``max_hr``.
        """
        return df.assign(max_hr=df["max_hr"].mask(df["max_hr"] > self._high))


class ValidateHeartRate(Step):
    """Keep rows whose average heart rate is physiologically possible.

    Rows without heart rate are dropped, unless ``keep_missing`` is set (they can then be
    imputed by :class:`ImputeMissing`).
    """

    def __init__(
        self, low: float = 40.0, high: float = 220.0, *, keep_missing: bool = False
    ) -> None:
        """Create the step.

        Args:
            low: Lowest plausible average heart rate, in bpm.
            high: Highest plausible average heart rate, in bpm.
            keep_missing: Keep rows without an average heart rate.
        """
        self._low = low
        self._high = high
        self._keep_missing = keep_missing

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with an implausible (or, by default, missing) average heart rate.

        Args:
            df: The activity table.

        Returns:
            The table with only plausible heart rates.
        """
        ok = df["avg_hr"].between(self._low, self._high)
        if self._keep_missing:
            ok |= df["avg_hr"].isna()
        return df[ok].reset_index(drop=True)


class DropAthletesWithoutHR(Step):
    """Drop every athlete who has no heart-rate reading at all."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only athletes with at least one heart-rate reading.

        Args:
            df: The activity table.

        Returns:
            The table without athletes who never recorded heart rate.
        """
        has_hr = df[["avg_hr", "max_hr"]].notna().any(axis=1).groupby(df["athlete_id"]).any()
        keep = has_hr[has_hr].index
        return df[df["athlete_id"].isin(keep)].reset_index(drop=True)


class EstimateHRMax(Step):
    """Estimate each athlete's maximum heart rate."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``hr_max``: the athlete's highest observed maximum heart rate.

        Args:
            df: The activity table.

        Returns:
            The table with the ``hr_max`` column.
        """
        return df.assign(hr_max=df.groupby("athlete_id")["max_hr"].transform("max"))


class MarkTriathlonRuns(Step):
    """Mark runs that start shortly after a ride (brick sessions or triathlon run legs)."""

    def __init__(self, max_gap_hours: float = 6.0) -> None:
        """Create the step.

        Args:
            max_gap_hours: Longest gap between the start of the ride and the start of the
                run, in hours (the exports record start times only).
        """
        self._max_gap_hours = max_gap_hours

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the ``is_triathlon`` flag.

        A run gets 1 if the athlete's previous activity is a ride that started at most
        ``max_gap_hours`` earlier, otherwise 0.

        Args:
            df: The activity table.

        Returns:
            The table sorted by athlete and date, with the ``is_triathlon`` column.
        """
        df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
        by_athlete = df.groupby("athlete_id")
        previous_type = by_athlete["type"].shift()
        gap_hours = (df["date"] - by_athlete["date"].shift()).dt.total_seconds() / 3600
        after_ride = (
            _is_run(df) & (previous_type == ActivityType.RIDE) & (gap_hours <= self._max_gap_hours)
        )
        return df.assign(is_triathlon=after_ride.astype(int))


class DropTriathlonRaces(Step):
    """Keep runs after a ride as training history, but never as road races to predict.

    The race detector cannot tell brick training sessions from triathlon run legs, and run legs
    are slower than road races of the same distance, so neither belongs to the races a road-race
    model learns from. Must run after :class:`~race_predictor.processing.DetectRaces`.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set ``is_race`` to ``False`` for runs after a ride.

        Args:
            df: The activity table with ``is_race`` and ``is_triathlon``.

        Returns:
            The table with updated ``is_race``; no rows are removed.
        """
        return df.assign(is_race=df["is_race"] & (df["is_triathlon"] != 1))


class ImputeMissing(Step):
    """Fill missing numbers with the athlete's median for that activity type.

    Falls back to the median of that activity type over all athletes. ``hr_max`` is never
    imputed: it is already a per-athlete aggregate.
    """

    def __init__(self, columns: Iterable[str] | None = None) -> None:
        """Create the step.

        Args:
            columns: Columns to impute; all numeric columns except ``hr_max`` by default.
        """
        self._columns = None if columns is None else list(columns)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute the missing values.

        Args:
            df: The activity table.

        Returns:
            A copy of the table with the gaps filled where a median exists.
        """
        df = df.copy()
        numeric = [str(c) for c in df.select_dtypes("number").columns if c != "hr_max"]
        for column in self._columns or numeric:
            by_athlete = df.groupby(["athlete_id", "type"])[column].transform("median")
            by_type = df.groupby("type")[column].transform("median")
            df[column] = df[column].fillna(by_athlete).fillna(by_type)
        return df
