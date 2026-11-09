"""Race detection strategies.

The exports do not say which runs were races, so a detector decides. Detectors are
interchangeable (Strategy pattern): anything with a ``detect(df) -> Series[bool]`` method
satisfies the :class:`RaceDetector` protocol, and :class:`DetectRaces` plugs one into the
cleaning pipeline.

:class:`HeuristicRaceDetector` ports ``identify_races`` from the thesis notebook
``personalized_model.ipynb``.
"""

from __future__ import annotations

from typing import Final, Protocol

import numpy as np
import pandas as pd

from race_predictor.domain import STANDARD_DISTANCES_M, ActivityType
from race_predictor.processing.pipeline import Step

#: Runs longer than this score as a race on distance alone.
ULTRA_DISTANCE_M: Final = 40_000.0
#: A run this much faster than the athlete's rolling average pace counts as a speed spike.
SPEED_SPIKE_RATIO: Final = 1.4


class RaceDetector(Protocol):
    """Anything that decides which rows of an activity table are races."""

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Boolean Series aligned to ``df.index``: ``True`` where the row is a race."""
        ...


def is_standard_distance(distance_m: pd.Series, tolerance: float = 0.1) -> pd.Series:
    """Flag distances within ``tolerance`` (relative) of 5 km, 10 km, half or marathon."""
    flags = pd.Series(False, index=distance_m.index)
    for standard_m in STANDARD_DISTANCES_M.values():
        flags |= distance_m.between(standard_m * (1 - tolerance), standard_m * (1 + tolerance))
    return flags


class StandardDistanceDetector:
    """Simplest strategy: every run of a standard race distance is a race."""

    def __init__(self, tolerance: float = 0.1) -> None:
        self.tolerance = tolerance

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Flag runs within ``tolerance`` of a standard race distance."""
        return (df["type"] == ActivityType.RUN) & is_standard_distance(
            df["distance_m"], self.tolerance
        )


class HeuristicRaceDetector:
    """Score-based heuristic from the thesis.

    A run scores 2 points each for a high heart rate (above ``hr_threshold`` x HRmax), an
    ultra distance and a speed spike against the athlete's rolling average, and 1 point each for
    following a ride and for a standard race distance. It is a race if the score reaches
    ``min_score``, it covers at least ``min_distance_m`` and it had little stopped time.

    Needs the ``hr_max`` column (:class:`~race_predictor.processing.pipeline.EstimateHRMax`);
    uses ``is_triathlon`` when present.
    """

    def __init__(
        self,
        hr_threshold: float = 0.86,
        days_window: int = 365,
        min_distance_m: float = 3000.0,
        max_time_ratio: float = 1.1,
        min_score: int = 2,
    ) -> None:
        self.hr_threshold = hr_threshold
        self.days_window = days_window
        self.min_distance_m = min_distance_m
        self.max_time_ratio = max_time_ratio
        self.min_score = min_score

    def _rolling_speed(self, df: pd.DataFrame) -> pd.Series:
        """Mean run speed of each athlete over the trailing ``days_window`` days."""
        out = pd.Series(np.nan, index=df.index)
        runs = df[df["type"] == ActivityType.RUN]
        for _, athlete_runs in runs.groupby("athlete_id"):
            athlete_runs = athlete_runs.sort_values("date")
            rolling = (
                athlete_runs.set_index("date")["avg_speed_ms"]
                .rolling(f"{self.days_window}D", min_periods=1)
                .mean()
            )
            out.loc[athlete_runs.index] = rolling.to_numpy()
        return out

    def reasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """Every scoring condition per row, the score and the decision (for reviewing labels)."""
        after_ride = (
            df["is_triathlon"].astype(bool)
            if "is_triathlon" in df
            else pd.Series(False, index=df.index)
        )
        out = pd.DataFrame(
            {
                "high_hr": df["avg_hr"] > df["hr_max"] * self.hr_threshold,
                "ultra": df["distance_m"] > ULTRA_DISTANCE_M,
                "speed_spike": df["avg_speed_ms"] > SPEED_SPIKE_RATIO * self._rolling_speed(df),
                "after_ride": after_ride,
                "standard_distance": is_standard_distance(df["distance_m"]),
            },
            index=df.index,
        )
        strong = out[["high_hr", "ultra", "speed_spike"]].sum(axis=1)
        weak = out[["after_ride", "standard_distance"]].sum(axis=1)
        out["score"] = 2 * strong + weak
        stopped_ratio = df["elapsed_s"] / df["moving_s"].replace(0, np.nan)
        out["is_race"] = (
            (df["type"] == ActivityType.RUN)
            & (df["distance_m"] >= self.min_distance_m)
            & (out["score"] >= self.min_score)
            & (stopped_ratio <= self.max_time_ratio)
        )
        return out

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Flag the rows whose score reaches ``min_score``."""
        return self.reasons(df)["is_race"]


class DetectRaces(Step):
    """Add the boolean ``is_race`` column using a :class:`RaceDetector`."""

    def __init__(self, detector: RaceDetector | None = None) -> None:
        self.detector: RaceDetector = detector or HeuristicRaceDetector()

    def __repr__(self) -> str:
        return f"DetectRaces({type(self.detector).__name__})"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(is_race=self.detector.detect(df).fillna(False).astype(bool))
