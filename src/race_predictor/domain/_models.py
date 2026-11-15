"""Domain model: activities, athletes and races.

All quantities use canonical units: metres, seconds, metres per second, beats per minute and
degrees Celsius. A missing measurement is ``None``.

These classes enforce only *structural* invariants (non-negative quantities, consistent
ownership). Plausibility checks on messy real-world data, such as impossible speeds or heart
rates, belong to the cleaning pipeline, so that parsers can still represent dirty rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final

from race_predictor.exceptions import DataValidationError


class ActivityType(StrEnum):
    """Kind of recorded activity."""

    RUN = "Run"
    RIDE = "Ride"
    SWIM = "Swim"
    WEIGHT_TRAINING = "Weight Training"
    WORKOUT = "Workout"


class RaceCategory(StrEnum):
    """Standard race distance a run is closest to."""

    FIVE_K = "5k"
    TEN_K = "10k"
    HALF_MARATHON = "half marathon"
    MARATHON = "marathon"
    OTHER = "other"

    @classmethod
    def from_distance(cls, distance_m: float, tolerance: float = 0.1) -> RaceCategory:
        """Classify a distance as the standard race distance it is closest to.

        Args:
            distance_m: Distance covered, in metres.
            tolerance: Allowed relative deviation from the standard distance (0.1 = 10 %).

        Returns:
            The matching category, or ``OTHER`` if no standard distance is within
            ``tolerance``.
        """
        for category, standard_m in STANDARD_DISTANCES_M.items():
            if abs(distance_m - standard_m) <= tolerance * standard_m:
                return category
        return cls.OTHER


STANDARD_DISTANCES_M: Final[dict[RaceCategory, float]] = {
    RaceCategory.FIVE_K: 5_000.0,
    RaceCategory.TEN_K: 10_000.0,
    RaceCategory.HALF_MARATHON: 21_097.5,
    RaceCategory.MARATHON: 42_195.0,
}


#: Activity fields that cannot be negative (temperature can, so it is not listed).
_NON_NEGATIVE_FIELDS: Final = (
    "distance_m",
    "elapsed_s",
    "moving_s",
    "avg_speed_ms",
    "max_speed_ms",
    "avg_hr",
    "max_hr",
    "elevation_gain_m",
    "elevation_loss_m",
)


@dataclass(frozen=True, slots=True)
class Activity:
    """A single recorded activity."""

    athlete_id: str
    date: datetime
    type: ActivityType
    distance_m: float | None = None
    elapsed_s: float | None = None
    moving_s: float | None = None
    avg_speed_ms: float | None = None
    max_speed_ms: float | None = None
    avg_hr: float | None = None
    max_hr: float | None = None
    elevation_gain_m: float | None = None
    elevation_loss_m: float | None = None
    avg_temp_c: float | None = None

    def __post_init__(self) -> None:
        """Check the structural invariants of the activity.

        Raises:
            DataValidationError: If ``athlete_id`` is empty or a quantity is negative.
        """
        if not self.athlete_id:
            raise DataValidationError("athlete_id must not be empty")
        for name in _NON_NEGATIVE_FIELDS:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise DataValidationError(f"{name} must not be negative, got {value}")

    @property
    def distance_km(self) -> float | None:
        """Distance in kilometres.

        Returns:
            The distance in km, or ``None`` if it is unknown.
        """
        return None if self.distance_m is None else self.distance_m / 1000

    @property
    def pace_s_per_km(self) -> float | None:
        """Average moving pace.

        Returns:
            Pace in seconds per km, or ``None`` without a positive distance and a moving time.
        """
        if not self.distance_m or self.moving_s is None:
            return None
        return self.moving_s / (self.distance_m / 1000)


@dataclass(frozen=True, slots=True)
class Race:
    """A run that was raced, together with the standard distance it corresponds to."""

    activity: Activity
    category: RaceCategory

    def __post_init__(self) -> None:
        """Check that the race is a run with a distance.

        Raises:
            DataValidationError: If the activity is not a run or has no positive distance.
        """
        if self.activity.type is not ActivityType.RUN:
            raise DataValidationError(f"a race must be a run, got {self.activity.type}")
        if not self.activity.distance_m:
            raise DataValidationError("a race needs a positive distance")

    @classmethod
    def from_activity(cls, activity: Activity, tolerance: float = 0.1) -> Race:
        """Build a race, deriving the category from the activity's distance.

        Args:
            activity: The raced run.
            tolerance: Allowed relative deviation from a standard distance.

        Returns:
            The race with its :class:`RaceCategory`.

        Raises:
            DataValidationError: If the activity has no distance or is not a run.
        """
        if activity.distance_m is None:
            raise DataValidationError("a race needs a distance")
        return cls(activity, RaceCategory.from_distance(activity.distance_m, tolerance))

    @property
    def time_s(self) -> float | None:
        """Race time: the moving time, falling back to the elapsed time.

        Returns:
            Race time in seconds, or ``None`` if neither time is known.
        """
        if self.activity.moving_s is not None:
            return self.activity.moving_s
        return self.activity.elapsed_s


@dataclass(slots=True)
class Athlete:
    """An athlete and their recorded activities."""

    id: str
    source: str
    activities: list[Activity] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Check the id and that every activity belongs to this athlete.

        Raises:
            DataValidationError: If the id is empty or an activity belongs to someone else.
        """
        if not self.id:
            raise DataValidationError("athlete id must not be empty")
        for activity in self.activities:
            self._check_ownership(activity)

    def _check_ownership(self, activity: Activity) -> None:
        """Check that an activity belongs to this athlete.

        Args:
            activity: The activity to check.

        Raises:
            DataValidationError: If the activity's ``athlete_id`` is not this athlete's id.
        """
        if activity.athlete_id != self.id:
            raise DataValidationError(
                f"activity belongs to {activity.athlete_id!r}, not to athlete {self.id!r}"
            )

    def add_activity(self, activity: Activity) -> None:
        """Append an activity.

        Args:
            activity: The activity; it must belong to this athlete.

        Raises:
            DataValidationError: If the activity belongs to another athlete.
        """
        self._check_ownership(activity)
        self.activities.append(activity)

    def runs(self) -> list[Activity]:
        """Select the athlete's runs.

        Returns:
            The runs only, oldest first.
        """
        return sorted(
            (a for a in self.activities if a.type is ActivityType.RUN), key=lambda a: a.date
        )

    @property
    def hr_max(self) -> float | None:
        """Estimate the athlete's maximum heart rate.

        Returns:
            Highest maximum heart rate ever recorded, in bpm, or ``None`` without readings.
        """
        recorded = [a.max_hr for a in self.activities if a.max_hr is not None]
        return max(recorded) if recorded else None
