"""Build one training example per race.

Port of ``build_sequences_and_statics_split`` from the thesis notebook
``personalized_model.ipynb``, without the train/test split baked in (splitting belongs to the
evaluation). Every example has two views of the athlete's preparation:

* a *sequence*: one row per run in the window before the race (for recurrent models),
* *static* features: cross-training totals and hand-crafted summaries of that window (for
  tabular models and the Riegel formula).

The target is race pace in seconds per kilometre.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields

import numpy as np
import numpy.typing as npt
import pandas as pd

from race_predictor.domain import ActivityType
from race_predictor.exceptions import DataValidationError

FloatArray = npt.NDArray[np.float64]

SEQ_FEATURES = (
    "days_to_race",
    "distance_m",
    "elapsed_s",
    "moving_s",
    "avg_speed_ms",
    "max_speed_ms",
    "avg_hr",
    "max_hr",
    "elevation_gain_m",
    "elevation_loss_m",
    "avg_temp_c",
)

_MIN_PACE_RUN_M = 3_000.0
_MIN_REFERENCE_RUN_M = 5_000.0
_SHORT_WINDOW_DAYS = 28
_LONG_WINDOW_DAYS = 90


def _total(frame: pd.DataFrame, column: str) -> float:
    """Sum a column, treating an empty or all-missing column as zero.

    Args:
        frame: The activities to sum over.
        column: Column to sum.

    Returns:
        The sum, or 0.0 if there are no values.
    """
    return float(frame[column].sum(min_count=1)) if frame[column].notna().any() else 0.0


@dataclass(frozen=True)
class CrossTraining:
    """Swimming, cycling and gym load in the window before a race."""

    swim_t_total: float
    swim_dist_total: float
    swim_sessions: float
    bike_t_total: float
    bike_dist_total: float
    bike_sessions: float
    gym_t_total: float
    gym_sessions: float

    @classmethod
    def from_activities(cls, activities: pd.DataFrame) -> "CrossTraining":
        """Total up the non-running activities of a window.

        Args:
            activities: The athlete's activities in the window, runs excluded.

        Returns:
            The time, distance and number of sessions of each sport.
        """
        swim, bike, gym = (
            activities[activities["type"] == sport]
            for sport in (ActivityType.SWIM, ActivityType.RIDE, ActivityType.WEIGHT_TRAINING)
        )
        return cls(
            swim_t_total=_total(swim, "elapsed_s"),
            swim_dist_total=_total(swim, "distance_m"),
            swim_sessions=float(len(swim)),
            bike_t_total=_total(bike, "elapsed_s"),
            bike_dist_total=_total(bike, "distance_m"),
            bike_sessions=float(len(bike)),
            gym_t_total=_total(gym, "elapsed_s"),
            gym_sessions=float(len(gym)),
        )


@dataclass(frozen=True)
class TrainingSummary:
    """Hand-crafted summary of the runs in the window before a race."""

    n_runs: float
    total_km: float
    mean_pace_s_km: float
    best_pace_s_km: float
    longest_run_km: float
    mean_hr: float
    km_last_28d: float
    km_last_90d: float
    weekly_km: float

    @classmethod
    def from_window(
        cls, window: pd.DataFrame, race_date: pd.Timestamp, days_window: int
    ) -> "TrainingSummary":
        """Summarise the runs of a training window.

        Args:
            window: Runs before the race.
            race_date: Date of the race.
            days_window: Length of the window, in days (to compute the weekly volume).

        Returns:
            The summary; ``best_pace_s_km`` is ``NaN`` without a long enough run.
        """
        pace = 1_000.0 / window["avg_speed_ms"].replace(0, np.nan)
        km = window["distance_m"] / 1_000.0
        age_days = (race_date - window["date"]).dt.days
        long_enough = window["distance_m"] >= _MIN_PACE_RUN_M
        return cls(
            n_runs=float(len(window)),
            total_km=float(km.sum()),
            mean_pace_s_km=float(pace.mean()),
            best_pace_s_km=float(pace[long_enough].min()) if long_enough.any() else np.nan,
            longest_run_km=float(km.max()),
            mean_hr=float(window["avg_hr"].mean()),
            km_last_28d=float(km[age_days <= _SHORT_WINDOW_DAYS].sum()),
            km_last_90d=float(km[age_days <= _LONG_WINDOW_DAYS].sum()),
            weekly_km=float(km.sum() / (days_window / 7)),
        )


@dataclass(frozen=True)
class RiegelReference:
    """The performance the Riegel formula projects to the race distance."""

    riegel_ref_distance_m: float
    riegel_ref_time_s: float

    @classmethod
    def from_window(cls, window: pd.DataFrame) -> "RiegelReference":
        """Find the fastest run of at least 5 km in a training window.

        Args:
            window: Runs before the race.

        Returns:
            The distance and time of that run; ``NaN`` for both if the window has no such run.
        """
        candidates = window[
            (window["distance_m"] >= _MIN_REFERENCE_RUN_M) & (window["avg_speed_ms"] > 0)
        ]
        if candidates.empty:
            return cls(np.nan, np.nan)
        best = candidates.iloc[int(np.argmax(candidates["avg_speed_ms"].to_numpy()))]
        distance_m = float(best["distance_m"])
        return cls(distance_m, distance_m / float(best["avg_speed_ms"]))


@dataclass(frozen=True)
class RaceFeatures:
    """The static (non-sequence) features of one race."""

    years_of_experience: float
    race_distance_m: float
    race_elevation_gain_m: float
    is_triathlon: float
    cross_training: CrossTraining
    training: TrainingSummary
    riegel: RiegelReference

    def as_row(self) -> dict[str, float]:
        """Flatten the features into one table row.

        Returns:
            One value per feature, keyed by the column names of :attr:`RaceDataset.static`.
        """
        return {
            "years_of_experience": self.years_of_experience,
            "race_distance_m": self.race_distance_m,
            "race_elevation_gain_m": self.race_elevation_gain_m,
            "is_triathlon": self.is_triathlon,
            **asdict(self.cross_training),
            **asdict(self.training),
            **asdict(self.riegel),
        }


AGGREGATES = tuple(f.name for f in fields(TrainingSummary))
RIEGEL_COLUMNS = tuple(f.name for f in fields(RiegelReference))
STATIC_NUMERIC = (
    "years_of_experience",
    *(f.name for f in fields(CrossTraining)),
    "race_distance_m",
    "race_elevation_gain_m",
)
STATIC_BINARY = ("is_triathlon",)


@dataclass(frozen=True)
class RaceDataset:
    """Features and targets of a set of races, one entry per race.

    Attributes:
        seqs: Runs before each race, each of shape ``(n_runs, len(SEQ_FEATURES))``, oldest
            first.
        static: Per-race tabular features (``STATIC_NUMERIC``, ``STATIC_BINARY``,
            ``AGGREGATES`` and the Riegel reference).
        y: Race pace in seconds per km.
        meta: ``athlete_id``, ``race_date`` and ``race_distance_m`` of each race.
    """

    seqs: list[npt.NDArray[np.float32]]
    static: pd.DataFrame
    y: FloatArray
    meta: pd.DataFrame

    tabular_columns = (*STATIC_NUMERIC, *STATIC_BINARY, *AGGREGATES)

    def __post_init__(self) -> None:
        """Check that all parts describe the same number of races.

        Raises:
            DataValidationError: If the parts have different lengths.
        """
        sizes = {len(self.seqs), len(self.static), len(self.y), len(self.meta)}
        if len(sizes) != 1:
            raise DataValidationError(f"inconsistent race counts: {sorted(sizes)}")

    def __len__(self) -> int:
        """Count the races.

        Returns:
            The number of races.
        """
        return len(self.y)

    def subset(
        self, index: Sequence[int] | npt.NDArray[np.integer] | npt.NDArray[np.bool_]
    ) -> "RaceDataset":
        """Select some of the races.

        Args:
            index: Integer positions of the races, or a boolean mask over all races.

        Returns:
            A new dataset with only the selected races, in the order of ``index``.
        """
        array = np.asarray(index)
        positions = np.flatnonzero(array) if array.dtype == bool else array.astype(np.intp)
        return RaceDataset(
            [self.seqs[i] for i in positions],
            self.static.iloc[positions].reset_index(drop=True),
            self.y[positions],
            self.meta.iloc[positions].reset_index(drop=True),
        )

    @property
    def race_time_s(self) -> FloatArray:
        """Race times computed from the pace and the race distance.

        Returns:
            Race time of every race, in seconds.
        """
        distance_km: FloatArray = self.meta["race_distance_m"].to_numpy(dtype=float) / 1_000.0
        return self.y * distance_km


def _years_of_experience(runs: pd.DataFrame) -> float:
    """Measure how long an athlete has been running.

    Args:
        runs: All runs of the athlete.

    Returns:
        Years between the first and the last run; 0.0 with fewer than two runs.
    """
    if len(runs) < 2:
        return 0.0
    return float((runs["date"].max() - runs["date"].min()).days / 365.0)


def _example(
    activities: pd.DataFrame,
    runs: pd.DataFrame,
    race: pd.Series,
    years_of_experience: float,
    days_window: int,
    min_runs: int,
) -> tuple[RaceFeatures, npt.NDArray[np.float32]] | None:
    """Build the features of one race.

    Only activities strictly before the race date are used, so the race cannot leak into its
    own features.

    Args:
        activities: All activities of the athlete (source of the cross-training totals).
        runs: The athlete's runs (source of the sequence and the run summaries).
        race: The race row.
        years_of_experience: Years the athlete has been running.
        days_window: Length of the window before the race, in days.
        min_runs: Fewest runs the window must have.

    Returns:
        ``(static features, run sequence)``, or ``None`` if the window has too few runs.
    """
    race_date: pd.Timestamp = race["date"]
    start = race_date - pd.Timedelta(days=days_window)
    in_window = (runs["date"] >= start) & (runs["date"] < race_date)
    window = runs[in_window].assign(days_to_race=(race_date - runs["date"]).dt.days)
    if len(window) < min_runs:
        return None

    cross_training = activities[
        (activities["type"] != ActivityType.RUN)
        & (activities["date"] >= start)
        & (activities["date"] < race_date)
    ]
    features = RaceFeatures(
        years_of_experience=years_of_experience,
        race_distance_m=float(race["distance_m"]),
        race_elevation_gain_m=float(race["elevation_gain_m"]),
        is_triathlon=float(race.get("is_triathlon", 0)),
        cross_training=CrossTraining.from_activities(cross_training),
        training=TrainingSummary.from_window(window, race_date, days_window),
        riegel=RiegelReference.from_window(window),
    )
    return features, window[list(SEQ_FEATURES)].to_numpy(dtype=np.float32)


def build_race_dataset(df: pd.DataFrame, days_window: int = 365, min_runs: int = 3) -> RaceDataset:
    """Build one example per race.

    Races without a time or distance, or with fewer than ``min_runs`` runs in the preceding
    ``days_window`` days, are left out.

    Args:
        df: Activity table cleaned by ``cleaning_pipeline("train")``.
        days_window: Length of the window before each race, in days.
        min_runs: Fewest runs a race's window must have.

    Returns:
        The dataset, with races ordered by athlete and date.

    Raises:
        DataValidationError: If no race qualifies.
    """
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
    seqs: list[npt.NDArray[np.float32]] = []
    statics: list[RaceFeatures] = []
    targets: list[float] = []
    metas: list[dict[str, object]] = []

    for athlete_id, activities in df.groupby("athlete_id"):
        runs = activities[activities["type"] == ActivityType.RUN]
        years = _years_of_experience(runs)
        for _, race in runs[runs["is_race"]].iterrows():
            time_s = race["moving_s"] if pd.notna(race["moving_s"]) else race["elapsed_s"]
            if pd.isna(time_s) or not race["distance_m"] > 0:
                continue
            example = _example(activities, runs, race, years, days_window, min_runs)
            if example is None:
                continue
            statics.append(example[0])
            seqs.append(example[1])
            targets.append(float(time_s) / (float(race["distance_m"]) / 1_000.0))
            metas.append(
                {
                    "athlete_id": athlete_id,
                    "race_date": race["date"],
                    "race_distance_m": float(race["distance_m"]),
                }
            )

    if not statics:
        raise DataValidationError(
            f"no race has {min_runs} or more earlier runs in its {days_window}-day window"
        )
    return RaceDataset(
        seqs,
        pd.DataFrame([features.as_row() for features in statics]),
        np.asarray(targets, dtype=np.float64),
        pd.DataFrame(metas),
    )
