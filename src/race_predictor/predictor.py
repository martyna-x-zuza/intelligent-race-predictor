"""Predict the race times of an athlete from their activity export.

:class:`RacePredictor` wraps one :class:`~race_predictor.models.RaceModel` and adds a range around
each prediction. The range comes from the model's errors on athletes it was not trained on, so it
says how far off a prediction for a new runner typically is. The command line (``race-predictor
predict``) and the Streamlit app are thin layers over this module.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from race_predictor.domain import STANDARD_DISTANCES_M
from race_predictor.evaluation import ModelFactory, grouped_splits, load_dataset
from race_predictor.features import RaceDataset, build_query_dataset
from race_predictor.models import RaceModel, RiegelBaseline
from race_predictor.processing import cleaning_pipeline
from race_predictor.sources import athletes_to_frame, load_athlete

_ATHLETE_ID = "athlete"


@dataclass(frozen=True)
class Prediction:
    """The predicted result of one race.

    Attributes:
        distance_m: Race distance, in metres.
        time_s: Predicted finish time, in seconds.
        low_s: Fast end of the range, in seconds.
        high_s: Slow end of the range, in seconds.
    """

    distance_m: float
    time_s: float
    low_s: float
    high_s: float


def format_time(seconds: float) -> str:
    """Format a duration as ``h:mm:ss``.

    Args:
        seconds: Duration in seconds.

    Returns:
        For example ``"1:45:30"``; ``"-"`` if ``seconds`` is not a finite number.
    """
    if not np.isfinite(seconds):
        return "-"
    whole = round(seconds)
    return f"{whole // 3_600}:{whole % 3_600 // 60:02d}:{whole % 60:02d}"


def load_activities(path: str | Path) -> pd.DataFrame:
    """Load one Strava or Garmin export as an activity table.

    Args:
        path: The export file; the format is detected from its header.

    Returns:
        One row per activity, ready for :meth:`RacePredictor.predict`.
    """
    return athletes_to_frame([load_athlete(path, _ATHLETE_ID)])


class RacePredictor:
    """A model together with the typical error of its predictions."""

    def __init__(
        self,
        model_factory: ModelFactory = RiegelBaseline,
        interval: tuple[float, float] = (0.1, 0.9),
        n_splits: int = 5,
    ) -> None:
        """Create an unfitted predictor.

        Args:
            model_factory: Creates a fresh, unfitted model.
            interval: Lower and upper quantile of the error that bound the range.
            n_splits: Folds of the athlete-grouped cross-validation that measures the error.
        """
        self._make = model_factory
        self._interval = interval
        self._n_splits = n_splits
        self._model: RaceModel | None = None
        self._ratios = (1.0, 1.0)

    @classmethod
    def from_data_dir(cls, data_dir: Path, model_factory: ModelFactory = RiegelBaseline) -> Self:
        """Train a predictor on the Strava and Garmin exports of a directory.

        Args:
            data_dir: Directory with ``strava_activities/`` and ``garmin_activities/``.
            model_factory: Creates a fresh, unfitted model.

        Returns:
            The fitted predictor.
        """
        return cls(model_factory).fit(load_dataset(data_dir))

    def fit(self, ds: RaceDataset) -> Self:
        """Fit the model and measure its error on athletes it has not seen.

        Each fold's model predicts the races of athletes held out from its training. The ratio of
        the true to the predicted pace of those races gives the range.

        Args:
            ds: Training races.

        Returns:
            The predictor itself.
        """
        ratios = []
        for train, test in grouped_splits(ds, self._n_splits):
            held_out = ds.subset(test)
            ratios.append(held_out.y / self._make().fit(ds.subset(train)).predict(held_out))
        low, high = np.quantile(np.concatenate(ratios), self._interval)
        self._ratios = (float(low), float(high))
        self._model = self._make().fit(ds)
        return self

    def predict(
        self,
        activities: pd.DataFrame,
        distances_m: Sequence[float] = tuple(STANDARD_DISTANCES_M.values()),
        race_date: pd.Timestamp | None = None,
        elevation_gain_m: float = 0.0,
    ) -> list[Prediction]:
        """Predict the times of races the athlete could run.

        Args:
            activities: Activity table of one athlete, for example from :func:`load_activities`.
            distances_m: Race distances, in metres; 5 km to marathon by default.
            race_date: Date of the races; the day after the last activity by default.
            elevation_gain_m: Elevation gain of the races, in metres.

        Returns:
            One prediction per distance.

        Raises:
            RuntimeError: If the predictor has not been fitted.
            DataValidationError: If the athlete has too few runs before the race date.
        """
        if self._model is None:
            raise RuntimeError("the predictor must be fitted before predicting")
        cleaned = cleaning_pipeline("predict").run(activities)
        ds = build_query_dataset(cleaned, distances_m, race_date, elevation_gain_m)
        times = self._model.predict(ds) * np.asarray(distances_m, dtype=float) / 1_000.0
        low, high = self._ratios
        return [
            Prediction(distance_m, float(time), float(time * low), float(time * high))
            for distance_m, time in zip(distances_m, times, strict=True)
        ]
