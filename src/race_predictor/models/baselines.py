"""Baselines: the Riegel formula, ridge regression and gradient boosting."""

from __future__ import annotations

from typing import Any, ClassVar, Final, Self, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from race_predictor.features import RaceDataset
from race_predictor.features.dataset import FloatArray
from race_predictor.models.base import RaceModel

#: Fatigue exponent of Riegel (1981).
RIEGEL_EXPONENT: Final = 1.06

_Number = TypeVar("_Number", float, npt.NDArray[np.float64], "pd.Series[float]")


def riegel_time(
    ref_time_s: _Number,
    ref_distance_m: _Number,
    target_distance_m: _Number,
    exponent: float = RIEGEL_EXPONENT,
) -> _Number:
    """Riegel's formula ``T2 = T1 * (D2 / D1) ** exponent``.

    Works on floats, arrays or Series; the result has the type of the inputs.
    """
    time_s: _Number = ref_time_s * (target_distance_m / ref_distance_m) ** exponent
    return time_s


class RiegelBaseline(RaceModel):
    """Project the athlete's fastest training run of at least 5 km to the race distance.

    Falls back to the mean training pace when an athlete has no such run.
    """

    name = "Riegel"

    def __init__(self, exponent: float = RIEGEL_EXPONENT) -> None:
        self.exponent = exponent
        self._fallback = np.nan

    def fit(self, ds: RaceDataset) -> Self:
        self._fallback = float(ds.y.mean())
        return self

    def predict(self, ds: RaceDataset) -> FloatArray:
        static = ds.static
        distance_m = static["race_distance_m"].to_numpy(dtype=float)
        time_s = riegel_time(
            static["riegel_ref_time_s"].to_numpy(dtype=float),
            static["riegel_ref_distance_m"].to_numpy(dtype=float),
            distance_m,
            self.exponent,
        )
        pace: FloatArray = time_s / (distance_m / 1000)
        return np.where(np.isnan(pace), self._fallback, pace)


class TabularModel(RaceModel):
    """Base of scikit-learn models fitted on the tabular features (Template Method).

    Subclasses only provide :meth:`make_estimator`. Feature columns that are entirely missing in
    the training data are zeroed, because some estimators cannot handle them.
    """

    def __init__(self) -> None:
        self._estimator: Any = None  # scikit-learn has no type hints
        self._empty_columns = np.zeros(len(RaceDataset.tabular_columns), dtype=bool)

    def make_estimator(self) -> Any:
        """A fresh, unfitted scikit-learn regressor."""
        raise NotImplementedError

    def _features(self, ds: RaceDataset) -> FloatArray:
        features = ds.static[list(ds.tabular_columns)].to_numpy(dtype=float, copy=True)
        features[:, self._empty_columns] = 0.0
        return features

    def fit(self, ds: RaceDataset) -> Self:
        raw = ds.static[list(ds.tabular_columns)].to_numpy(dtype=float)
        self._empty_columns = np.isnan(raw).all(axis=0)
        self._estimator = self.make_estimator().fit(self._features(ds), ds.y)
        return self

    def predict(self, ds: RaceDataset) -> FloatArray:
        if self._estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return np.asarray(self._estimator.predict(self._features(ds)), dtype=float)


class RidgeBaseline(TabularModel):
    """Ridge regression on median-imputed, standardised features."""

    name: ClassVar[str] = "Ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        super().__init__()
        self.alpha = alpha

    def make_estimator(self) -> Any:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(self.alpha))


class GradientBoostingBaseline(TabularModel):
    """scikit-learn histogram gradient boosting with small, fixed hyper-parameters."""

    name: ClassVar[str] = "HistGB"

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        max_leaf_nodes: int = 8,
        min_samples_leaf: int = 5,
    ) -> None:
        super().__init__()
        self.params: dict[str, float | int] = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
        }

    def make_estimator(self) -> Any:
        return HistGradientBoostingRegressor(random_state=0, **self.params)
