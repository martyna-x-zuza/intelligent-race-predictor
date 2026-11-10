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

from race_predictor.features import FloatArray, RaceDataset
from race_predictor.models._base import RaceModel

#: Fatigue exponent of Riegel (1981).
RIEGEL_EXPONENT: Final = 1.06

_Number = TypeVar("_Number", float, npt.NDArray[np.float64], "pd.Series[float]")


def riegel_time(
    ref_time_s: _Number,
    ref_distance_m: _Number,
    target_distance_m: _Number,
    exponent: float = RIEGEL_EXPONENT,
) -> _Number:
    """Predict a race time with Riegel's formula ``T2 = T1 * (D2 / D1) ** exponent``.

    Works on floats, arrays or Series; all three arguments must be of the same kind.

    Args:
        ref_time_s: Time of the reference performance, in seconds.
        ref_distance_m: Distance of the reference performance, in metres.
        target_distance_m: Distance to predict, in metres.
        exponent: Fatigue exponent; 1.06 in Riegel's paper.

    Returns:
        Predicted time over ``target_distance_m``, in seconds, of the type of the inputs.
    """
    time_s: _Number = ref_time_s * (target_distance_m / ref_distance_m) ** exponent
    return time_s


class RiegelBaseline(RaceModel):
    """Project the athlete's fastest training run of at least 5 km to the race distance.

    Falls back to the mean training pace when an athlete has no such run.
    """

    name = "Riegel"

    def __init__(self, exponent: float = RIEGEL_EXPONENT) -> None:
        """Create the model.

        Args:
            exponent: Fatigue exponent of the formula.
        """
        self._exponent = exponent
        self._fallback = np.nan

    def fit(self, ds: RaceDataset) -> Self:
        """Remember the mean training pace as the fallback.

        Args:
            ds: Training races.

        Returns:
            The model itself.
        """
        self._fallback = float(ds.y.mean())
        return self

    def predict(self, ds: RaceDataset) -> FloatArray:
        """Project each race's reference run to the race distance.

        Args:
            ds: Races to predict.

        Returns:
            Predicted pace of every race in s/km; the fallback where there is no reference run.
        """
        static = ds.static
        distance_m = static["race_distance_m"].to_numpy(dtype=float)
        time_s = riegel_time(
            static["riegel_ref_time_s"].to_numpy(dtype=float),
            static["riegel_ref_distance_m"].to_numpy(dtype=float),
            distance_m,
            self._exponent,
        )
        pace: FloatArray = time_s / (distance_m / 1000)
        return np.where(np.isnan(pace), self._fallback, pace)


class TabularModel(RaceModel):
    """Base of scikit-learn models fitted on the tabular features (Template Method).

    Subclasses only provide :meth:`make_estimator`. Feature columns that are entirely missing in
    the training data are zeroed, because some estimators cannot handle them.
    """

    def __init__(self) -> None:
        """Create an unfitted model."""
        self._estimator: Any = None  # scikit-learn has no type hints
        self._empty_columns = np.zeros(len(RaceDataset.tabular_columns), dtype=bool)

    def make_estimator(self) -> Any:
        """Create the regressor to fit.

        Returns:
            A fresh, unfitted scikit-learn regressor.

        Raises:
            NotImplementedError: In this base class; subclasses override it.
        """
        raise NotImplementedError

    def _features(self, ds: RaceDataset) -> FloatArray:
        """Build the feature matrix, zeroing the columns that were empty in training.

        Args:
            ds: Races to build features for.

        Returns:
            Matrix of shape ``(len(ds), len(tabular_columns))``.
        """
        features = ds.static[list(ds.tabular_columns)].to_numpy(dtype=float, copy=True)
        features[:, self._empty_columns] = 0.0
        return features

    def fit(self, ds: RaceDataset) -> Self:
        """Fit a fresh estimator on the tabular features.

        Args:
            ds: Training races.

        Returns:
            The model itself.
        """
        raw = ds.static[list(ds.tabular_columns)].to_numpy(dtype=float)
        self._empty_columns = np.isnan(raw).all(axis=0)
        self._estimator = self.make_estimator().fit(self._features(ds), ds.y)
        return self

    def predict(self, ds: RaceDataset) -> FloatArray:
        """Predict race paces with the fitted estimator.

        Args:
            ds: Races to predict.

        Returns:
            Predicted pace of every race, in s/km.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self._estimator is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return np.asarray(self._estimator.predict(self._features(ds)), dtype=float)


class RidgeBaseline(TabularModel):
    """Ridge regression on median-imputed, standardised features."""

    name: ClassVar[str] = "Ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        """Create the model.

        Args:
            alpha: L2 penalty of the ridge regression.
        """
        super().__init__()
        self._alpha = alpha

    def make_estimator(self) -> Any:
        """Create the imputer, scaler and ridge regression chain.

        Returns:
            An unfitted scikit-learn pipeline.
        """
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(self._alpha))


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
        """Create the model.

        Args:
            learning_rate: Shrinkage of every boosting step.
            max_iter: Number of boosting iterations.
            max_leaf_nodes: Largest number of leaves per tree.
            min_samples_leaf: Fewest training races per leaf.
        """
        super().__init__()
        self._params: dict[str, float | int] = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
        }

    def make_estimator(self) -> Any:
        """Create the gradient boosting regressor.

        Returns:
            An unfitted ``HistGradientBoostingRegressor`` with a fixed random seed.
        """
        return HistGradientBoostingRegressor(random_state=0, **self._params)
