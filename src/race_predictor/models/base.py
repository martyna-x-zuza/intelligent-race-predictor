"""Common interface of every race-time model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

from race_predictor.features import RaceDataset
from race_predictor.features.dataset import FloatArray


class RaceModel(ABC):
    """A model fitted on a :class:`RaceDataset` that predicts race pace in s/km.

    All models share this interface, so the evaluation treats a formula, a linear model and a
    neural network alike.
    """

    #: Short label used in result tables.
    name: ClassVar[str]

    @abstractmethod
    def fit(self, ds: RaceDataset) -> Self:
        """Fit on the races of ``ds`` and return ``self``."""

    @abstractmethod
    def predict(self, ds: RaceDataset) -> FloatArray:
        """Predict the race pace in s/km of every race in ``ds``."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
