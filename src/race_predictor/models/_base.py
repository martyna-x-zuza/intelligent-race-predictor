"""Common interface of every race-time model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Self

from race_predictor.features import FloatArray, RaceDataset


class RaceModel(ABC):
    """A model fitted on a :class:`RaceDataset` that predicts race pace in s/km.

    All models share this interface, so the evaluation treats a formula, a linear model and a
    neural network alike.
    """

    #: Short label used in result tables.
    name: ClassVar[str]

    @abstractmethod
    def fit(self, ds: RaceDataset) -> Self:
        """Fit the model.

        Args:
            ds: Training races.

        Returns:
            The fitted model itself, so calls can be chained.
        """

    @abstractmethod
    def predict(self, ds: RaceDataset) -> FloatArray:
        """Predict race paces.

        Args:
            ds: Races to predict; their targets are not used.

        Returns:
            Predicted pace of every race, in s/km.
        """

    def __repr__(self) -> str:
        """Show the model by its class name.

        Returns:
            For example ``"RidgeBaseline()"``.
        """
        return f"{type(self).__name__}()"
