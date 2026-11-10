"""Race-time models.

This is the public interface of the package; its modules are internal.
"""

from race_predictor.models._base import RaceModel
from race_predictor.models._baselines import (
    RIEGEL_EXPONENT,
    GradientBoostingBaseline,
    RidgeBaseline,
    RiegelBaseline,
    TabularModel,
    riegel_time,
)

__all__ = [
    "RIEGEL_EXPONENT",
    "GradientBoostingBaseline",
    "RaceModel",
    "RidgeBaseline",
    "RiegelBaseline",
    "TabularModel",
    "riegel_time",
]
