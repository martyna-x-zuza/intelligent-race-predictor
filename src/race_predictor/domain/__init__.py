"""Domain model of the package.

This is the public interface of the package; its modules are internal.
"""

from race_predictor.domain._models import (
    STANDARD_DISTANCES_M,
    Activity,
    ActivityType,
    Athlete,
    Race,
    RaceCategory,
)

__all__ = [
    "STANDARD_DISTANCES_M",
    "Activity",
    "ActivityType",
    "Athlete",
    "Race",
    "RaceCategory",
]
