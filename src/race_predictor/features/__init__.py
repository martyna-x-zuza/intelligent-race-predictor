"""Feature construction: one example per race.

This is the public interface of the package; its modules are internal.
"""

from race_predictor.features._dataset import (
    AGGREGATES,
    RIEGEL_COLUMNS,
    SEQ_FEATURES,
    STATIC_BINARY,
    STATIC_NUMERIC,
    FloatArray,
    RaceDataset,
    build_race_dataset,
)

__all__ = [
    "AGGREGATES",
    "RIEGEL_COLUMNS",
    "SEQ_FEATURES",
    "STATIC_BINARY",
    "STATIC_NUMERIC",
    "FloatArray",
    "RaceDataset",
    "build_race_dataset",
]
