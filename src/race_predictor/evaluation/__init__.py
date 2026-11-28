"""Metrics, evaluation protocols and model comparison.

This is the public interface of the package; its modules are internal.
"""

from race_predictor.evaluation._metrics import Metrics, evaluate
from race_predictor.evaluation._runner import (
    AVAILABLE_FACTORIES,
    DEFAULT_FACTORIES,
    EvaluationParams,
    ModelFactory,
    ModelResult,
    compare_models,
    load_dataset,
    protocol_predictions,
    run_baselines,
    write_report,
)
from race_predictor.evaluation._splits import chronological_split, grouped_splits

__all__ = [
    "AVAILABLE_FACTORIES",
    "DEFAULT_FACTORIES",
    "EvaluationParams",
    "Metrics",
    "ModelFactory",
    "ModelResult",
    "chronological_split",
    "compare_models",
    "evaluate",
    "grouped_splits",
    "load_dataset",
    "protocol_predictions",
    "run_baselines",
    "write_report",
]
