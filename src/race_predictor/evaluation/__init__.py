"""Metrics, evaluation protocols and model comparison.

This is the public interface of the package; its modules are internal.
"""

from race_predictor.evaluation._metrics import evaluate
from race_predictor.evaluation._runner import (
    DEFAULT_FACTORIES,
    EvaluationParams,
    ModelFactory,
    compare_models,
    load_dataset,
    protocol_predictions,
    run_baselines,
    write_report,
)
from race_predictor.evaluation._splits import Index, chronological_split, grouped_splits

__all__ = [
    "DEFAULT_FACTORIES",
    "EvaluationParams",
    "Index",
    "ModelFactory",
    "chronological_split",
    "compare_models",
    "evaluate",
    "grouped_splits",
    "load_dataset",
    "protocol_predictions",
    "run_baselines",
    "write_report",
]
