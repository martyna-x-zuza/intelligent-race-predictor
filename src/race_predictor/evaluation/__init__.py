"""Metrics and evaluation protocols.

Model comparison lives in :mod:`race_predictor.evaluation.runner`, which is also the
command-line entry point (``python -m race_predictor.evaluation.runner``).
"""

from race_predictor.evaluation.metrics import evaluate
from race_predictor.evaluation.splits import chronological_split, grouped_splits

__all__ = ["chronological_split", "evaluate", "grouped_splits"]
