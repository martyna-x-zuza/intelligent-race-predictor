"""Regression metrics.

The median absolute percentage error is reported next to the mean: a few badly predicted
races pull the mean far above the typical error.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from race_predictor.exceptions import DataValidationError


def evaluate(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """Compute the regression metrics of a prediction.

    Args:
        y_true: True values; all must be positive.
        y_pred: Predicted values, same shape as ``y_true``.

    Returns:
        ``mae``, ``rmse``, ``nrmse_pct`` (RMSE over the range of ``y_true``), ``mape_pct`` and
        ``median_ape_pct``; percentages are in the ``*_pct`` keys.

    Raises:
        DataValidationError: If the arrays are empty, differ in shape or ``y_true`` has a
            non-positive value (percentage errors are then undefined).
    """
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape or true.size == 0:
        raise DataValidationError(f"cannot compare shapes {true.shape} and {pred.shape}")
    if (true <= 0).any():
        raise DataValidationError("true values must be positive")
    error = pred - true
    rmse = float(np.sqrt(np.mean(error**2)))
    span = float(true.max() - true.min()) or 1.0
    ape = np.abs(error) / true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "nrmse_pct": 100 * rmse / span,
        "mape_pct": float(100 * np.mean(ape)),
        "median_ape_pct": float(100 * np.median(ape)),
    }
