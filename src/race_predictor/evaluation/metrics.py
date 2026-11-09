"""Regression metrics.

The median absolute percentage error is reported next to the mean: a few badly predicted
races pull the mean far above the typical error.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from race_predictor.exceptions import DataValidationError


def evaluate(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """MAE, RMSE, NRMSE (RMSE over the range of ``y_true``), MAPE and median APE.

    Percentages are in the ``*_pct`` keys.

    Raises:
        DataValidationError: If the arrays are empty, differ in length or ``y_true`` has a
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
