"""One definition of data cleaning, shared by training and prediction.

If training and the app cleaned data with separate code, a fix in one would silently miss the
other and the model would see differently prepared data at prediction time. Both therefore
come from :func:`cleaning_pipeline`: the shared steps are identical and only what has to differ
does.

* ``"train"``: drop rows and athletes without heart rate (race detection needs it), then label
  races and keep runs after a ride out of the races to predict.
* ``"predict"``: keep rows without heart rate (users may not wear a strap), label races with the
  same detector, and impute gaps only *after* detection so imputed heart rates never make a run
  look like a race.
"""

from __future__ import annotations

from typing import Literal

from race_predictor.processing._pipeline import (
    DropAthletesWithoutHR,
    DropTriathlonRaces,
    EstimateHRMax,
    ImputeMissing,
    MarkTriathlonRuns,
    Pipeline,
    RemoveAnomalies,
    RemoveImplausiblyFastRuns,
    Step,
    ValidateHeartRate,
    ValidateMaxHeartRate,
)
from race_predictor.processing._races import DetectRaces, RaceDetector

Purpose = Literal["train", "predict"]


def _shared_steps() -> list[Step]:
    """Build the steps applied to every dataset, for training and prediction alike.

    Returns:
        New step instances, in order.
    """
    return [RemoveAnomalies(), RemoveImplausiblyFastRuns(), ValidateMaxHeartRate()]


def cleaning_pipeline(
    purpose: Purpose = "train",
    detector: RaceDetector | None = None,
    *,
    include_triathlon_races: bool = False,
) -> Pipeline:
    """Build the cleaning chain for training or for prediction.

    Args:
        purpose: ``"train"`` or ``"predict"`` (see the module docstring).
        detector: Race detection strategy; :class:`HeuristicRaceDetector` by default.
        include_triathlon_races: Keep runs after a ride as races (thesis behaviour).

    Returns:
        The pipeline, ready to :meth:`~Pipeline.run`.

    Raises:
        ValueError: If ``purpose`` is not ``"train"`` or ``"predict"``.
    """
    if purpose not in ("train", "predict"):
        raise ValueError(f"purpose must be 'train' or 'predict', got {purpose!r}")
    labelling: list[Step] = [EstimateHRMax(), MarkTriathlonRuns(), DetectRaces(detector)]
    if not include_triathlon_races:
        labelling.append(DropTriathlonRaces())
    if purpose == "predict":
        return Pipeline(
            [*_shared_steps(), ValidateHeartRate(keep_missing=True), *labelling, ImputeMissing()]
        )
    return Pipeline([*_shared_steps(), ValidateHeartRate(), DropAthletesWithoutHR(), *labelling])
