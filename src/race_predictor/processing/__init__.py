"""Cleaning, race detection and training windows."""

from race_predictor.processing.cleaning import Purpose, cleaning_pipeline, shared_steps
from race_predictor.processing.pipeline import (
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
from race_predictor.processing.races import (
    DetectRaces,
    HeuristicRaceDetector,
    RaceDetector,
    StandardDistanceDetector,
)
from race_predictor.processing.windows import training_windows

__all__ = [
    "DetectRaces",
    "DropAthletesWithoutHR",
    "DropTriathlonRaces",
    "EstimateHRMax",
    "HeuristicRaceDetector",
    "ImputeMissing",
    "MarkTriathlonRuns",
    "Pipeline",
    "Purpose",
    "RaceDetector",
    "RemoveAnomalies",
    "RemoveImplausiblyFastRuns",
    "StandardDistanceDetector",
    "Step",
    "ValidateHeartRate",
    "ValidateMaxHeartRate",
    "cleaning_pipeline",
    "shared_steps",
    "training_windows",
]
