"""Cleaning, race detection and training windows.

This is the public interface of the package; its modules are internal.
"""

from race_predictor.processing._cleaning import Purpose, cleaning_pipeline
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
from race_predictor.processing._races import (
    DetectRaces,
    HeuristicRaceDetector,
    RaceDetector,
    StandardDistanceDetector,
)
from race_predictor.processing._windows import training_windows

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
    "training_windows",
]
