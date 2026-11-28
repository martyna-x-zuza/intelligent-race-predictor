"""Compare models under both evaluation protocols and record the results.

The command line entry point is ``race-predictor baselines`` (:mod:`race_predictor.cli`).
"""

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from race_predictor import __version__
from race_predictor.evaluation._metrics import evaluate
from race_predictor.evaluation._splits import chronological_split, grouped_splits
from race_predictor.features import FloatArray, Index, RaceDataset, build_race_dataset
from race_predictor.models import (
    GradientBoostingBaseline,
    GRUModel,
    LSTMModel,
    RaceModel,
    RidgeBaseline,
    RiegelBaseline,
)
from race_predictor.processing import Pipeline, cleaning_pipeline
from race_predictor.sources import athletes_to_frame, iter_athletes
from race_predictor.utils import timer

ModelFactory = Callable[[], RaceModel]

_EXPORT_DIRS = ("strava_activities", "garmin_activities")

DEFAULT_FACTORIES: Mapping[str, ModelFactory] = {
    "Riegel": RiegelBaseline,
    "Ridge": RidgeBaseline,
    "HistGB": GradientBoostingBaseline,
}
AVAILABLE_FACTORIES: Mapping[str, ModelFactory] = {
    **DEFAULT_FACTORIES,
    "GRU": GRUModel,
    "LSTM": LSTMModel,
}

_CONCLUSIONS_HEADING = "## Conclusions"
_CONCLUSIONS_TEMPLATE = f"\n{_CONCLUSIONS_HEADING}\n\n(to be written after reading the table)\n"


@dataclass(frozen=True)
class EvaluationParams:
    """Parameters of one evaluation run, saved next to its results.

    Attributes:
        data_dir: Directory with the Strava and Garmin exports.
        models: Names of the compared models.
        n_splits: Folds of the athlete-grouped cross-validation.
        test_fraction: Share of each athlete's races held out by the chronological split.
    """

    data_dir: Path
    models: tuple[str, ...]
    n_splits: int = 5
    test_fraction: float = 0.2


@dataclass(frozen=True)
class ModelResult:
    """Evaluation of one model under one protocol: one row of the results table.

    Attributes:
        model: Name of the model.
        protocol: Name of the evaluation protocol.
        mae_time_min: Mean absolute error of the race time, in minutes.
        rmse_time_min: Root mean squared error of the race time, in minutes.
        mape_pct: Mean absolute percentage error of the race time.
        median_ape_pct: Median absolute percentage error of the race time.
        mae_pace_s_km: Mean absolute error of the pace, in s/km.
        fit_s: Wall-clock time of fitting and predicting, in seconds.
    """

    model: str
    protocol: str
    mae_time_min: float
    rmse_time_min: float
    mape_pct: float
    median_ape_pct: float
    mae_pace_s_km: float
    fit_s: float


def _score(
    model: str, protocol: str, ds: RaceDataset, pace_pred: FloatArray, fit_s: float
) -> ModelResult:
    """Compute the time and pace metrics of predicted paces.

    Args:
        model: Name of the model.
        protocol: Name of the evaluation protocol.
        ds: The predicted races (with their true paces).
        pace_pred: Predicted pace of every race in ``ds``, in s/km.
        fit_s: Wall-clock time of fitting and predicting, in seconds.

    Returns:
        The metrics on the total race time and on the pace.
    """
    distance_km = ds.meta["race_distance_m"].to_numpy(dtype=float) / 1_000.0
    time = evaluate(ds.race_time_s, pace_pred * distance_km)
    pace = evaluate(ds.y, pace_pred)
    return ModelResult(
        model=model,
        protocol=protocol,
        mae_time_min=time.mae / 60,
        rmse_time_min=time.rmse / 60,
        mape_pct=time.mape_pct,
        median_ape_pct=time.median_ape_pct,
        mae_pace_s_km=pace.mae,
        fit_s=fit_s,
    )


def protocol_predictions(
    ds: RaceDataset, make: ModelFactory, n_splits: int = 5, test_fraction: float = 0.2
) -> Iterator[tuple[str, FloatArray, Index]]:
    """Lazily evaluate one model under both protocols.

    A fresh model is fitted for every split. For grouped cross-validation the out-of-fold
    predictions of all folds are pooled.

    Args:
        ds: All races.
        make: Creates a fresh, unfitted model.
        n_splits: Folds of the grouped cross-validation.
        test_fraction: Share of each athlete's races held out by the chronological split.

    Yields:
        ``(protocol name, predicted paces, positions of the predicted races in ds)``.
    """
    train, test = chronological_split(ds, test_fraction)
    yield "chronological", make().fit(ds.subset(train)).predict(ds.subset(test)), test

    predictions, positions = [], []
    for train, test in grouped_splits(ds, n_splits):
        predictions.append(make().fit(ds.subset(train)).predict(ds.subset(test)))
        positions.append(test)
    yield "grouped-CV", np.concatenate(predictions), np.concatenate(positions)


def compare_models(
    ds: RaceDataset,
    factories: Mapping[str, ModelFactory] = DEFAULT_FACTORIES,
    n_splits: int = 5,
    test_fraction: float = 0.2,
) -> pd.DataFrame:
    """Evaluate several models under both protocols.

    Args:
        ds: All races.
        factories: Model name to a function creating a fresh model.
        n_splits: Folds of the grouped cross-validation.
        test_fraction: Share of each athlete's races held out by the chronological split.

    Returns:
        One row per model and protocol, with the fields of :class:`ModelResult` as columns.
    """
    results = []
    for name, make in factories.items():
        predictions = protocol_predictions(ds, make, n_splits, test_fraction)
        while True:
            with timer(name) as elapsed:  # fitting happens lazily, inside next()
                result = next(predictions, None)
            if result is None:
                break
            protocol, predicted, positions = result
            results.append(_score(name, protocol, ds.subset(positions), predicted, elapsed.seconds))
    return pd.DataFrame([asdict(result) for result in results])


def load_dataset(data_dir: Path, pipeline: Pipeline | None = None) -> RaceDataset:
    """Load, clean and featurise all exports of a data directory.

    Args:
        data_dir: Directory with ``strava_activities/`` and ``garmin_activities/``.
        pipeline: Cleaning pipeline; ``cleaning_pipeline("train")`` by default.

    Returns:
        One example per race.
    """
    athletes = [athlete for name in _EXPORT_DIRS for athlete in iter_athletes(data_dir / name)]
    cleaned = (pipeline or cleaning_pipeline("train")).run(athletes_to_frame(athletes))
    return build_race_dataset(cleaned)


def write_report(path: Path, generated: str) -> None:
    """Write a generated report, keeping the hand-written conclusions of the previous one.

    Numbers come from the script, but the ``## Conclusions`` section is written by hand, so a
    re-run must not overwrite it.

    Args:
        path: The report file; created with its parent directories if missing.
        generated: The new report, ending with a ``## Conclusions`` placeholder.
    """
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    if _CONCLUSIONS_HEADING in previous and _CONCLUSIONS_HEADING in generated:
        kept = previous[previous.index(_CONCLUSIONS_HEADING) :]
        generated = generated[: generated.index(_CONCLUSIONS_HEADING)] + kept
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated, encoding="utf-8")


def _results_markdown(ds: RaceDataset, table: pd.DataFrame, params: EvaluationParams) -> str:
    """Render the results report.

    Args:
        ds: The evaluated races.
        table: Output of :func:`compare_models`.
        params: Parameters of the run.

    Returns:
        Markdown with the dataset size, the protocols, the metrics table and a placeholder for
        the conclusions.
    """
    athletes = ds.meta["athlete_id"].nunique()
    held_out = round(100 * params.test_fraction)
    return (
        "# Model comparison\n\n"
        f"{len(ds)} races from {athletes} athletes (Strava and Garmin exports). Metrics are on "
        "total race time (predicted pace x distance), except `mae_pace_s_km`.\n\n"
        f"- `chronological`: thesis protocol, the latest {held_out} % of each athlete's races "
        "are held out.\n"
        f"- `grouped-CV`: {params.n_splits}-fold cross-validation holding out whole "
        "athletes.\n\n"
        f"{table.to_markdown(index=False, floatfmt='.2f')}\n\n"
        "Generated by `race-predictor baselines`; the run's parameters (`params.json`) and raw "
        "results (`results.csv`) are saved to its artifacts directory.\n"
        f"{_CONCLUSIONS_TEMPLATE}"
    )


def _write_artifacts(
    directory: Path, params: EvaluationParams, ds: RaceDataset, table: pd.DataFrame, seconds: float
) -> None:
    """Save the parameters and raw results of a run.

    Args:
        directory: Artifact directory of the run; created if missing.
        params: Parameters of the run.
        ds: The evaluated races.
        table: Output of :func:`compare_models`.
        seconds: Wall-clock duration of the run.
    """
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        **asdict(params),
        "data_dir": str(params.data_dir),
        "n_races": len(ds),
        "n_athletes": int(ds.meta["athlete_id"].nunique()),
        "duration_s": round(seconds, 2),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
    }
    (directory / "params.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    table.to_csv(directory / "results.csv", index=False)


def run_baselines(
    params: EvaluationParams,
    report_path: Path,
    artifacts_dir: Path,
    factories: Mapping[str, ModelFactory] = AVAILABLE_FACTORIES,
) -> pd.DataFrame:
    """Evaluate the models and record the run.

    Writes the markdown report (keeping its conclusions) and saves ``params.json`` and
    ``results.csv`` to ``artifacts_dir``.

    Args:
        params: What to evaluate and how; ``params.models`` must name entries of ``factories``.
        report_path: Markdown report to write, for example ``docs/RESULTS_BASELINES.md``.
        artifacts_dir: Directory for the raw outputs of the run.
        factories: Available models by name.

    Returns:
        The results table.

    Raises:
        ValueError: If ``params.models`` names a model that ``factories`` does not have.
    """
    unknown = set(params.models) - set(factories)
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; available: {sorted(factories)}")
    chosen = {name: factories[name] for name in params.models}
    with timer("baselines") as elapsed:
        ds = load_dataset(params.data_dir)
        table = compare_models(ds, chosen, params.n_splits, params.test_fraction)
    write_report(report_path, _results_markdown(ds, table, params))
    _write_artifacts(artifacts_dir, params, ds, table, elapsed.seconds)
    return table
