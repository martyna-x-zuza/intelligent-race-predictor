"""Compare models under both evaluation protocols and write the results table.

Run as ``python -m race_predictor.evaluation.runner --data data``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from race_predictor.evaluation.metrics import evaluate
from race_predictor.evaluation.splits import Index, chronological_split, grouped_splits
from race_predictor.features import RaceDataset, build_race_dataset
from race_predictor.features.dataset import FloatArray
from race_predictor.models import (
    GradientBoostingBaseline,
    RaceModel,
    RidgeBaseline,
    RiegelBaseline,
)
from race_predictor.processing import Pipeline, cleaning_pipeline
from race_predictor.sources import athletes_to_frame, iter_athletes
from race_predictor.utils import timer

ModelFactory = Callable[[], RaceModel]

#: Sub-directories of the data directory that hold the exports.
EXPORT_DIRS = ("strava_activities", "garmin_activities")

DEFAULT_FACTORIES: Mapping[str, ModelFactory] = {
    "Riegel": RiegelBaseline,
    "Ridge": RidgeBaseline,
    "HistGB": GradientBoostingBaseline,
}


def score(ds: RaceDataset, pace_pred: FloatArray) -> dict[str, float]:
    """Time and pace metrics of predicted paces for the races of ``ds``."""
    distance_km = ds.meta["race_distance_m"].to_numpy(dtype=float) / 1000
    time = evaluate(ds.race_time_s, pace_pred * distance_km)
    pace = evaluate(ds.y, pace_pred)
    return {
        "MAE_time_min": time["mae"] / 60,
        "RMSE_time_min": time["rmse"] / 60,
        "MAPE_%": time["mape_pct"],
        "MedAPE_%": time["median_ape_pct"],
        "MAE_pace_s_km": pace["mae"],
    }


def protocol_predictions(
    ds: RaceDataset, make: ModelFactory, n_splits: int = 5
) -> Iterator[tuple[str, FloatArray, Index]]:
    """Yield ``(protocol, predicted paces, positions of the predicted races)``.

    A fresh model is fitted for every split. For grouped cross-validation the out-of-fold
    predictions of all folds are pooled.
    """
    train, test = chronological_split(ds)
    yield "chronological", make().fit(ds.subset(train)).predict(ds.subset(test)), test

    predictions, positions = [], []
    for train, test in grouped_splits(ds, n_splits):
        predictions.append(make().fit(ds.subset(train)).predict(ds.subset(test)))
        positions.append(test)
    yield "grouped-CV", np.concatenate(predictions), np.concatenate(positions)


def compare_models(
    ds: RaceDataset, factories: Mapping[str, ModelFactory] = DEFAULT_FACTORIES, n_splits: int = 5
) -> pd.DataFrame:
    """One row per model and protocol, with the wall-clock time of each evaluation."""
    rows = []
    for name, make in factories.items():
        predictions = protocol_predictions(ds, make, n_splits)
        while True:
            with timer(name) as elapsed:  # fitting happens lazily, inside next()
                result = next(predictions, None)
            if result is None:
                break
            protocol, predicted, positions = result
            metrics = score(ds.subset(positions), predicted)
            rows.append({"model": name, "protocol": protocol, **metrics, "fit_s": elapsed.seconds})
    return pd.DataFrame(rows)


def load_dataset(data_dir: Path, pipeline: Pipeline | None = None) -> RaceDataset:
    """Races of all exports under ``data_dir``, cleaned for training."""
    athletes = [athlete for name in EXPORT_DIRS for athlete in iter_athletes(data_dir / name)]
    cleaned = (pipeline or cleaning_pipeline("train")).run(athletes_to_frame(athletes))
    return build_race_dataset(cleaned)


CONCLUSIONS_HEADING = "## Conclusions"
CONCLUSIONS_TEMPLATE = f"\n{CONCLUSIONS_HEADING}\n\n(to be written after reading the table)\n"


def write_report(path: Path, generated: str) -> None:
    """Write a generated report to ``path``, keeping its hand-written conclusions.

    Numbers come from the script, but the ``## Conclusions`` section is written by hand, so a
    re-run must not overwrite it.
    """
    previous = path.read_text(encoding="utf-8") if path.exists() else ""
    if CONCLUSIONS_HEADING in previous and CONCLUSIONS_HEADING in generated:
        kept = previous[previous.index(CONCLUSIONS_HEADING) :]
        generated = generated[: generated.index(CONCLUSIONS_HEADING)] + kept
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated, encoding="utf-8")


def results_markdown(ds: RaceDataset, table: pd.DataFrame, seconds: float) -> str:
    """The results report: dataset size, protocols and the metrics table."""
    athletes = ds.meta["athlete_id"].nunique()
    return (
        "# Baseline results\n\n"
        f"{len(ds)} races from {athletes} athletes (Strava and Garmin exports). Metrics are on "
        "total race time (predicted pace x distance), except `MAE_pace_s_km`.\n\n"
        "- `chronological`: thesis protocol, the latest 20 % of each athlete's races are held "
        "out.\n"
        "- `grouped-CV`: 5-fold cross-validation holding out whole athletes.\n\n"
        f"{table.to_markdown(index=False, floatfmt='.2f')}\n\n"
        f"Generated by `python -m race_predictor.evaluation.runner` in {seconds:.1f} s.\n"
        f"{CONCLUSIONS_TEMPLATE}"
    )


def main(argv: list[str] | None = None) -> None:
    """Evaluate the baselines on ``--data`` and write the report to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("docs/RESULTS_BASELINES.md"))
    args = parser.parse_args(argv)

    with timer("baselines") as elapsed:
        ds = load_dataset(args.data)
        table = compare_models(ds)
    report = results_markdown(ds, table, elapsed.seconds)
    write_report(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
