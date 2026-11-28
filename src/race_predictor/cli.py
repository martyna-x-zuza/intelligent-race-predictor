"""Command line entry point of the project.

Installed as the ``race-predictor`` command (see ``[project.scripts]`` in ``pyproject.toml``)
and runnable as ``python -m race_predictor``. Every user-facing action is a subcommand, so new
actions are added here instead of as separate scripts.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from race_predictor import __version__
from race_predictor.domain import STANDARD_DISTANCES_M
from race_predictor.evaluation import (
    AVAILABLE_FACTORIES,
    DEFAULT_FACTORIES,
    EvaluationParams,
    run_baselines,
)
from race_predictor.exceptions import RacePredictorError
from race_predictor.predictor import RacePredictor, format_time, load_activities


def _build_parser() -> argparse.ArgumentParser:
    """Define the command line interface.

    Returns:
        The parser, with one subparser per subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="race-predictor", description="Predict running race times from training data."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show warnings")
    commands = parser.add_subparsers(dest="command", required=True)

    baselines = commands.add_parser(
        "baselines", help="evaluate the baseline models and write the results report"
    )
    baselines.add_argument("--data", type=Path, default=Path("data"), help="exports directory")
    baselines.add_argument(
        "--report", type=Path, default=Path("docs/RESULTS_BASELINES.md"), help="report to write"
    )
    baselines.add_argument(
        "--artifacts",
        type=Path,
        default=Path(".artifacts/baselines"),
        help="directory for the run's parameters and raw results",
    )
    baselines.add_argument(
        "--models",
        nargs="+",
        choices=sorted(AVAILABLE_FACTORIES),
        default=list(DEFAULT_FACTORIES),
        help="models to compare (GRU and LSTM need the 'deep' extra)",
    )
    baselines.add_argument("--n-splits", type=int, default=5, help="folds of the grouped CV")
    baselines.add_argument(
        "--test-fraction", type=float, default=0.2, help="share held out by the chronological split"
    )

    predict = commands.add_parser(
        "predict", help="predict race times from a Strava or Garmin export"
    )
    predict.add_argument("export", type=Path, help="Strava activities.csv or Garmin Connect CSV")
    predict.add_argument(
        "--data", type=Path, default=Path("data"), help="training exports directory"
    )
    predict.add_argument(
        "--model",
        choices=sorted(AVAILABLE_FACTORIES),
        default="Riegel",
        help="model to predict with",
    )
    predict.add_argument(
        "--distances-km",
        type=float,
        nargs="+",
        default=[metres / 1_000.0 for metres in STANDARD_DISTANCES_M.values()],
        help="race distances in km (default: 5 km to marathon)",
    )
    predict.add_argument(
        "--elevation-m", type=float, default=0.0, help="elevation gain of the races"
    )
    return parser


def _baselines(args: argparse.Namespace) -> None:
    """Run the ``baselines`` subcommand.

    Args:
        args: Parsed arguments of the subcommand.
    """
    params = EvaluationParams(
        data_dir=args.data,
        models=tuple(args.models),
        n_splits=args.n_splits,
        test_fraction=args.test_fraction,
    )
    table = run_baselines(params, args.report, args.artifacts)
    print(table.to_string(index=False, float_format="{:.2f}".format))
    print(f"\nReport: {args.report}\nArtifacts: {args.artifacts}")


def _predict(args: argparse.Namespace) -> None:
    """Run the ``predict`` subcommand.

    Args:
        args: Parsed arguments of the subcommand.
    """
    activities = load_activities(args.export)
    predictor = RacePredictor.from_data_dir(args.data, AVAILABLE_FACTORIES[args.model])
    distances_m = [km * 1_000.0 for km in args.distances_km]
    predictions = predictor.predict(activities, distances_m, elevation_gain_m=args.elevation_m)
    print(f"{'distance':>10}  {'predicted':>9}  {'range (10-90 %)':>19}")
    for prediction in predictions:
        time_range = f"{format_time(prediction.low_s)} - {format_time(prediction.high_s)}"
        print(
            f"{prediction.distance_m / 1_000.0:>7g} km  {format_time(prediction.time_s):>9}  "
            f"{time_range:>19}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and run the chosen subcommand.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when ``None``.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)
    try:
        if args.command == "baselines":
            _baselines(args)
        else:
            _predict(args)
    except RacePredictorError as error:
        raise SystemExit(f"error: {error}") from error
