"""Command line entry point of the project.

Installed as the ``race-predictor`` command (see ``[project.scripts]`` in ``pyproject.toml``)
and runnable as ``python -m race_predictor``. Every user-facing action is a subcommand, so new
actions (prediction, the app) are added here instead of as separate scripts.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from race_predictor import __version__
from race_predictor.evaluation import DEFAULT_FACTORIES, EvaluationParams, run_baselines


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
        choices=sorted(DEFAULT_FACTORIES),
        default=list(DEFAULT_FACTORIES),
        help="models to compare",
    )
    baselines.add_argument("--n-splits", type=int, default=5, help="folds of the grouped CV")
    baselines.add_argument(
        "--test-fraction", type=float, default=0.2, help="share held out by the chronological split"
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
    print(table.to_string(index=False, float_format="{:.2f}".format))  # noqa: T201
    print(f"\nReport: {args.report}\nArtifacts: {args.artifacts}")  # noqa: T201


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line and run the chosen subcommand.

    Args:
        argv: Arguments without the program name; ``sys.argv[1:]`` when ``None``.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.verbose else logging.ERROR)
    if args.command == "baselines":
        _baselines(args)
