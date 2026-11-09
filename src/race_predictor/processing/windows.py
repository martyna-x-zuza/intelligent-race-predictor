"""Training windows: the runs an athlete did before each race."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from race_predictor.domain import ActivityType


def training_windows(
    df: pd.DataFrame, days_window: int = 365, min_runs: int = 3
) -> Iterator[tuple[pd.Series, pd.DataFrame]]:
    """Lazily yield ``(race, preceding_runs)`` for every race in a cleaned activity table.

    ``preceding_runs`` holds the athlete's runs in ``[race date - days_window, race date)``:
    strictly before the race, so nothing about the race itself leaks into its features. Races
    with fewer than ``min_runs`` preceding runs are skipped.
    """
    runs = df[df["type"] == ActivityType.RUN].sort_values("date")
    for _, athlete_runs in runs.groupby("athlete_id"):
        for _, race in athlete_runs[athlete_runs["is_race"]].iterrows():
            start = race["date"] - pd.Timedelta(days=days_window)
            window = athlete_runs[
                (athlete_runs["date"] >= start) & (athlete_runs["date"] < race["date"])
            ]
            if len(window) >= min_runs:
                yield race, window
