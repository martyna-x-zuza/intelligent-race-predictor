"""Evaluation protocols.

* :func:`chronological_split`, the thesis protocol: each athlete's most recent races are the
  test set, so every test athlete was also seen in training.
* :func:`grouped_splits`, stricter: whole athletes are held out (``GroupKFold``), which measures
  how well a model generalises to a runner it has never seen.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold

from race_predictor.features import RaceDataset

Index = npt.NDArray[np.intp]


def chronological_split(ds: RaceDataset, test_fraction: float = 0.2) -> tuple[Index, Index]:
    """``(train, test)`` positions: the latest ``test_fraction`` of each athlete's races.

    Every athlete contributes at least one test race.
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    test = np.zeros(len(ds), dtype=bool)
    for _, races in ds.meta.groupby("athlete_id"):
        latest_first = races.sort_values("race_date", ascending=False).index.to_numpy()
        n_test = max(1, int(len(latest_first) * test_fraction))
        test[latest_first[:n_test]] = True
    return np.flatnonzero(~test), np.flatnonzero(test)


def grouped_splits(ds: RaceDataset, n_splits: int = 5) -> Iterator[tuple[Index, Index]]:
    """Lazily yield ``(train, test)`` positions of folds that keep each athlete in one fold.

    Uses fewer folds when there are fewer than ``n_splits`` athletes.
    """
    groups = ds.meta["athlete_id"].to_numpy()
    folds = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    for train, test in folds.split(np.zeros(len(ds)), groups=groups):
        yield np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp)
