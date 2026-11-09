import math

import numpy as np
import pandas as pd
import pytest

from race_predictor.features import RaceDataset
from race_predictor.models import (
    GradientBoostingBaseline,
    RaceModel,
    RidgeBaseline,
    RiegelBaseline,
    riegel_time,
)


def make_dataset(n: int = 40, seed: int = 0) -> RaceDataset:
    """Synthetic races whose pace depends on weekly volume, with a Riegel reference run."""
    rng = np.random.default_rng(seed)
    weekly_km = rng.uniform(10, 80, n)
    distance_m = rng.choice([5000.0, 10_000.0, 21_097.5], n)
    pace = 360 - 1.5 * weekly_km + rng.normal(0, 3, n)
    static = pd.DataFrame(
        {column: rng.normal(size=n) for column in RaceDataset.tabular_columns}
    ).assign(
        weekly_km=weekly_km,
        race_distance_m=distance_m,
        swim_t_total=np.nan,  # a column missing for every race
        riegel_ref_distance_m=5000.0,
        riegel_ref_time_s=pace * 5 * 0.97,
    )
    meta = pd.DataFrame(
        {
            "athlete_id": [f"a{i % 8}" for i in range(n)],
            "race_date": pd.date_range("2023-01-01", periods=n, freq="7D"),
            "race_distance_m": distance_m,
        }
    )
    seqs = [np.zeros((3, 11), dtype=np.float32) for _ in range(n)]
    return RaceDataset(seqs, static, pace, meta)


def test_riegel_time_of_the_same_distance_is_unchanged() -> None:
    assert riegel_time(1200.0, 5000.0, 5000.0) == pytest.approx(1200.0)


def test_riegel_time_marathon_from_half() -> None:
    # 1:30 half marathon -> about 3:08 marathon with the 1.06 exponent.
    assert riegel_time(5400.0, 21_097.5, 42_195.0) / 3600 == pytest.approx(3.13, abs=0.01)


def test_riegel_time_works_on_arrays() -> None:
    times = riegel_time(np.array([1200.0, 1500.0]), np.full(2, 5000.0), np.full(2, 10_000.0))
    assert times.shape == (2,)
    assert (times > 2 * np.array([1200.0, 1500.0])).all()  # longer races are slower per km


def test_riegel_baseline_projects_the_reference_run() -> None:
    ds = make_dataset()
    pace = RiegelBaseline().fit(ds).predict(ds)
    expected = riegel_time(
        ds.static["riegel_ref_time_s"].to_numpy(dtype=float),
        np.full(len(ds), 5000.0),
        ds.static["race_distance_m"].to_numpy(dtype=float),
    ) / (ds.static["race_distance_m"].to_numpy(dtype=float) / 1000)
    np.testing.assert_allclose(pace, expected)


def test_riegel_baseline_falls_back_to_the_mean_pace() -> None:
    ds = make_dataset()
    static = ds.static.assign(riegel_ref_time_s=math.nan)
    no_reference = RaceDataset(ds.seqs, static, ds.y, ds.meta)
    pace = RiegelBaseline().fit(ds).predict(no_reference)
    np.testing.assert_allclose(pace, ds.y.mean())


@pytest.mark.parametrize("model", [RidgeBaseline(), GradientBoostingBaseline()])
def test_tabular_models_learn_the_signal(model: RaceModel) -> None:
    train, test = make_dataset(seed=1), make_dataset(seed=2)
    error = np.abs(model.fit(train).predict(test) - test.y).mean()
    assert error < np.abs(test.y - train.y.mean()).mean() / 2  # beats predicting the mean


def test_tabular_model_must_be_fitted() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        RidgeBaseline().predict(make_dataset())


def test_fit_returns_the_model() -> None:
    model = RidgeBaseline(alpha=1.0)
    assert model.fit(make_dataset()) is model
    assert repr(model) == "RidgeBaseline()"
