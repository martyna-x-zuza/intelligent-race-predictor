from pathlib import Path

import numpy as np
import pytest

from race_predictor.evaluation import chronological_split, evaluate, grouped_splits
from race_predictor.evaluation.runner import (
    compare_models,
    load_dataset,
    main,
    protocol_predictions,
    write_report,
)
from race_predictor.exceptions import DataValidationError
from race_predictor.models import RiegelBaseline
from race_predictor.utils import timer
from tests.models.test_baselines import make_dataset

DATA = Path(__file__).parents[2] / "data"


def test_evaluate() -> None:
    metrics = evaluate([100.0, 200.0], [110.0, 180.0])
    assert metrics["mae"] == pytest.approx(15.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert metrics["mape_pct"] == pytest.approx(10.0)
    assert metrics["median_ape_pct"] == pytest.approx(10.0)
    assert metrics["nrmse_pct"] == pytest.approx(100 * metrics["rmse"] / 100)


@pytest.mark.parametrize(
    ("y_true", "y_pred"), [([], []), ([1.0], [1.0, 2.0]), ([0.0, 1.0], [1.0, 1.0])]
)
def test_evaluate_rejects_invalid_input(y_true: list[float], y_pred: list[float]) -> None:
    with pytest.raises(DataValidationError):
        evaluate(y_true, y_pred)


def test_chronological_split_holds_out_each_athletes_latest_races() -> None:
    ds = make_dataset(n=40)  # 8 athletes, 5 races each
    train, test = chronological_split(ds)
    assert len(test) == 8 and len(set(train) | set(test)) == 40
    for athlete, races in ds.meta.groupby("athlete_id"):
        held_out = races.index.intersection(test.tolist())
        assert len(held_out) == 1, athlete
        assert races.loc[held_out, "race_date"].iloc[0] == races["race_date"].max()


def test_chronological_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        chronological_split(make_dataset(), test_fraction=1.0)


def test_grouped_splits_never_share_an_athlete() -> None:
    ds = make_dataset()
    folds = list(grouped_splits(ds, n_splits=4))
    assert len(folds) == 4
    athletes = ds.meta["athlete_id"]
    for train, test in folds:
        assert not set(athletes.iloc[train]) & set(athletes.iloc[test])
    assert sorted(np.concatenate([test for _, test in folds])) == list(range(len(ds)))


def test_grouped_splits_use_fewer_folds_with_few_athletes() -> None:
    ds = make_dataset(n=10)
    small = ds.subset(np.flatnonzero(ds.meta["athlete_id"].isin(["a0", "a1"])))
    assert len(list(grouped_splits(small, n_splits=5))) == 2


def test_protocol_predictions_cover_both_protocols() -> None:
    ds = make_dataset()
    results = {name: (pred, idx) for name, pred, idx in protocol_predictions(ds, RiegelBaseline)}
    assert set(results) == {"chronological", "grouped-CV"}
    pred, idx = results["grouped-CV"]
    assert len(pred) == len(idx) == len(ds)  # every race predicted exactly once


def test_compare_models_table() -> None:
    table = compare_models(make_dataset(), {"Riegel": RiegelBaseline})
    assert table["protocol"].tolist() == ["chronological", "grouped-CV"]
    assert {"MAE_time_min", "MAPE_%", "MedAPE_%", "fit_s"} <= set(table.columns)
    assert (table["fit_s"] >= 0).all()


def test_timer_measures_even_when_the_block_raises() -> None:
    with pytest.raises(ZeroDivisionError), timer("boom") as elapsed:
        _ = 1 / 0
    assert elapsed.label == "boom" and elapsed.seconds >= 0


@pytest.mark.skipif(not DATA.exists(), reason="sample exports not present")
def test_baselines_on_the_sample_exports(tmp_path: Path) -> None:
    assert len(load_dataset(DATA)) > 50
    out = tmp_path / "results.md"
    main(["--data", str(DATA), "--out", str(out)])
    assert "| Riegel" in out.read_text()


def test_write_report_keeps_hand_written_conclusions(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    write_report(path, "# Results\n\nold numbers\n\n## Conclusions\n\n(todo)\n")
    path.write_text(path.read_text().replace("(todo)", "Riegel is hard to beat."))
    write_report(path, "# Results\n\nnew numbers\n\n## Conclusions\n\n(todo)\n")
    text = path.read_text()
    assert "new numbers" in text and "old numbers" not in text
    assert "Riegel is hard to beat." in text and "(todo)" not in text
