import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from race_predictor.features import SEQ_FEATURES, RaceDataset
from race_predictor.models import GRUModel, LSTMModel, SequenceModel
from race_predictor.models._sequence import _pad_sequences
from tests.models.test_baselines import make_dataset

N_FEATURES = len(SEQ_FEATURES)


def make_sequence_dataset(n: int = 40, seed: int = 0) -> RaceDataset:
    """Races whose pace depends on the distance (a static feature) and on a run feature."""
    rng = np.random.default_rng(seed)
    base = make_dataset(n, seed)
    seqs = []
    for _ in range(n):
        seq = rng.normal(size=(int(rng.integers(3, 12)), N_FEATURES)).astype(np.float32)
        seq[0, -1] = np.nan
        seqs.append(seq)
    distance_km = base.meta["race_distance_m"].to_numpy() / 1_000.0
    pace = 300 + 3 * distance_km + 10 * np.array([seq[-1, 1] for seq in seqs])
    static = base.static.assign(race_distance_m=base.meta["race_distance_m"], swim_t_total=0.0)
    return RaceDataset(seqs, static, pace, base.meta)


def test_pad_sequences_pads_at_the_end_and_keeps_the_most_recent_runs() -> None:
    short = np.ones((2, N_FEATURES), dtype=np.float32)
    long = np.arange(5, dtype=np.float32)[:, None] * np.ones(N_FEATURES, dtype=np.float32)
    scaler = StandardScaler(with_mean=False, with_std=False).fit(long)

    padded = _pad_sequences([short, long], scaler, max_runs=3)

    assert padded.shape == (2, 3, N_FEATURES)
    assert (padded[0, :2] == 1).all() and (padded[0, 2] == 999).all()
    assert padded[1, :, 0].tolist() == [2, 3, 4]


def test_pad_sequences_replaces_missing_values_with_the_training_mean() -> None:
    seq = np.array([[1.0] * N_FEATURES, [3.0] * N_FEATURES], dtype=np.float32)
    seq[0, 0] = np.nan
    scaler = StandardScaler().fit(seq)

    padded = _pad_sequences([seq], scaler, max_runs=2)

    assert padded[0, 0, 0] == 0.0
    assert not np.isnan(padded).any()


@pytest.mark.parametrize("model_class", [GRUModel, LSTMModel])
class TestSequenceModels:
    @pytest.fixture(autouse=True)
    def _require_keras(self) -> None:
        pytest.importorskip("keras")

    def test_the_recurrent_cell_honours_the_padding_mask(
        self, model_class: type[SequenceModel]
    ) -> None:
        keras = pytest.importorskip("keras")
        assert getattr(keras.layers, model_class.cell)(1).supports_masking

    def test_learn_the_signal(self, model_class: type[SequenceModel]) -> None:
        train, test = make_sequence_dataset(seed=1), make_sequence_dataset(seed=2)
        model = model_class(epochs=60, patience=60).fit(train)
        error = np.abs(model.predict(test) - test.y).mean()
        assert error < np.abs(test.y - train.y.mean()).mean()

    def test_predictions_are_one_finite_pace_per_race(
        self, model_class: type[SequenceModel]
    ) -> None:
        ds = make_sequence_dataset()
        pace = model_class(epochs=2).fit(ds).predict(ds.subset(np.arange(5)))
        assert pace.shape == (5,)
        assert np.isfinite(pace).all()

    def test_a_seed_makes_training_reproducible(self, model_class: type[SequenceModel]) -> None:
        ds = make_sequence_dataset()
        first = model_class(epochs=3, seed=7).fit(ds).predict(ds)
        second = model_class(epochs=3, seed=7).fit(ds).predict(ds)
        np.testing.assert_allclose(first, second, rtol=1e-4)

    def test_long_histories_are_cut_to_the_most_recent_runs(
        self, model_class: type[SequenceModel]
    ) -> None:
        ds = make_sequence_dataset(n=10)
        long = RaceDataset(
            [np.zeros((300, N_FEATURES), dtype=np.float32)] * len(ds), ds.static, ds.y, ds.meta
        )
        assert model_class(epochs=1).fit(long).predict(long).shape == (10,)

    def test_predicting_before_fitting_fails(self, model_class: type[SequenceModel]) -> None:
        with pytest.raises(RuntimeError, match="fitted"):
            model_class().predict(make_sequence_dataset())
