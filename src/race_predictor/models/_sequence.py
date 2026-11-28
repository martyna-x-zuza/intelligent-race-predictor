"""Recurrent models: a GRU and an LSTM over the runs before a race, plus the static features.

Both share one hybrid layout (a recurrent encoder for the run sequence, a dense branch for the
static features and a regression head), so a difference in the results comes from the recurrent
cell alone. Keras is imported only when a model is fitted, so the rest of the package works
without it (``pip install -e ".[deep]"``).
"""

from collections.abc import Sequence
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import StandardScaler

from race_predictor.features import (
    SEQ_FEATURES,
    STATIC_BINARY,
    STATIC_NUMERIC,
    FloatArray,
    RaceDataset,
)
from race_predictor.models._base import RaceModel

Float32Array = npt.NDArray[np.float32]

_PAD_VALUE = 999.0
_MAX_RUNS = 256
_VALIDATION_SHARE = 0.2


def _pad_sequences(
    seqs: Sequence[Float32Array], scaler: StandardScaler, max_runs: int
) -> Float32Array:
    """Standardise the run sequences and pad them to one length.

    Missing values become 0 after scaling, which is the training mean. Sequences longer than
    ``max_runs`` keep their most recent runs; shorter ones are padded at the end.

    Args:
        seqs: One array of shape ``(n_runs, len(SEQ_FEATURES))`` per race, oldest run first.
        scaler: Fitted scaler of the run features.
        max_runs: Length every sequence is padded or cut to.

    Returns:
        Array of shape ``(len(seqs), max_runs, len(SEQ_FEATURES))``.
    """
    padded = np.full((len(seqs), max_runs, len(SEQ_FEATURES)), _PAD_VALUE, dtype=np.float32)
    for row, seq in zip(padded, seqs, strict=True):
        scaled = np.nan_to_num(scaler.transform(seq[-max_runs:]))
        row[: len(scaled)] = scaled
    return padded


def _dense_block(inputs: Any, units: int, dropout: float) -> Any:
    """Apply a dense layer with batch normalisation and dropout.

    Args:
        inputs: Keras tensor to transform.
        units: Width of the dense layer.
        dropout: Dropout rate.

    Returns:
        The transformed Keras tensor.
    """
    from keras import layers

    hidden = layers.Dense(units, activation="relu", kernel_regularizer="l2")(inputs)
    return layers.Dropout(dropout)(layers.BatchNormalization()(hidden))


class SequenceModel(RaceModel):
    """Base of the recurrent models (Template Method): subclasses only choose the cell.

    The network standardises its inputs and the target, and is trained with early stopping on a
    random share of the training races.
    """

    cell: ClassVar[str]

    def __init__(
        self,
        units: int = 16,
        dense_units: int = 32,
        dropout: float = 0.3,
        learning_rate: float = 8e-4,
        epochs: int = 300,
        patience: int = 30,
        batch_size: int = 16,
        seed: int = 0,
    ) -> None:
        """Create an unfitted model.

        Args:
            units: Width of each recurrent layer.
            dense_units: Width of the dense layers.
            dropout: Dropout rate.
            learning_rate: Learning rate of the Adam optimiser.
            epochs: Most epochs to train.
            patience: Epochs without validation improvement before training stops.
            batch_size: Races per training batch.
            seed: Seed of the random number generators.
        """
        self._units = units
        self._dense_units = dense_units
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._patience = patience
        self._batch_size = batch_size
        self._seed = seed
        self._network: Any = None
        self._seq_scaler = StandardScaler()
        self._static_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        self._max_runs = 0

    def _inputs(self, ds: RaceDataset) -> list[Float32Array]:
        """Build the two network inputs.

        Args:
            ds: Races to build inputs for.

        Returns:
            The padded standardised run sequences and the standardised static features.
        """
        numeric = ds.static[list(STATIC_NUMERIC)].to_numpy(dtype=float)
        binary = ds.static[list(STATIC_BINARY)].to_numpy(dtype=float)
        static = np.column_stack([np.nan_to_num(self._static_scaler.transform(numeric)), binary])
        return [
            _pad_sequences(ds.seqs, self._seq_scaler, self._max_runs),
            static.astype(np.float32),
        ]

    def _build_network(self, n_static: int) -> Any:
        """Build and compile the hybrid network.

        Args:
            n_static: Number of static input features.

        Returns:
            The compiled Keras model.
        """
        import keras
        from keras import layers

        recurrent = getattr(layers, self.cell)
        seq_in = layers.Input(shape=(self._max_runs, len(SEQ_FEATURES)))
        seq = layers.Masking(mask_value=_PAD_VALUE)(seq_in)
        seq = layers.Bidirectional(
            recurrent(self._units, return_sequences=True, kernel_regularizer="l2")
        )(seq)
        seq = layers.BatchNormalization()(layers.Dropout(self._dropout)(seq))
        seq = layers.Bidirectional(recurrent(self._units, kernel_regularizer="l2"))(seq)
        seq = _dense_block(seq, self._dense_units, self._dropout)

        static_in = layers.Input(shape=(n_static,))
        static = _dense_block(static_in, self._dense_units, self._dropout)

        merged = _dense_block(layers.concatenate([seq, static]), self._dense_units, self._dropout)
        network = keras.Model([seq_in, static_in], layers.Dense(1)(merged))
        network.compile(optimizer=keras.optimizers.Adam(self._learning_rate), loss="mae")
        return network

    def fit(self, ds: RaceDataset) -> Self:
        """Fit the scalers and train the network with early stopping.

        Args:
            ds: Training races.

        Returns:
            The model itself.
        """
        import keras

        keras.utils.set_random_seed(self._seed)
        self._seq_scaler = StandardScaler().fit(np.vstack(ds.seqs))
        self._static_scaler = StandardScaler().fit(ds.static[list(STATIC_NUMERIC)].to_numpy(float))
        self._y_scaler = StandardScaler().fit(ds.y.reshape(-1, 1))
        self._max_runs = min(max(len(seq) for seq in ds.seqs), _MAX_RUNS)

        inputs = self._inputs(ds)
        target = self._y_scaler.transform(ds.y.reshape(-1, 1)).ravel()
        # Keras validates on the last rows and the races are ordered by athlete: shuffle first.
        order = np.random.default_rng(self._seed).permutation(len(ds))
        self._network = self._build_network(inputs[1].shape[1])
        self._network.fit(
            [x[order] for x in inputs],
            target[order],
            validation_split=_VALIDATION_SHARE,
            epochs=self._epochs,
            batch_size=self._batch_size,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=self._patience, restore_best_weights=True)
            ],
        )
        return self

    def predict(self, ds: RaceDataset) -> FloatArray:
        """Predict race paces with the trained network.

        Args:
            ds: Races to predict.

        Returns:
            Predicted pace of every race, in s/km.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self._network is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        scaled = self._network.predict(self._inputs(ds), verbose=0)
        return np.asarray(self._y_scaler.inverse_transform(scaled), dtype=float).ravel()


class GRUModel(SequenceModel):
    """Hybrid network with gated recurrent units, as in the thesis."""

    name = "GRU"
    cell = "GRU"


class LSTMModel(SequenceModel):
    """The same network with long short-term memory cells."""

    name = "LSTM"
    cell = "LSTM"
