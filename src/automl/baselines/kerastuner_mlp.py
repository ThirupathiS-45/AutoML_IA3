"""KerasTuner MLP with a custom search space and a custom block (course Module 3).

Two things here go beyond the AutoKeras baseline:

1. **A hand-written search space.** Depth, width, dropout, learning rate and the
   choice of optimiser are declared explicitly, so we control exactly what is
   searched instead of accepting AutoKeras's defaults.

2. **A custom block.** :class:`LagSelection` is a Keras layer that multiplies the
   input by a fixed binary mask. The tuner chooses that mask -- one boolean per lag
   feature -- which turns *feature selection into a hyperparameter*. Asking "is the
   8-week lag worth keeping?" becomes part of the same search as "how wide should
   layer two be?". At least one feature is always kept, so the mask can never zero
   out the whole input.

Research-only: needs the ``research`` dependency group.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl.features import TARGET_COLUMN, split_xy, to_price_level
from automl.metrics import all_metrics
from automl.split import Window, slice_window

LOGGER = logging.getLogger(__name__)


def _numeric_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot encode categories and return a dense float matrix.

    Args:
        frame: A feature matrix.

    Returns:
        A tuple of (matrix, column names).
    """
    encoded = pd.get_dummies(
        frame,
        columns=[
            column
            for column in frame.columns
            if str(frame[column].dtype) in {"category", "object", "string"}
        ],
    )
    encoded = encoded.astype(float)
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    return encoded.to_numpy(dtype=float), list(encoded.columns)


def build_lag_selection_layer(mask: np.ndarray):
    """Create the custom block: a layer that keeps only the selected features.

    Args:
        mask: A 1-D binary array, one entry per input column.

    Returns:
        A Keras layer applying that mask elementwise.
    """
    import keras

    class LagSelection(keras.layers.Layer):
        """Zero out the input columns the tuner chose to discard."""

        def __init__(self, selection: np.ndarray, **kwargs: Any) -> None:
            """Store the mask as a non-trainable constant.

            Args:
                selection: Binary mask over the input columns.
                **kwargs: Passed through to the Keras base layer.
            """
            super().__init__(**kwargs)
            self.selection = np.asarray(selection, dtype="float32")

        def call(self, inputs):
            """Apply the mask.

            Args:
                inputs: A batch of feature vectors.

            Returns:
                The masked batch.
            """
            return inputs * keras.ops.convert_to_tensor(self.selection)

        def compute_output_shape(self, input_shape):
            """Masking does not change the shape.

            Args:
                input_shape: Shape of the input.

            Returns:
                The same shape.
            """
            return input_shape

    return LagSelection(mask)


def make_hypermodel(columns: list[str], selectable_prefixes: tuple[str, ...] = ("lag_",)):
    """Build the KerasTuner ``HyperModel`` function for our search space.

    Args:
        columns: Input column names, in matrix order.
        selectable_prefixes: Column name prefixes the custom block may switch off.

    Returns:
        A ``build(hp)`` callable that KerasTuner can search over.
    """
    import keras

    selectable = [
        index for index, name in enumerate(columns) if name.startswith(selectable_prefixes)
    ]

    def build(hp):
        """Construct one candidate model from a hyperparameter proposal.

        Args:
            hp: KerasTuner hyperparameter container.

        Returns:
            A compiled Keras model.
        """
        mask = np.ones(len(columns), dtype="float32")
        chosen = 0
        for index in selectable:
            keep = hp.Boolean(f"use_{columns[index]}", default=True)
            mask[index] = 1.0 if keep else 0.0
            chosen += int(keep)
        if selectable and chosen == 0:
            mask[selectable[0]] = 1.0  # never discard every lag

        inputs = keras.Input(shape=(len(columns),))
        hidden = build_lag_selection_layer(mask)(inputs)
        hidden = keras.layers.Normalization()(hidden)

        for layer_index in range(hp.Int("n_layers", 1, 3)):
            hidden = keras.layers.Dense(
                units=hp.Int(f"units_{layer_index}", 16, 128, step=16),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )(hidden)
            hidden = keras.layers.Dropout(hp.Float("dropout", 0.0, 0.4, step=0.1))(hidden)

        outputs = keras.layers.Dense(1)(hidden)
        model = keras.Model(inputs, outputs)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        optimizer = (
            keras.optimizers.Adam(learning_rate)
            if hp.Choice("optimizer", ["adam", "rmsprop"]) == "adam"
            else keras.optimizers.RMSprop(learning_rate)
        )
        model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
        return model

    return build


def run_kerastuner_mlp(
    features: pd.DataFrame,
    window: Window,
    seed: int = 0,
    max_trials: int = 8,
    epochs: int = 40,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Search the custom MLP space on one window and score it on the test block.

    Args:
        features: The full feature frame.
        window: The window to evaluate.
        seed: Random seed.
        max_trials: Number of architectures to try.
        epochs: Training epochs per trial.
        directory: Where KerasTuner writes its state. Temporary when omitted.

    Returns:
        A record with the metrics, the winning hyperparameters and the elapsed time.
    """
    import keras_tuner as kt
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)

    train = slice_window(features, window, "train")
    valid = slice_window(features, window, "valid")
    test = slice_window(features, window, "test")

    x_train_df, y_train = split_xy(train)
    x_valid_df, y_valid = split_xy(valid)
    x_test_df, _ = split_xy(test)

    combined = pd.concat([x_train_df, x_valid_df, x_test_df], ignore_index=True)
    matrix, columns = _numeric_matrix(combined)
    n_train, n_valid = len(x_train_df), len(x_valid_df)
    x_train = matrix[:n_train]
    x_valid = matrix[n_train : n_train + n_valid]
    x_test = matrix[n_train + n_valid :]

    workdir = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="kerastuner_"))
    started = pd.Timestamp.utcnow()
    try:
        tuner = kt.RandomSearch(
            make_hypermodel(columns),
            objective="val_mae",
            max_trials=max_trials,
            seed=seed,
            overwrite=True,
            directory=str(workdir),
            project_name="mandi_mlp",
        )
        tuner.search(
            x_train,
            np.asarray(y_train, dtype=float),
            validation_data=(x_valid, np.asarray(y_valid, dtype=float)),
            epochs=epochs,
            verbose=0,
        )
        best_hp = tuner.get_best_hyperparameters(1)[0]
        # KerasTuner wraps a plain build function in a DefaultHyperModel, which is
        # not itself callable; the builder lives on `.build`.
        model = tuner.hypermodel.build(best_hp)
        model.fit(
            np.vstack([x_train, x_valid]),
            np.concatenate([np.asarray(y_train, dtype=float), np.asarray(y_valid, dtype=float)]),
            epochs=epochs,
            verbose=0,
        )
        raw = np.asarray(model.predict(x_test, verbose=0), dtype=float).ravel()
        predictions = to_price_level(test, raw)
        kept_lags = [
            name
            for name in columns
            if name.startswith("lag_") and best_hp.values.get(f"use_{name}", True)
        ]
    finally:
        if directory is None:
            shutil.rmtree(workdir, ignore_errors=True)

    elapsed = (pd.Timestamp.utcnow() - started).total_seconds()
    metrics = all_metrics(test[TARGET_COLUMN], predictions)
    LOGGER.info(
        "kerastuner mlp: mae=%.2f rmse=%.2f lags kept=%s (%.1fs)",
        metrics["mae"],
        metrics["rmse"],
        kept_lags,
        elapsed,
    )
    return {
        "method": "kerastuner_mlp",
        "window_id": window.window_id,
        "seed": seed,
        "max_trials": max_trials,
        "epochs": epochs,
        "lags_kept": "|".join(kept_lags),
        "seconds": round(elapsed, 2),
        **metrics,
    }
