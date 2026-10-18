"""AutoKeras ``StructuredDataRegressor`` baseline (course Module 2).

AutoKeras is the "just give it the table" end of AutoML: it picks the preprocessing,
the network shape and the training settings itself. It is included here as a
reference point, not as a competitor -- on a few hundred weekly rows a neural network
is not expected to beat gradient-boosted trees, and the run is kept deliberately small
so it finishes on a CPU.

Research-only: needs the ``research`` dependency group and is never imported by the
Flask app or the Docker image.
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


def _encode_jointly(frames: list[pd.DataFrame]) -> list[np.ndarray]:
    """One-hot encode categoricals consistently across several frames.

    AutoKeras 3.0 rejects pandas DataFrames outright -- a break from the 1.0.x API the
    course textbook uses -- so the features must be handed over as a dense float
    array. Encoding the frames together guarantees the train and test matrices share
    a column layout.

    Args:
        frames: Feature matrices to encode, in order.

    Returns:
        One float array per input frame, with the same row counts.
    """
    lengths = [len(frame) for frame in frames]
    combined = pd.concat(frames, ignore_index=True)
    categorical = [
        column
        for column in combined.columns
        if str(combined[column].dtype) in {"category", "object", "string"}
    ]
    encoded = pd.get_dummies(combined, columns=categorical).astype(float)
    encoded = encoded.fillna(encoded.median(numeric_only=True)).fillna(0.0)
    matrix = encoded.to_numpy(dtype="float32")

    out, start = [], 0
    for length in lengths:
        out.append(matrix[start : start + length])
        start += length
    return out


def run_autokeras(
    features: pd.DataFrame,
    window: Window,
    seed: int = 0,
    max_trials: int = 3,
    epochs: int = 30,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Fit an AutoKeras regressor on one window and score it on the test block.

    Args:
        features: The full feature frame.
        window: The window to evaluate.
        seed: Random seed for TensorFlow and NumPy.
        max_trials: How many architectures AutoKeras may try. Kept small for CPU.
        epochs: Training epochs per trial.
        directory: Where AutoKeras writes its search state. A temporary directory is
            used and deleted when omitted.

    Returns:
        A record with the metrics, the configuration used and the wall-clock seconds.
    """
    import tensorflow as tf
    from autokeras import AutoModel, StructuredDataBlock, StructuredDataInput
    from autokeras.blocks import RegressionHead

    class _IntShapeRegressionHead(RegressionHead):
        """Coerce the head's output shape to Python ints before building.

        AutoKeras 3.0.0 derives its output shape from NumPy and hands Keras 3 a
        ``np.int64``. Keras 3's ``Dense`` checks ``isinstance(units, int)``, which a
        NumPy integer fails, so the model never builds. This shim is the smallest
        change that gets the documented baseline running; it alters no behaviour.
        """

        def build(self, hp, inputs=None):
            """Build the head with an integer-typed output width.

            Args:
                hp: KerasTuner hyperparameter container.
                inputs: Upstream tensors.

            Returns:
                The head's output tensor.
            """
            self.shape = tuple(int(dim) for dim in self.shape)
            return super().build(hp, inputs)

    tf.keras.utils.set_random_seed(seed)

    train = slice_window(features, window, "train")
    valid = slice_window(features, window, "valid")
    test = slice_window(features, window, "test")

    x_train_df, y_train = split_xy(pd.concat([train, valid], ignore_index=True))
    x_test_df, _ = split_xy(test)
    x_train, x_test = _encode_jointly([x_train_df, x_test_df])

    workdir = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="autokeras_"))
    started = pd.Timestamp.utcnow()
    try:
        # AutoModel with an explicit StructuredDataBlock and RegressionHead is what
        # StructuredDataRegressor builds internally; spelling it out lets the shim
        # above replace the head without patching the library globally.
        input_node = StructuredDataInput()
        hidden = StructuredDataBlock(normalize=True)(input_node)
        output_node = _IntShapeRegressionHead(
            output_dim=1, loss="mean_absolute_error", metrics=["mean_absolute_error"]
        )(hidden)
        model = AutoModel(
            inputs=input_node,
            outputs=output_node,
            max_trials=max_trials,
            overwrite=True,
            directory=str(workdir),
            project_name="mandi_price",
            seed=seed,
            objective="val_loss",
        )
        model.fit(
            x_train,
            np.asarray(y_train, dtype="float32"),
            epochs=epochs,
            validation_split=0.2,
            verbose=0,
        )
        raw = np.asarray(model.predict(x_test, verbose=0), dtype=float).ravel()
        predictions = to_price_level(test, raw)
    finally:
        if directory is None:
            shutil.rmtree(workdir, ignore_errors=True)

    elapsed = (pd.Timestamp.utcnow() - started).total_seconds()
    metrics = all_metrics(test[TARGET_COLUMN], predictions)
    LOGGER.info("autokeras: mae=%.2f rmse=%.2f (%.1fs)", metrics["mae"], metrics["rmse"], elapsed)
    return {
        "method": "autokeras_structured",
        "window_id": window.window_id,
        "seed": seed,
        "max_trials": max_trials,
        "epochs": epochs,
        "seconds": round(elapsed, 2),
        **metrics,
    }
