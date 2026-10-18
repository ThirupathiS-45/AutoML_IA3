"""Save and load the production model together with a model card.

A pickle on its own is an unlabelled jar. The card next to it records what the model
was trained on, how well it scored, which commit produced it and when -- enough that
somebody finding ``model.pkl`` six months from now can tell whether to trust it.

Run with::

    uv run python -m automl.registry --window -1
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from automl.config import ensure_parent, resolve_path, setup_logging
from automl.evaluate import evaluate_on_test
from automl.features import build_forecast_rows, feature_columns, split_xy
from automl.models.lgbm import fit
from automl.pipeline import build_setup, load_weekly
from automl.split import Window, slice_window
from automl.tuner.bo_tuner import BOTuner
from automl.tuner.space import SearchSpace

LOGGER = logging.getLogger(__name__)

MODEL_PATH = "models/model.pkl"
CARD_PATH = "models/model_card.json"
SERVING_PATH = "models/serving_features.pkl"


@dataclass
class ModelCard:
    """Metadata describing one saved model.

    Attributes:
        version: Semantic-ish version string for the artefact.
        created_utc: When the model was saved.
        git_commit: Short commit hash of the code that produced it.
        crop: The commodity modelled.
        entity_level: Whether rows are mandis, districts or states.
        markets: The market names the model was trained on.
        train_start: First training week.
        train_end: Last training week used for fitting.
        test_start: First week of the held-out test block.
        test_end: Last week of the held-out test block.
        horizon_weeks: How far ahead the model predicts.
        hyperparameters: The tuned LightGBM settings.
        metrics: Test metrics measured on the held-out block.
        feature_names: Model input columns, in order.
        n_train_rows: Number of rows the model was fitted on.
        notes: Free text.
    """

    version: str
    created_utc: str
    git_commit: str
    crop: str
    entity_level: str
    markets: list[str]
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    horizon_weeks: int
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    feature_names: list[str]
    n_train_rows: int
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def git_commit() -> str:
    """Return the short git commit hash of the working tree.

    Returns:
        The short hash, or ``"unknown"`` outside a git repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=resolve_path("."),
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_model(
    model: Any,
    card: ModelCard,
    model_path: str | Path = MODEL_PATH,
    card_path: str | Path = CARD_PATH,
) -> tuple[Path, Path]:
    """Write the model and its card to disk.

    Args:
        model: The fitted estimator.
        card: Its metadata.
        model_path: Where to write the pickle.
        card_path: Where to write the JSON card.

    Returns:
        A tuple of (model path, card path).
    """
    model_file = ensure_parent(model_path)
    card_file = ensure_parent(card_path)
    joblib.dump(model, model_file)
    card_file.write_text(json.dumps(asdict(card), indent=2, default=str) + "\n", encoding="utf-8")
    LOGGER.info("saved model to %s and card to %s", model_file, card_file)
    return model_file, card_file


def _to_numpy_backed(frame: pd.DataFrame) -> pd.DataFrame:
    """Rewrite PyArrow-backed string dtypes as plain NumPy object columns.

    pandas 3 stores strings in a PyArrow-backed dtype by default. Pickling such a
    frame embeds a reference to ``pyarrow``, which the runtime image deliberately
    does not install -- it alone adds about 170 MB for a library the app never calls.
    Converting to object dtype keeps the values identical while making the artefact
    loadable with pandas alone.

    Args:
        frame: The frame about to be pickled.

    Returns:
        A copy with no PyArrow-backed columns.
    """
    out = frame.copy()
    for column in out.columns:
        dtype = out[column].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            categories = dtype.categories.astype(object)
            out[column] = pd.Categorical(
                out[column].astype(object), categories=categories, ordered=dtype.ordered
            )
        elif pd.api.types.is_string_dtype(dtype) and dtype is not object:
            out[column] = out[column].astype(object)
    # The column index is itself a string Index, and in pandas 3 that is also
    # PyArrow-backed. Missing it leaves a pyarrow reference in the pickle even when
    # every column has been converted.
    out.columns = pd.Index(list(out.columns), dtype=object)
    out.index = pd.RangeIndex(len(out))
    return out


def save_serving_features(rows: pd.DataFrame, path: str | Path = SERVING_PATH) -> Path:
    """Store one forecast row per market, for the web app to serve.

    The app must be able to answer "what is next week's price?" without shipping the
    whole dataset into the container, so the single latest row per market travels
    with the model. A joblib pickle is used rather than parquet so the runtime image
    does not need pyarrow, which alone would add about 170 MB.

    Args:
        rows: Forecast rows from :func:`automl.features.build_forecast_rows`, one per
            market and dated one horizon beyond the last observation.
        path: Destination pickle path.

    Returns:
        The path written to.
    """
    resolved = ensure_parent(path)
    joblib.dump(_to_numpy_backed(rows.reset_index(drop=True)), resolved)
    LOGGER.info("saved %d serving rows to %s", len(rows), resolved)
    return resolved


def load_serving_features(path: str | Path = SERVING_PATH) -> pd.DataFrame:
    """Load the per-market serving rows saved alongside the model.

    Args:
        path: Path to the pickle file.

    Returns:
        One row per market.

    Raises:
        FileNotFoundError: If the file has not been written yet.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} not found. Run: uv run python -m automl.registry")
    return joblib.load(resolved)


def load_model(model_path: str | Path = MODEL_PATH) -> Any:
    """Load the pickled production model.

    Args:
        model_path: Path to the pickle.

    Returns:
        The fitted estimator.

    Raises:
        FileNotFoundError: If no model has been registered yet.
    """
    resolved = resolve_path(model_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} not found. Run: uv run python -m automl.registry")
    return joblib.load(resolved)


def load_card(card_path: str | Path = CARD_PATH) -> dict[str, Any]:
    """Load the model card.

    Args:
        card_path: Path to the JSON card.

    Returns:
        The card contents.

    Raises:
        FileNotFoundError: If no card has been written yet.
    """
    resolved = resolve_path(card_path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} not found. Run: uv run python -m automl.registry")
    return json.loads(resolved.read_text(encoding="utf-8"))


def register_best_model(
    window_index: int = -1,
    seed: int = 0,
    budget: int | None = None,
    version: str = "0.1.0",
) -> ModelCard:
    """Tune on the chosen window, fit the winner on all available data and save it.

    Args:
        window_index: Which window to tune on. ``-1`` means the most recent.
        seed: Random seed.
        budget: Trial budget. Defaults to the experiment config's value.
        version: Version string recorded in the card.

    Returns:
        The model card that was written.
    """
    setup = build_setup()
    window: Window = setup.windows[window_index]
    space = SearchSpace.from_config(setup.search_space_config)
    trials = budget or int(setup.experiment_config["budget"]["trials"])
    n_initial = int(setup.experiment_config["budget"]["n_initial_random"])

    from automl.evaluate import make_objective

    tuner = BOTuner(
        space, make_objective(setup.features, window, seed), seed=seed, n_initial_random=n_initial
    )
    history = tuner.run(budget=trials, window_id=window.window_id)
    best = history.best()
    assert best is not None
    params = dict(best.params)

    metrics = evaluate_on_test(setup.features, window, params, seed)

    train = slice_window(setup.features, window, "train")
    valid = slice_window(setup.features, window, "valid")
    test = slice_window(setup.features, window, "test")
    everything = pd.concat([train, valid, test], ignore_index=True)
    x_all, y_all = split_xy(everything)
    model = fit(params, x_all, y_all, seed)

    card = ModelCard(
        version=version,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        crop=str(setup.data_config["crop"]),
        entity_level=str(setup.data_config["source"]["entity_level"]),
        markets=sorted(str(name) for name in setup.features["market"].unique()),
        train_start=str(window.train_start.date()),
        train_end=str(window.test_end.date()),
        test_start=str(window.test_start.date()),
        test_end=str(window.test_end.date()),
        horizon_weeks=int(setup.features_config["target"]["horizon_weeks"]),
        hyperparameters=params,
        metrics={key: round(value, 3) for key, value in metrics.items()},
        feature_names=feature_columns(setup.features),
        n_train_rows=int(len(everything)),
        notes=(
            "Metrics come from the held-out test block of the tuning window. The "
            "shipped model is then refitted on train+valid+test so it is as current "
            "as possible; those metrics are therefore out-of-sample for the tuning "
            "run but not for the shipped weights."
        ),
        extra={"tuner": "custom GP Bayesian optimization", "budget": trials, "seed": seed},
    )
    save_model(model, card)
    weekly = load_weekly(setup.data_config)
    save_serving_features(build_forecast_rows(weekly, setup.features_config))
    return card


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args()
    setup_logging()
    card = register_best_model(args.window, args.seed, args.budget, args.version)
    print(json.dumps(asdict(card), indent=2, default=str))


if __name__ == "__main__":
    main()
