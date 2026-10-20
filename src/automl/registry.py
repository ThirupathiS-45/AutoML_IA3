"""Save and load the production model together with a model card.

A pickle on its own is an unlabelled jar. The card next to it records what the model
was trained on, how well it scored, which commit produced it and when -- enough that
somebody finding ``model.pkl`` six months from now can tell whether to trust it.

One model is registered per crop, each in its own directory under ``models/``, plus an
``index.json`` the web API reads to discover what is available.

Run with::

    uv run python -m automl.registry                 # every configured crop
    uv run python -m automl.registry --crop Onion    # just one
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

from automl.config import ensure_parent, load_config, resolve_path, setup_logging
from automl.evaluate import evaluate_on_test
from automl.features import build_forecast_rows, feature_columns, split_xy
from automl.models.lgbm import fit
from automl.pipeline import build_setup, crop_names, crop_slug, load_weekly
from automl.split import Window, slice_window
from automl.tuner.bo_tuner import BOTuner
from automl.tuner.space import SearchSpace

LOGGER = logging.getLogger(__name__)

MODELS_DIR = "models"
INDEX_PATH = "models/index.json"
SNAPSHOT_PATH = "models/weekly_snapshot.csv"

MODEL_FILE = "model.pkl"
CARD_FILE = "model_card.json"
SERVING_FILE = "serving_features.pkl"


def crop_dir(slug: str) -> Path:
    """Return the directory holding one crop's artefacts.

    Args:
        slug: Crop slug, as configured in ``configs/data.yaml``.

    Returns:
        The resolved directory path. Nothing is created.
    """
    return resolve_path(MODELS_DIR) / slug


def model_path(slug: str) -> Path:
    """Path of one crop's pickled model.

    Args:
        slug: Crop slug.

    Returns:
        The resolved path.
    """
    return crop_dir(slug) / MODEL_FILE


def card_path(slug: str) -> Path:
    """Path of one crop's model card.

    Args:
        slug: Crop slug.

    Returns:
        The resolved path.
    """
    return crop_dir(slug) / CARD_FILE


def serving_path(slug: str) -> Path:
    """Path of one crop's serving rows.

    Args:
        slug: Crop slug.

    Returns:
        The resolved path.
    """
    return crop_dir(slug) / SERVING_FILE


@dataclass
class ModelCard:
    """Metadata describing one saved model.

    Attributes:
        version: Semantic-ish version string for the artefact.
        created_utc: When the model was saved.
        git_commit: Short commit hash of the code that produced it.
        crop: The commodity modelled.
        crop_slug: Filesystem slug for the commodity.
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
    crop_slug: str
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
    model_path: str | Path,
    card_path: str | Path,
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


def save_serving_features(rows: pd.DataFrame, path: str | Path) -> Path:
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


def load_serving_features(path: str | Path) -> pd.DataFrame:
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


def load_model(model_path: str | Path) -> Any:
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


def load_card(card_path: str | Path) -> dict[str, Any]:
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


def register_crop(
    crop: str,
    seed: int = 0,
    budget: int | None = None,
    version: str = "1.0.0",
) -> ModelCard:
    """Tune one crop on its most recent window, fit the winner on all data, save it.

    Args:
        crop: Crop display name, as listed in ``configs/data.yaml``.
        seed: Random seed.
        budget: Trial budget. Defaults to the experiment config's value.
        version: Version string recorded in the card.

    Returns:
        The model card that was written.
    """
    from automl.evaluate import make_objective

    setup = build_setup(crop=crop)
    slug = crop_slug(setup.data_config, crop)
    window: Window = setup.windows[-1]
    space = SearchSpace.from_config(setup.search_space_config)
    trials = budget or int(setup.experiment_config["budget"]["trials"])
    n_initial = int(setup.experiment_config["budget"]["n_initial_random"])

    tuner = BOTuner(
        space,
        make_objective(setup.features, window, seed),
        seed=seed,
        n_initial_random=n_initial,
    )
    history = tuner.run(budget=trials, window_id=window.window_id)
    best = history.best()
    assert best is not None
    params = dict(best.params)

    metrics = evaluate_on_test(setup.features, window, params, seed)

    blocks = [slice_window(setup.features, window, part) for part in ("train", "valid", "test")]
    everything = pd.concat(blocks, ignore_index=True)
    x_all, y_all = split_xy(everything)
    model = fit(params, x_all, y_all, seed)

    card = ModelCard(
        version=version,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit(),
        crop=crop,
        crop_slug=slug,
        entity_level=str(setup.data_config["source"]["entity_level"]),
        markets=setup.markets,
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
        extra={
            "tuner": "custom GP Bayesian optimization",
            "budget": trials,
            "seed": seed,
            "target_mode": str(setup.features_config["target"].get("mode", "level")),
        },
    )
    save_model(model, card, model_path(slug), card_path(slug))

    weekly = load_weekly(setup.data_config)
    if "crop" in weekly.columns:
        weekly = weekly[weekly["crop"].astype(str).str.lower() == crop.lower()]
    save_serving_features(
        build_forecast_rows(weekly.reset_index(drop=True), setup.features_config),
        serving_path(slug),
    )
    return card


def write_index(cards: list[ModelCard]) -> Path:
    """Write the registry index the web API reads to discover available crops.

    Args:
        cards: One card per registered crop.

    Returns:
        The path written to.
    """
    path = ensure_parent(INDEX_PATH)
    payload = {
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "crops": [
            {
                "crop": card.crop,
                "slug": card.crop_slug,
                "version": card.version,
                "markets": card.markets,
                "test_mae": card.metrics.get("mae"),
                "horizon_weeks": card.horizon_weeks,
            }
            for card in sorted(cards, key=lambda c: c.crop)
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("registry index written to %s (%d crops)", path, len(cards))
    return path


def write_weekly_snapshot(weekly: pd.DataFrame, path: str | Path = SNAPSHOT_PATH) -> Path:
    """Ship the processed weekly price table alongside the models.

    The API computes every answer from this table at request time. It travels as CSV
    rather than parquet so the runtime image needs no pyarrow, which alone would add
    about 170 MB for a dependency the app would otherwise never call.

    Args:
        weekly: The processed weekly frame.
        path: Destination CSV path.

    Returns:
        The path written to.
    """
    resolved = ensure_parent(path)
    columns = [c for c in ("crop", "market", "week_start", "modal_price", "n_days") if c in weekly]
    weekly[columns].to_csv(resolved, index=False)
    LOGGER.info("weekly snapshot written to %s (%d rows)", resolved, len(weekly))
    return resolved


def load_index(path: str | Path = INDEX_PATH) -> dict[str, Any]:
    """Read the registry index.

    Args:
        path: Path to ``models/index.json``.

    Returns:
        The index contents.

    Raises:
        FileNotFoundError: If no crop has been registered yet.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{resolved} not found. Run: uv run python -m automl.registry")
    return json.loads(resolved.read_text(encoding="utf-8"))


def register_all(
    crops: list[str] | None = None,
    seed: int = 0,
    budget: int | None = None,
    version: str = "1.0.0",
) -> list[ModelCard]:
    """Register a model for every configured crop and write the index.

    Args:
        crops: Crop names to register. Defaults to every crop in the data config.
        seed: Random seed.
        budget: Trial budget per crop.
        version: Version string recorded in each card.

    Returns:
        The cards that were written. A crop that fails is logged and skipped so one
        bad series cannot block the rest of the registry.
    """
    data_config = load_config("configs/data.yaml")
    wanted = crops or crop_names(data_config)
    cards: list[ModelCard] = []
    for crop in wanted:
        LOGGER.info("=" * 60)
        LOGGER.info("registering %s", crop)
        try:
            cards.append(register_crop(crop, seed=seed, budget=budget, version=version))
        except Exception:
            LOGGER.exception("could not register %s; skipping it", crop)
    if cards:
        write_index(cards)
        write_weekly_snapshot(load_weekly(data_config))
    return cards


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", action="append", help="repeatable; defaults to all crops")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()
    setup_logging()
    cards = register_all(args.crop, args.seed, args.budget, args.version)
    for card in cards:
        print(f"{card.crop:10s} v{card.version}  test MAE {card.metrics['mae']:8.2f}")


if __name__ == "__main__":
    main()
