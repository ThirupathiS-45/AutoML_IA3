"""Phase 9: point the same tuner at a completely different problem.

If the Bayesian optimizer in ``automl.tuner`` is a general tool rather than something
quietly shaped around LightGBM, it should tune an unrelated pipeline without a single
change to its code. Here it tunes a TF-IDF plus linear-classifier pipeline on the
20 Newsgroups text dataset. Only the search space and the objective differ; the
space, surrogate, acquisition and loop modules are imported untouched.

The comparison is against random search under the same budget and seeds.

Run with::

    uv run python -m automl.experiments.run_generality --config configs/generality.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from automl.config import ensure_parent, load_config, setup_logging
from automl.tuner.bo_tuner import BOTuner, random_search
from automl.tuner.space import SearchSpace

LOGGER = logging.getLogger(__name__)


def load_newsgroups(config: dict[str, Any]) -> tuple[list[str], Any, list[str], Any]:
    """Fetch the 20 Newsgroups train and test splits.

    Args:
        config: Parsed ``configs/generality.yaml``.

    Returns:
        A tuple of (train texts, train labels, test texts, test labels).
    """
    from sklearn.datasets import fetch_20newsgroups

    kwargs = {
        "categories": list(config["data"]["categories"]),
        "remove": tuple(config["data"]["remove"]),
        "shuffle": True,
        "random_state": 0,
    }
    train = fetch_20newsgroups(subset="train", **kwargs)
    test = fetch_20newsgroups(subset="test", **kwargs)
    LOGGER.info("20 newsgroups: %d train, %d test docs", len(train.data), len(test.data))
    return train.data, train.target, test.data, test.target


def make_text_objective(x_train: list[str], y_train: Any, seed: int):
    """Build the objective: cross-validated error of a TF-IDF plus SGD pipeline.

    Args:
        x_train: Training documents.
        y_train: Training labels.
        seed: Random seed.

    Returns:
        A callable mapping hyperparameters to ``1 - accuracy``, which the tuner
        minimises just as it minimises MAE on the price task.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    def objective(params: dict[str, Any]) -> float:
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        min_df=int(params["tfidf_min_df"]),
                        max_features=int(params["tfidf_max_features"]),
                        ngram_range=(1, int(params["tfidf_ngram_max"])),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "clf",
                    SGDClassifier(
                        loss="modified_huber",
                        alpha=float(params["sgd_alpha"]),
                        max_iter=1500,
                        tol=1e-4,
                        random_state=seed,
                    ),
                ),
            ]
        )
        scores = cross_val_score(pipeline, x_train, y_train, cv=3, scoring="accuracy", n_jobs=1)
        return float(1.0 - scores.mean())

    return objective


def run(config_path: str | Path) -> pd.DataFrame:
    """Compare the custom tuner against random search on the text task.

    Args:
        config_path: Path to ``configs/generality.yaml``.

    Returns:
        The results table.
    """
    config = load_config(config_path)
    space = SearchSpace.from_config(config)
    budget = int(config["budget"]["trials"])
    n_initial = int(config["budget"]["n_initial_random"])

    x_train, y_train, x_test, y_test = load_newsgroups(config)

    records: list[dict[str, Any]] = []
    for seed in [int(value) for value in config["seeds"]]:
        objective = make_text_objective(x_train, y_train, seed)

        runners = {
            "random_search": lambda obj=objective, s=seed: random_search(
                space, obj, budget, seed=s
            ),
            "custom_bo": lambda obj=objective, s=seed: BOTuner(
                space, obj, seed=s, n_initial_random=n_initial
            ).run(budget=budget),
        }
        for name, runner in runners.items():
            started = time.perf_counter()
            history = runner()
            elapsed = time.perf_counter() - started
            best = history.best()
            assert best is not None
            accuracy = _test_accuracy(dict(best.params), x_train, y_train, x_test, y_test, seed)
            LOGGER.info(
                "%s seed=%d: cv error %.4f, test accuracy %.4f (%.1fs)",
                name,
                seed,
                best.score,
                accuracy,
                elapsed,
            )
            records.append(
                {
                    "method": name,
                    "seed": seed,
                    "budget": budget,
                    "best_cv_error": round(float(best.score), 5),
                    "test_accuracy": round(accuracy, 5),
                    "seconds": round(elapsed, 2),
                    "params": str(best.params),
                }
            )

    table = pd.DataFrame(records)
    path = ensure_parent(config["paths"]["results_csv"])
    table.to_csv(path, index=False)
    LOGGER.info("wrote %s", path)
    return table


def _test_accuracy(
    params: dict[str, Any],
    x_train: list[str],
    y_train: Any,
    x_test: list[str],
    y_test: Any,
    seed: int,
) -> float:
    """Refit the winning configuration and score it on the held-out test split.

    Args:
        params: The chosen hyperparameters.
        x_train: Training documents.
        y_train: Training labels.
        x_test: Test documents.
        y_test: Test labels.
        seed: Random seed.

    Returns:
        Test-set accuracy.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=int(params["tfidf_min_df"]),
                    max_features=int(params["tfidf_max_features"]),
                    ngram_range=(1, int(params["tfidf_ngram_max"])),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="modified_huber",
                    alpha=float(params["sgd_alpha"]),
                    max_iter=1500,
                    tol=1e-4,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return float(pipeline.score(x_test, y_test))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generality.yaml")
    args = parser.parse_args()
    setup_logging()
    table = run(args.config)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
