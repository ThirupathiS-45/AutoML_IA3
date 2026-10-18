"""Phase 6: measure the speed-up from evaluating trials in parallel (course Module 5).

Bayesian optimization is sequential by nature: each proposal is supposed to learn from
every previous result. To use more than one CPU core we relax that slightly into
**batch Bayesian optimization**. Each round, the acquisition function ranks the random
candidate pool and we take the top ``n_workers`` points instead of just the best one,
evaluate all of them at once, then refit the surrogate on the whole batch.

The trade is explicit: a batch of four proposals is chosen with less information than
four sequential proposals would have been, so the same trial budget may find a slightly
worse configuration. What we buy is wall-clock time. Both numbers are recorded so the
trade can be judged rather than assumed.

Run with::

    uv run python -m automl.experiments.run_parallel --workers 1 2 4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from automl.config import ensure_parent, setup_logging
from automl.evaluate import make_objective
from automl.features import split_xy
from automl.metrics import all_metrics
from automl.models.lgbm import fit_predict
from automl.pipeline import build_setup
from automl.split import Window, slice_window
from automl.tuner.acquisition import expected_improvement
from automl.tuner.space import SearchSpace
from automl.tuner.surrogate import GaussianProcessSurrogate

LOGGER = logging.getLogger(__name__)


def _score_params(
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    seed: int,
) -> float:
    """Train one configuration and return its validation MAE.

    Defined at module level so :mod:`joblib` can pickle it for worker processes.

    Args:
        params: Hyperparameters in natural units.
        x_train: Training features.
        y_train: Training labels.
        x_valid: Validation features.
        y_valid: Validation labels.
        seed: Random seed.

    Returns:
        Validation mean absolute error, or infinity if the fit failed.
    """
    try:
        predictions = fit_predict(params, x_train, y_train, x_valid, seed)
        return all_metrics(y_valid, predictions)["mae"]
    except Exception:  # noqa: BLE001 - a bad config must not kill the batch
        return float("inf")


def _pick_batch(
    space: SearchSpace,
    surrogate: GaussianProcessSurrogate,
    rng: np.random.Generator,
    best_score: float,
    batch_size: int,
    n_candidates: int = 2048,
) -> list[np.ndarray]:
    """Choose ``batch_size`` distinct candidates by Expected Improvement.

    Args:
        space: The hyperparameter space.
        surrogate: The fitted surrogate.
        rng: Seeded random generator.
        best_score: Lowest validation error so far.
        batch_size: How many points to return.
        n_candidates: Size of the random pool to rank.

    Returns:
        A list of candidate vectors, best first.
    """
    candidates = space.sample(rng, n_candidates)
    mean, std = surrogate.predict(candidates)
    reference = best_score if np.isfinite(best_score) else float(np.min(mean))
    scores = expected_improvement(mean, std, reference)
    order = np.argsort(-scores)[:batch_size]
    return [candidates[index] for index in order]


def batch_search(
    setup_features: pd.DataFrame,
    window: Window,
    space: SearchSpace,
    budget: int,
    n_workers: int,
    seed: int,
    n_initial_random: int = 5,
) -> dict[str, Any]:
    """Run one batch Bayesian search and time it.

    Args:
        setup_features: The full feature frame.
        window: The window to tune on.
        space: The hyperparameter space.
        budget: Total number of evaluations.
        n_workers: Parallel worker processes, and therefore the batch size.
        seed: Random seed.
        n_initial_random: Random evaluations before the surrogate is used.

    Returns:
        A record with the worker count, elapsed seconds and best validation MAE.
    """
    train = slice_window(setup_features, window, "train")
    valid = slice_window(setup_features, window, "valid")
    x_train, y_train = split_xy(train)
    x_valid, y_valid = split_xy(valid)

    rng = np.random.default_rng(seed)
    vectors: list[np.ndarray] = list(space.sample(rng, min(n_initial_random, budget)))
    scores: list[float] = []

    started = time.perf_counter()
    with Parallel(n_jobs=n_workers, backend="loky") as parallel:
        pending = vectors
        while True:
            batch_params = [space.from_vector(vector) for vector in pending]
            batch_scores = parallel(
                delayed(_score_params)(params, x_train, y_train, x_valid, y_valid, seed)
                for params in batch_params
            )
            scores.extend(float(score) for score in batch_scores)
            if len(scores) >= budget:
                break

            finite = np.isfinite(np.array(scores))
            surrogate = GaussianProcessSurrogate(len(space), seed=seed)
            if finite.sum() >= 2:
                surrogate.fit(np.array(vectors)[finite], np.array(scores)[finite])
            remaining = budget - len(scores)
            pending = _pick_batch(
                space,
                surrogate,
                rng,
                float(np.min(scores)) if scores else float("inf"),
                min(n_workers, remaining),
            )
            vectors.extend(pending)
    elapsed = time.perf_counter() - started

    usable = [score for score in scores[:budget] if np.isfinite(score)]
    return {
        "n_workers": n_workers,
        "budget": budget,
        "seed": seed,
        "window_id": window.window_id,
        "seconds": round(elapsed, 3),
        "best_valid_mae": min(usable) if usable else float("nan"),
        "n_evaluations": len(scores[:budget]),
    }


def run(
    workers: list[int],
    budget: int,
    seeds: list[int],
    window_index: int,
    output: str | Path,
) -> pd.DataFrame:
    """Time the batch search at several worker counts and write the results CSV.

    Args:
        workers: Worker counts to measure.
        budget: Trial budget, identical for every worker count.
        seeds: Random seeds to repeat over.
        window_index: Zero-based index of the window to tune on.
        output: Destination CSV path.

    Returns:
        The timing table, including the speed-up relative to a single worker.
    """
    setup = build_setup()
    space = SearchSpace.from_config(setup.search_space_config)
    window = setup.windows[window_index]

    # Warm the process pool and any lazy imports so the first timing is not inflated.
    make_objective(setup.features, window, seeds[0])(space.from_vector(np.full(len(space), 0.5)))

    records = []
    for seed in seeds:
        for n_workers in workers:
            record = batch_search(setup.features, window, space, budget, n_workers, seed)
            LOGGER.info(
                "workers=%d seed=%d -> %.2fs, best valid MAE %.2f",
                n_workers,
                seed,
                record["seconds"],
                record["best_valid_mae"],
            )
            records.append(record)

    table = pd.DataFrame(records)
    baseline = (
        table[table["n_workers"] == 1].groupby("seed")["seconds"].mean().rename("baseline_seconds")
    )
    table = table.merge(baseline, on="seed", how="left")
    table["speedup_vs_1_worker"] = (table["baseline_seconds"] / table["seconds"]).round(3)

    path = ensure_parent(output)
    table.to_csv(path, index=False)
    LOGGER.info("wrote %s", path)
    return table


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--output", default="results/parallel.csv")
    args = parser.parse_args()
    setup_logging()
    table = run(args.workers, args.budget, args.seeds, args.window, args.output)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
