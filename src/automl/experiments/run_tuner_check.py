"""Phase 4 evidence: our Bayesian optimizer against random search, same budget.

Random search is the honest control. It is not a straw man -- on small budgets and
low-dimensional spaces it is often competitive, and if our surrogate-driven search
cannot beat it then the extra machinery is not earning its place.

Both methods get the identical trial budget, the identical window and the identical
seeds, and both are scored on validation MAE, which is the quantity they optimise.
Test MAE is also reported, measured once after the search, to show whether any
validation gain survived onto held-out weeks.

Run with::

    uv run python -m automl.experiments.run_tuner_check --window 0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from automl.config import ensure_parent, setup_logging
from automl.evaluate import evaluate_on_test, make_objective
from automl.pipeline import build_setup
from automl.tuner.bo_tuner import BOTuner, random_search
from automl.tuner.space import SearchSpace

LOGGER = logging.getLogger(__name__)


def run(
    window_index: int, budget: int | None, seeds: list[int], output: str | Path
) -> pd.DataFrame:
    """Compare the custom tuner against random search on one window.

    Args:
        window_index: Zero-based index of the window to tune on.
        budget: Trial budget. Defaults to the experiment config's value.
        seeds: Random seeds to repeat over.
        output: Destination CSV path.

    Returns:
        One row per (method, seed).
    """
    setup = build_setup()
    window = setup.windows[window_index]
    space = SearchSpace.from_config(setup.search_space_config)
    trials = budget or int(setup.experiment_config["budget"]["trials"])
    n_initial = int(setup.experiment_config["budget"]["n_initial_random"])

    records: list[dict[str, Any]] = []
    for seed in seeds:
        objective = make_objective(setup.features, window, seed)
        runners = {
            "random_search": lambda obj=objective, s=seed: random_search(
                space, obj, trials, seed=s, window_id=window.window_id
            ),
            "custom_bo": lambda obj=objective, s=seed: BOTuner(
                space, obj, seed=s, n_initial_random=n_initial
            ).run(budget=trials, window_id=window.window_id),
        }
        for method, runner in runners.items():
            started = time.perf_counter()
            history = runner()
            elapsed = time.perf_counter() - started
            best = history.best()
            assert best is not None
            metrics = evaluate_on_test(setup.features, window, dict(best.params), seed)
            LOGGER.info(
                "%s seed=%d: valid MAE %.2f, test MAE %.2f (%.1fs)",
                method,
                seed,
                best.score,
                metrics["mae"],
                elapsed,
            )
            records.append(
                {
                    "method": method,
                    "window_id": window.window_id,
                    "seed": seed,
                    "budget": trials,
                    "valid_mae": round(float(best.score), 3),
                    "test_mae": round(metrics["mae"], 3),
                    "test_rmse": round(metrics["rmse"], 3),
                    "seconds": round(elapsed, 2),
                    "params": str(best.params),
                }
            )

    table = pd.DataFrame(records)
    path = ensure_parent(output)
    table.to_csv(path, index=False)
    LOGGER.info("wrote %s", path)
    return table


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", default="results/tuner_vs_random.csv")
    args = parser.parse_args()
    setup_logging()
    table = run(args.window, args.budget, args.seeds, args.output)
    print()
    print(
        table.groupby("method")[["valid_mae", "test_mae", "seconds"]]
        .agg(["mean", "std"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
