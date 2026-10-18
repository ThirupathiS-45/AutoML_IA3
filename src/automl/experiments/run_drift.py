"""Phase 5: the core experiment. Four retraining strategies, every window, every seed.

What happens for one seed:

**Window 1** is the deployment moment. A cold Bayesian search runs once and its winner
becomes "the model in production". Its trial history becomes the shared starting
knowledge that the warm strategies will later inherit. Every strategy therefore begins
from exactly the same place.

**Every later window** is a moment where the market may have moved. First the drift
check runs. Then five strategies are compared under an identical trial budget:

``no_retrain``
    Nothing happens. The production model, still fitted on window 1's data with window
    1's hyperparameters, keeps predicting. Costs zero trials.
``refit``
    Window 1's hyperparameters are kept but the model is refitted on current data.
    Costs zero trials.
``cold_bo``
    A fresh Bayesian search with no memory of anything.
``warm_bo_all``
    A Bayesian search whose surrogate is pre-loaded with every trial this lineage has
    run in earlier windows.
``warm_bo_topk``
    A Bayesian search that spends its first ``top_k`` trials re-measuring the best
    configurations from earlier windows.

Each warm strategy keeps its *own* accumulating history, so one strategy's trials never
help another's. Results go to ``results/drift_experiment.csv``, with the per-trial
detail in ``results/drift_traces.csv``.

Run with::

    uv run python -m automl.experiments.run_drift --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl.config import ensure_parent, setup_logging
from automl.drift import detect_input_drift, error_drift_verdict
from automl.evaluate import evaluate_frozen_model, evaluate_on_test, make_objective
from automl.features import feature_columns
from automl.pipeline import Setup, build_setup
from automl.split import Window, slice_window
from automl.tuner.bo_tuner import COLD, WARM_ALL, WARM_TOPK, BOTuner
from automl.tuner.history import TrialHistory
from automl.tuner.space import SearchSpace

LOGGER = logging.getLogger(__name__)

WARM_STRATEGIES = {"warm_bo_all": WARM_ALL, "warm_bo_topk": WARM_TOPK}


def _trials_to_target(history: TrialHistory, target: float) -> int | None:
    """Count trials until the running best reached a target score.

    Args:
        history: The trials of one search, in order.
        target: The validation error to reach.

    Returns:
        The one-based trial number at which the running best first became less than
        or equal to ``target``, or ``None`` if it never did.
    """
    running = float("inf")
    for index, trial in enumerate(history.trials, start=1):
        running = min(running, trial.score)
        if running <= target:
            return index
    return None


def _run_search(
    setup: Setup,
    space: SearchSpace,
    window: Window,
    seed: int,
    budget: int,
    n_initial_random: int,
    mode: str,
    prior: TrialHistory | None,
    top_k: int,
) -> tuple[TrialHistory, float, list[dict[str, Any]]]:
    """Run one Bayesian search and time it.

    Args:
        setup: The loaded experiment setup.
        space: The hyperparameter space.
        window: The window being tuned.
        seed: Random seed.
        budget: Trial budget.
        n_initial_random: Random trials before the surrogate is used.
        mode: ``cold``, ``warm_all`` or ``warm_topk``.
        prior: Earlier trials for the warm modes.
        top_k: How many past configs ``warm_topk`` re-evaluates.

    Returns:
        A tuple of (history, wall-clock seconds, per-trial trace rows).
    """
    objective = make_objective(setup.features, window, seed)
    tuner = BOTuner(space, objective, seed=seed, n_initial_random=n_initial_random)
    started = time.perf_counter()
    history = tuner.run(
        budget=budget, window_id=window.window_id, mode=mode, prior=prior, top_k=top_k
    )
    elapsed = time.perf_counter() - started
    return history, elapsed, tuner.trace


def _drift_record(setup: Setup, previous: Window, current: Window, alpha: float) -> dict[str, Any]:
    """Run the input-drift check between two consecutive windows' test blocks.

    Args:
        setup: The loaded experiment setup.
        previous: The earlier window.
        current: The later window.
        alpha: Significance level.

    Returns:
        A flat record of the drift verdict.
    """
    reference = slice_window(setup.features, previous, "test")
    live = slice_window(setup.features, current, "test")
    numeric = [
        column
        for column in feature_columns(setup.features)
        if pd.api.types.is_numeric_dtype(setup.features[column])
    ]
    return detect_input_drift(reference, live, columns=numeric, alpha=alpha).as_record()


def run(config_path: str | Path = "configs/experiment.yaml") -> pd.DataFrame:
    """Execute the full drift experiment.

    Args:
        config_path: Path to ``configs/experiment.yaml``.

    Returns:
        The long results table, one row per (seed, window, strategy).
    """
    setup = build_setup(experiment_config_path=config_path)
    config = setup.experiment_config
    space = SearchSpace.from_config(setup.search_space_config)

    budget = int(config["budget"]["trials"])
    n_initial_random = int(config["budget"]["n_initial_random"])
    top_k = int(config["warm_start"]["top_k"])
    alpha = float(config["drift"]["p_value_threshold"])
    error_margin = float(config["drift"].get("error_rise_margin", 0.10))
    seeds = [int(seed) for seed in config["seeds"]]
    history_dir = ensure_parent(Path(config["paths"]["history_dir"]) / "placeholder").parent

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for seed in seeds:
        LOGGER.info("=" * 70)
        LOGGER.info("seed %d", seed)
        first = setup.windows[0]

        base_history, base_seconds, base_trace = _run_search(
            setup, space, first, seed, budget, n_initial_random, COLD, None, top_k
        )
        base_history.save(history_dir / f"seed{seed}_window1_cold.json", space.names)
        best_trial = base_history.best()
        assert best_trial is not None
        deployed_params = dict(best_trial.params)
        base_metrics = evaluate_on_test(setup.features, first, deployed_params, seed)

        for row in base_trace:
            traces.append({"seed": seed, "window_id": 1, "strategy": "cold_bo", **row})
        rows.append(
            {
                "seed": seed,
                "window_id": 1,
                "strategy": "cold_bo",
                "trials_used": len(base_history),
                "valid_mae": base_history.best_score(),
                "trials_to_cold_best": _trials_to_target(base_history, base_history.best_score()),
                "seconds": round(base_seconds, 3),
                "input_drift": None,
                "n_drifted_features": None,
                "error_ratio": None,
                "error_drift": None,
                "params": str(deployed_params),
                **base_metrics,
            }
        )

        # Each warm lineage starts from the same shared window-1 knowledge.
        lineage: dict[str, TrialHistory] = {
            name: TrialHistory(list(base_history.trials)) for name in WARM_STRATEGIES
        }
        previous_window = first
        previous_frozen_mae = base_metrics["mae"]

        for window in setup.windows[1:]:
            LOGGER.info("-" * 60)
            LOGGER.info("seed %d, window %d", seed, window.window_id)

            frozen_metrics = evaluate_frozen_model(
                setup.features, first, window, deployed_params, seed
            )
            drift = _drift_record(setup, previous_window, window, alpha)
            ratio, decayed = error_drift_verdict(
                previous_frozen_mae, frozen_metrics["mae"], margin=error_margin
            )
            drift["error_ratio"] = ratio
            drift["error_drift"] = decayed

            cold_history, cold_seconds, cold_trace = _run_search(
                setup, space, window, seed, budget, n_initial_random, COLD, None, top_k
            )
            cold_history.save(
                history_dir / f"seed{seed}_window{window.window_id}_cold.json", space.names
            )
            cold_target = cold_history.best_score()

            # --- strategies that spend no trials -----------------------------
            rows.append(
                {
                    "seed": seed,
                    "window_id": window.window_id,
                    "strategy": "no_retrain",
                    "trials_used": 0,
                    "valid_mae": np.nan,
                    "trials_to_cold_best": None,
                    "seconds": 0.0,
                    **drift,
                    "params": str(deployed_params),
                    **frozen_metrics,
                }
            )

            refit_started = time.perf_counter()
            refit_metrics = evaluate_on_test(setup.features, window, deployed_params, seed)
            rows.append(
                {
                    "seed": seed,
                    "window_id": window.window_id,
                    "strategy": "refit",
                    "trials_used": 0,
                    "valid_mae": np.nan,
                    "trials_to_cold_best": None,
                    "seconds": round(time.perf_counter() - refit_started, 3),
                    **drift,
                    "params": str(deployed_params),
                    **refit_metrics,
                }
            )

            # --- cold search --------------------------------------------------
            cold_best = cold_history.best()
            assert cold_best is not None
            cold_metrics = evaluate_on_test(setup.features, window, dict(cold_best.params), seed)
            for row in cold_trace:
                traces.append(
                    {"seed": seed, "window_id": window.window_id, "strategy": "cold_bo", **row}
                )
            rows.append(
                {
                    "seed": seed,
                    "window_id": window.window_id,
                    "strategy": "cold_bo",
                    "trials_used": len(cold_history),
                    "valid_mae": cold_target,
                    "trials_to_cold_best": _trials_to_target(cold_history, cold_target),
                    "seconds": round(cold_seconds, 3),
                    **drift,
                    "params": str(cold_best.params),
                    **cold_metrics,
                }
            )

            # --- warm searches --------------------------------------------------
            for name, mode in WARM_STRATEGIES.items():
                prior = lineage[name]
                warm_history, warm_seconds, warm_trace = _run_search(
                    setup, space, window, seed, budget, n_initial_random, mode, prior, top_k
                )
                warm_history.save(
                    history_dir / f"seed{seed}_window{window.window_id}_{name}.json", space.names
                )
                warm_best = warm_history.best()
                assert warm_best is not None
                warm_metrics = evaluate_on_test(
                    setup.features, window, dict(warm_best.params), seed
                )
                for row in warm_trace:
                    traces.append(
                        {"seed": seed, "window_id": window.window_id, "strategy": name, **row}
                    )
                rows.append(
                    {
                        "seed": seed,
                        "window_id": window.window_id,
                        "strategy": name,
                        "trials_used": len(warm_history),
                        "valid_mae": warm_history.best_score(),
                        "trials_to_cold_best": _trials_to_target(warm_history, cold_target),
                        "seconds": round(warm_seconds, 3),
                        **drift,
                        "params": str(warm_best.params),
                        **warm_metrics,
                    }
                )
                for trial in warm_history.trials:
                    lineage[name].add(trial)

            previous_window = window
            previous_frozen_mae = frozen_metrics["mae"]

    table = pd.DataFrame(rows)
    results_path = ensure_parent(config["paths"]["results_csv"])
    table.to_csv(results_path, index=False)
    traces_path = ensure_parent(
        str(results_path).replace("drift_experiment.csv", "drift_traces.csv")
    )
    pd.DataFrame(traces).to_csv(traces_path, index=False)
    LOGGER.info("wrote %s (%d rows) and %s", results_path, len(table), traces_path)
    return table


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    setup_logging()
    run(args.config)


if __name__ == "__main__":
    main()
