"""Phase 3: score every baseline on window 1 and write ``results/baselines.csv``.

Four reference points, cheapest first:

1. ``naive_last_week`` -- repeat last week's price.
2. ``naive_mean_4w`` -- the average of the last four weeks.
3. ``lightgbm_default`` -- LightGBM with its own default settings, untuned.
4. ``autokeras_structured`` and ``kerastuner_mlp`` -- the two deep-learning AutoML
   baselines from course Modules 2 and 3. Both are slow on a CPU and are skipped
   unless ``--with-deep`` is passed.

Run with::

    uv run python -m automl.experiments.run_baselines --with-deep
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from automl.baselines.naive import predict_last_week, predict_rolling_mean
from automl.config import ensure_parent, setup_logging
from automl.evaluate import evaluate_on_test
from automl.features import TARGET_COLUMN
from automl.metrics import all_metrics
from automl.models.lgbm import default_params
from automl.pipeline import build_setup
from automl.split import Window, slice_window

LOGGER = logging.getLogger(__name__)


def naive_records(features: pd.DataFrame, window: Window) -> list[dict[str, Any]]:
    """Score the two naive forecasters on a window's test block.

    Args:
        features: The full feature frame.
        window: The window to score.

    Returns:
        One record per naive method.
    """
    test = slice_window(features, window, "test")
    y_test = test[TARGET_COLUMN]
    records = []
    for name, predictions in (
        ("naive_last_week", predict_last_week(test)),
        ("naive_mean_4w", predict_rolling_mean(test, window=4)),
    ):
        usable = pd.notna(predictions)
        metrics = all_metrics(y_test[usable], predictions[usable])
        records.append(
            {
                "method": name,
                "window_id": window.window_id,
                "seed": 0,
                "n_test_rows": int(usable.sum()),
                "seconds": 0.0,
                **metrics,
            }
        )
    return records


def lightgbm_default_record(
    features: pd.DataFrame, window: Window, seed: int = 0
) -> dict[str, Any]:
    """Score untuned LightGBM on a window's test block.

    Args:
        features: The full feature frame.
        window: The window to score.
        seed: Random seed.

    Returns:
        One result record.
    """
    started = time.perf_counter()
    metrics = evaluate_on_test(features, window, default_params(), seed)
    return {
        "method": "lightgbm_default",
        "window_id": window.window_id,
        "seed": seed,
        "n_test_rows": len(slice_window(features, window, "test")),
        "seconds": round(time.perf_counter() - started, 3),
        **metrics,
    }


def run(window_index: int, with_deep: bool, output: str | Path) -> pd.DataFrame:
    """Run every baseline and write the results CSV.

    Args:
        window_index: Zero-based index of the window to evaluate.
        with_deep: Whether to include the AutoKeras and KerasTuner baselines.
        output: Destination CSV path.

    Returns:
        The results table.
    """
    setup = build_setup()
    window = setup.windows[window_index]
    LOGGER.info("baselines on window %d (%s)", window.window_id, window.as_record())

    records: list[dict[str, Any]] = []
    records.extend(naive_records(setup.features, window))
    records.append(lightgbm_default_record(setup.features, window))

    if with_deep:
        from automl.baselines.autokeras_baseline import run_autokeras
        from automl.baselines.kerastuner_mlp import run_kerastuner_mlp

        n_test = len(slice_window(setup.features, window, "test"))
        for runner in (run_autokeras, run_kerastuner_mlp):
            try:
                record = runner(setup.features, window, seed=0)
                record["n_test_rows"] = n_test
                records.append(record)
            except Exception:
                LOGGER.exception("%s failed; recording nothing for it", runner.__name__)
    else:
        LOGGER.info("skipping deep baselines; pass --with-deep to include them")

    table = pd.DataFrame(records)
    path = ensure_parent(output)
    table.to_csv(path, index=False)
    LOGGER.info("wrote %s", path)
    return table


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--window", type=int, default=0, help="zero-based window index")
    parser.add_argument("--with-deep", action="store_true")
    parser.add_argument("--output", default="results/baselines.csv")
    args = parser.parse_args()
    setup_logging()
    table = run(args.window, args.with_deep, args.output)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
