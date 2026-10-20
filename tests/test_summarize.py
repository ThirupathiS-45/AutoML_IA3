"""Tests for the experiment summariser, especially the paired statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.experiments.summarize import (
    head_to_head,
    paired_tests,
    summarise_by_strategy,
    summarise_trials_to_target,
    summarise_validation,
)

STRATEGIES = ["no_retrain", "refit", "cold_bo", "warm_bo_all", "warm_bo_topk"]


def synthetic_results(seed: int = 0, warm_edge: float = 10.0) -> pd.DataFrame:
    """Build a results table where warm strategies are consistently a little better.

    Args:
        seed: Random seed.
        warm_edge: How many rupees better the warm strategies are, per run.

    Returns:
        A table shaped like ``results/drift_experiment.csv``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(5):
        for window in range(1, 6):
            # Window difficulty is shared by every strategy: this is exactly the
            # variance a paired test is supposed to remove.
            difficulty = 100.0 + 60.0 * window
            for strategy in STRATEGIES:
                if window == 1 and strategy != "cold_bo":
                    continue
                edge = warm_edge if strategy.startswith("warm") else 0.0
                rows.append(
                    {
                        "crop": "Onion",
                        "seed": s,
                        "window_id": window,
                        "strategy": strategy,
                        "trials_used": 15 if "bo" in strategy else 0,
                        "valid_mae": difficulty * 0.9 if "bo" in strategy else None,
                        "trials_to_cold_best": 8 if "bo" in strategy else None,
                        "seconds": 1.0,
                        "mae": difficulty - edge + rng.normal(0, 2),
                        "rmse": difficulty * 1.4,
                        "mape": 7.0,
                        "n_drifted_features": 9,
                        "n_features_tested": 9,
                        "input_drift": True,
                        "error_ratio": 1.1,
                        "error_drift": True,
                    }
                )
    return pd.DataFrame(rows)


def test_window_one_is_excluded_from_strategy_means():
    table = summarise_by_strategy(synthetic_results())
    assert set(table["strategy"]) <= set(STRATEGIES)
    assert (table["n_runs"] == 20).all()


def test_paired_test_detects_a_consistent_small_edge():
    """A 10-rupee edge hidden under 300 rupees of window variance must still show up."""
    table = paired_tests(synthetic_results(warm_edge=10.0))
    row = table[table["comparison"] == "warm_bo_all vs cold_bo"].iloc[0]
    assert row["mean_mae_difference"] < 0
    assert row["p_value"] < 0.01
    assert row["significant_after_holm"]


def test_paired_test_finds_nothing_when_there_is_nothing():
    table = paired_tests(synthetic_results(warm_edge=0.0))
    assert not table["significant_after_holm"].any()


def test_paired_test_reports_pair_counts_and_wins():
    table = paired_tests(synthetic_results())
    assert (table["n_pairs"] == 20).all()
    for wins in table["wins"]:
        count, total = wins.split("/")
        assert 0 <= int(count) <= int(total) == 20


def test_paired_test_applies_holm_not_raw_thresholds():
    """With many comparisons, a borderline raw p-value must not pass as significant."""
    table = paired_tests(synthetic_results(warm_edge=2.2))
    borderline = table[(table["p_value"] > 0.01) & (table["p_value"] < 0.05)]
    for _, row in borderline.iterrows():
        assert not row["significant_after_holm"] or row["p_value"] <= 0.05 / len(table)


def test_validation_table_covers_only_searches():
    table = summarise_validation(synthetic_results())
    assert set(table["strategy"]) == {"cold_bo", "warm_bo_all", "warm_bo_topk"}


def test_head_to_head_excludes_the_reference_itself():
    table = head_to_head(synthetic_results(), "cold_bo")
    assert "cold_bo" not in set(table["strategy"])
    assert "beat_cold_bo" in table.columns


def test_trials_to_target_reports_a_fraction():
    table = summarise_trials_to_target(synthetic_results())
    for value in table["reached_target"]:
        assert "/" in value


def test_paired_test_returns_an_empty_table_rather_than_crashing():
    """Too few matched runs is a reason to report nothing, not to raise."""
    tiny = synthetic_results().head(6)
    table = paired_tests(tiny)
    assert table.empty
    assert "significant_after_holm" in table.columns


def test_summary_is_reproducible_for_a_fixed_seed():
    first = paired_tests(synthetic_results(seed=3))
    second = paired_tests(synthetic_results(seed=3))
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("edge", [0.0, 5.0, 25.0])
def test_mean_difference_tracks_the_injected_edge(edge):
    table = paired_tests(synthetic_results(warm_edge=edge))
    row = table[table["comparison"] == "warm_bo_all vs cold_bo"].iloc[0]
    assert row["mean_mae_difference"] == pytest.approx(-edge, abs=2.5)
