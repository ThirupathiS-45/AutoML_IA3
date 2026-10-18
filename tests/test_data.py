"""Tests for cleaning and weekly aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.data.aggregate import check_stop_rule, coverage_table, select_markets, to_weekly
from automl.data.clean import clean_daily, drop_duplicates, flag_outliers, standardize_market_names

CONFIG = {
    "crop": "onion",
    "n_markets": 2,
    "max_missing_week_fraction": 0.10,
    "clean": {
        "price_column": "modal_price",
        "min_price": 100.0,
        "max_price": 20000.0,
        "outlier_log_mad_threshold": 5.0,
    },
    "aggregate": {"frequency": "W-MON", "agg": "mean", "min_days_per_week": 3},
    "coverage_tie_tolerance": 0.0,
    "candidate_markets": [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}],
}


def daily_frame(n_days: int = 400, markets=("Alpha", "Beta", "Gamma"), seed: int = 0):
    """Build a synthetic daily panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="D")
    frames = []
    for offset, market in enumerate(markets):
        price = 1500 + 200 * offset + rng.normal(0, 60, n_days)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "market": market,
                    "commodity": "Onion",
                    "min_price": price - 300,
                    "max_price": price + 300,
                    "modal_price": price,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_market_names_are_normalised():
    frame = daily_frame(10, markets=("  alpha  ",))
    assert standardize_market_names(frame)["market"].iloc[0] == "Alpha"


def test_duplicate_market_dates_are_dropped():
    frame = daily_frame(10, markets=("Alpha",))
    doubled = pd.concat([frame, frame], ignore_index=True)
    assert len(drop_duplicates(doubled)) == len(frame)


def test_implausible_prices_are_dropped():
    frame = daily_frame(50, markets=("Alpha",))
    frame.loc[0, "modal_price"] = 5.0
    frame.loc[1, "modal_price"] = 99_000.0
    frame.loc[2, "modal_price"] = np.nan
    cleaned = clean_daily(frame, CONFIG)
    assert len(cleaned) == len(frame) - 3


def test_a_real_price_spike_is_flagged_but_kept():
    frame = daily_frame(200, markets=("Alpha",))
    frame.loc[100, "modal_price"] = 15_000.0
    cleaned = clean_daily(frame, CONFIG)
    assert len(cleaned) == len(frame)
    assert cleaned["is_outlier"].sum() >= 1


def test_outlier_flagging_survives_a_constant_series():
    frame = daily_frame(50, markets=("Alpha",))
    frame["modal_price"] = 1500.0
    flagged = flag_outliers(frame, CONFIG)
    assert not flagged["is_outlier"].any()


def test_weekly_aggregation_averages_within_the_week():
    frame = daily_frame(28, markets=("Alpha",))
    weekly = to_weekly(clean_daily(frame, CONFIG), CONFIG)
    first = weekly.dropna(subset=["modal_price"]).iloc[0]
    members = frame[
        (frame["date"] >= first["week_start"])
        & (frame["date"] < first["week_start"] + pd.Timedelta(days=7))
    ]
    assert first["modal_price"] == pytest.approx(members["modal_price"].mean())


def test_sparse_weeks_become_missing_not_misleading():
    frame = daily_frame(60, markets=("Alpha",))
    first_week = frame["date"].iloc[0]
    drop = frame["date"].between(first_week, first_week + pd.Timedelta(days=4))
    frame = frame[~drop].reset_index(drop=True)
    weekly = to_weekly(clean_daily(frame, CONFIG), CONFIG)
    assert pd.isna(weekly["modal_price"].iloc[0])


def test_every_market_shares_one_continuous_weekly_grid():
    frame = daily_frame(200)
    short = frame[~((frame["market"] == "Gamma") & (frame["date"] > "2022-04-01"))]
    weekly = to_weekly(clean_daily(short, CONFIG), CONFIG)
    counts = weekly.groupby("market").size()
    assert counts.nunique() == 1


def test_coverage_table_counts_missing_weeks():
    weekly = to_weekly(clean_daily(daily_frame(200), CONFIG), CONFIG)
    coverage = coverage_table(weekly)
    assert set(coverage["market"]) == {"Alpha", "Beta", "Gamma"}
    assert (coverage["weeks_present"] <= coverage["weeks_total"]).all()


def test_market_selection_keeps_the_best_covered():
    weekly = to_weekly(clean_daily(daily_frame(300), CONFIG), CONFIG)
    selected, coverage = select_markets(weekly, CONFIG)
    assert selected["market"].nunique() == CONFIG["n_markets"]
    assert len(coverage) == 3


def test_equally_covered_markets_fall_back_to_the_configured_order():
    """A two-week coverage difference must not silently drop a priority market."""
    config = {**CONFIG, "coverage_tie_tolerance": 0.05, "n_markets": 2}
    frame = daily_frame(300)
    # Make Alpha very slightly worse covered than the others.
    frame = frame[~((frame["market"] == "Alpha") & (frame["date"] < "2022-01-17"))]
    weekly = to_weekly(clean_daily(frame, config), config)
    selected, _ = select_markets(weekly, config)
    assert "Alpha" in set(selected["market"])


def test_a_market_outside_the_tolerance_is_still_dropped():
    config = {**CONFIG, "coverage_tie_tolerance": 0.01, "n_markets": 2}
    frame = daily_frame(300)
    frame = frame[~((frame["market"] == "Alpha") & (frame["date"] < "2022-06-01"))]
    weekly = to_weekly(clean_daily(frame, config), config)
    selected, _ = select_markets(weekly, config)
    assert "Alpha" not in set(selected["market"])


def test_stop_rule_fires_above_the_limit():
    coverage = pd.DataFrame({"market": ["Alpha", "Beta"], "missing_fraction": [0.02, 0.35]})
    violations = check_stop_rule(coverage, ["Alpha", "Beta"], CONFIG)
    assert len(violations) == 1
    assert "Beta" in violations[0]


def test_stop_rule_passes_when_coverage_is_good():
    coverage = pd.DataFrame({"market": ["Alpha"], "missing_fraction": [0.01]})
    assert check_stop_rule(coverage, ["Alpha"], CONFIG) == []
