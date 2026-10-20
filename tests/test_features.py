"""Tests that the feature builder never lets future information reach the model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.features import TARGET_COLUMN, build_features, feature_columns, season_of_month

from .conftest import make_weekly_panel


def test_lag_equals_price_k_weeks_earlier(weekly_panel, features_config):
    built = build_features(weekly_panel, features_config)
    source = weekly_panel.set_index(["market", "week_start"])["modal_price"]
    sample = built[built["market"] == "A"].iloc[40:50]
    for row in sample.itertuples():
        for lag in features_config["lags_weeks"]:
            expected = source.get(("A", row.week_start - pd.Timedelta(weeks=lag)))
            assert getattr(row, f"lag_{lag}") == pytest.approx(expected)


def test_rolling_mean_uses_only_weeks_before_the_target(weekly_panel, features_config):
    built = build_features(weekly_panel, features_config)
    series = (
        weekly_panel[weekly_panel["market"] == "A"]
        .set_index("week_start")["modal_price"]
        .sort_index()
    )
    row = built[built["market"] == "A"].iloc[60]
    for window in features_config["rolling_mean_windows_weeks"]:
        past = [
            series.loc[row.week_start - pd.Timedelta(weeks=offset)]
            for offset in range(1, window + 1)
        ]
        assert row[f"roll_mean_{window}"] == pytest.approx(float(np.mean(past)))


def test_target_is_the_price_of_its_own_week(weekly_panel, features_config):
    built = build_features(weekly_panel, features_config)
    merged = built.merge(weekly_panel, on=["market", "week_start"], suffixes=("", "_source"))
    assert np.allclose(merged[TARGET_COLUMN], merged["modal_price_source"])


def test_perturbing_a_week_cannot_change_that_weeks_features(features_config):
    """The decisive leakage test.

    Change the price of exactly one week, then confirm that the feature row whose
    target is that week is byte-for-byte unchanged. If any feature peeked at the
    target week, this assertion fails.
    """
    original = make_weekly_panel(n_weeks=120, markets=("A",))
    perturbed = original.copy()
    victim_week = original["week_start"].iloc[80]
    perturbed.loc[perturbed["week_start"] == victim_week, "modal_price"] += 99_999.0

    before = build_features(original, features_config)
    after = build_features(perturbed, features_config)
    columns = feature_columns(before)

    row_before = before.loc[before["week_start"] == victim_week, columns].reset_index(drop=True)
    row_after = after.loc[after["week_start"] == victim_week, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(row_before, row_after)

    earlier = before["week_start"] < victim_week
    pd.testing.assert_frame_equal(
        before.loc[earlier, columns].reset_index(drop=True),
        after.loc[earlier, columns].reset_index(drop=True),
    )


def test_perturbation_does_reach_later_weeks(features_config):
    """Sanity check: the previous test would pass trivially if lags were all NaN."""
    original = make_weekly_panel(n_weeks=120, markets=("A",))
    perturbed = original.copy()
    victim_week = original["week_start"].iloc[80]
    perturbed.loc[perturbed["week_start"] == victim_week, "modal_price"] += 99_999.0

    before = build_features(original, features_config)
    after = build_features(perturbed, features_config)
    next_week = victim_week + pd.Timedelta(weeks=1)

    lag_before = before.loc[before["week_start"] == next_week, "lag_1"].iloc[0]
    lag_after = after.loc[after["week_start"] == next_week, "lag_1"].iloc[0]
    assert lag_after - lag_before == pytest.approx(99_999.0)


def test_markets_do_not_contaminate_each_other(features_config):
    original = make_weekly_panel(n_weeks=120, markets=("A", "B"))
    perturbed = original.copy()
    mask = perturbed["market"] == "B"
    perturbed.loc[mask, "modal_price"] += 12_345.0

    before = build_features(original, features_config)
    after = build_features(perturbed, features_config)
    columns = feature_columns(before)

    a_before = before.loc[before["market"] == "A", columns].reset_index(drop=True)
    a_after = after.loc[after["market"] == "A", columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a_before, a_after)


def test_lag_shorter_than_horizon_is_rejected(weekly_panel, features_config):
    features_config["target"]["horizon_weeks"] = 4
    with pytest.raises(ValueError, match="smaller than the forecast horizon"):
        build_features(weekly_panel, features_config)


def test_no_raw_price_column_leaks_into_the_feature_list(weekly_panel, features_config):
    built = build_features(weekly_panel, features_config)
    columns = feature_columns(built)
    assert "modal_price" not in columns
    assert TARGET_COLUMN not in columns
    assert "week_start" not in columns


def test_season_mapping_covers_every_month():
    seasons = {season_of_month(month) for month in range(1, 13)}
    assert seasons == {"winter", "summer", "monsoon", "post_monsoon"}


def test_lags_never_cross_a_crop_boundary(features_config):
    """A series is one crop in one market; onion's history must not leak into tomato's."""
    import numpy as np

    from automl.features import build_features

    panel = make_weekly_panel(n_weeks=120, markets=("A",))
    onion = panel.assign(crop="Onion")
    tomato = panel.assign(crop="Tomato", modal_price=panel["modal_price"] + 5000.0)
    both = pd.concat([onion, tomato], ignore_index=True)

    built = build_features(both, features_config)
    for crop, group in built.groupby("crop", observed=True):
        source = both[both["crop"] == crop].set_index("week_start")["modal_price"].sort_index()
        row = group.iloc[40]
        for lag in features_config["lags_weeks"]:
            expected = source.loc[row.week_start - pd.Timedelta(weeks=lag)]
            assert row[f"lag_{lag}"] == pytest.approx(expected)
    # And the two crops must not share a single feature value by accident.
    pivot = built.pivot_table(index="week_start", columns="crop", values="lag_1")
    assert not np.allclose(pivot["Onion"].dropna(), pivot["Tomato"].dropna())


def test_crop_is_not_a_model_feature(features_config):
    from automl.features import build_features, feature_columns

    panel = make_weekly_panel(n_weeks=80, markets=("A",)).assign(crop="Onion")
    built = build_features(panel, features_config)
    assert "crop" in built.columns
    assert "crop" not in feature_columns(built)
