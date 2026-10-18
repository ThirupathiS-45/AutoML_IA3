"""Shared fixtures: small synthetic weekly panels that need no downloaded data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_weekly_panel(n_weeks: int = 160, markets: tuple[str, ...] = ("A", "B"), seed: int = 0):
    """Build a synthetic long weekly price panel.

    Args:
        n_weeks: Number of weeks per market.
        markets: Market names.
        seed: Random seed for the noise term.

    Returns:
        A dataframe with ``week_start``, ``market``, ``modal_price`` and ``n_days``.
    """
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2019-01-07", periods=n_weeks, freq="W-MON")
    frames = []
    for offset, market in enumerate(markets):
        trend = np.linspace(1000, 2000, n_weeks) + 300 * offset
        seasonal = 200 * np.sin(np.arange(n_weeks) * 2 * np.pi / 52)
        noise = rng.normal(0, 50, n_weeks)
        frames.append(
            pd.DataFrame(
                {
                    "week_start": weeks,
                    "market": market,
                    "modal_price": trend + seasonal + noise,
                    "n_days": 7,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def weekly_panel() -> pd.DataFrame:
    """A two-market, 160-week synthetic panel."""
    return make_weekly_panel()


@pytest.fixture
def features_config() -> dict:
    """A feature config mirroring ``configs/features.yaml``."""
    return {
        "target": {"column": "modal_price", "horizon_weeks": 1},
        "lags_weeks": [1, 2, 4, 8],
        "rolling_mean_windows_weeks": [4, 8],
        "calendar": {"month": True, "season": True},
        "categorical": ["market"],
    }


@pytest.fixture
def experiment_config() -> dict:
    """An experiment config mirroring ``configs/experiment.yaml``."""
    return {
        "splits": {
            "scheme": "expanding",
            "train_weeks": None,
            "n_windows": 4,
            "valid_weeks": 8,
            "test_weeks": 8,
            "min_train_weeks": 52,
        }
    }
