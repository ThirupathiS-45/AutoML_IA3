"""Shared setup used by every experiment: load the weekly series, build features, split."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from automl.config import load_config, resolve_path
from automl.features import build_features
from automl.split import Window, make_windows, weeks_index

LOGGER = logging.getLogger(__name__)


@dataclass
class Setup:
    """Everything an experiment needs before it starts running trials.

    Attributes:
        features: The supervised feature table.
        windows: The ordered evaluation windows.
        data_config: Parsed ``configs/data.yaml``.
        features_config: Parsed ``configs/features.yaml``.
        experiment_config: Parsed ``configs/experiment.yaml``.
        search_space_config: Parsed ``configs/search_space.yaml``.
    """

    features: pd.DataFrame
    windows: list[Window]
    data_config: dict
    features_config: dict
    experiment_config: dict
    search_space_config: dict


def load_weekly(data_config: dict) -> pd.DataFrame:
    """Read the processed weekly parquet produced in Phase 1.

    Args:
        data_config: Parsed ``configs/data.yaml``.

    Returns:
        The long weekly frame.

    Raises:
        FileNotFoundError: If Phase 1 has not been run.
    """
    path = resolve_path(data_config["paths"]["weekly_parquet"])
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run: uv run python -m automl.data.aggregate")
    frame = pd.read_parquet(path)
    LOGGER.info("loaded %d weekly rows from %s", len(frame), path.name)
    return frame


def build_setup(
    data_config_path: str | Path = "configs/data.yaml",
    features_config_path: str | Path = "configs/features.yaml",
    experiment_config_path: str | Path = "configs/experiment.yaml",
    search_space_config_path: str | Path = "configs/search_space.yaml",
) -> Setup:
    """Load every config, build the feature table and lay out the windows.

    Args:
        data_config_path: Path to the data config.
        features_config_path: Path to the feature config.
        experiment_config_path: Path to the experiment config.
        search_space_config_path: Path to the search space config.

    Returns:
        A fully populated :class:`Setup`.
    """
    data_config = load_config(data_config_path)
    features_config = load_config(features_config_path)
    experiment_config = load_config(experiment_config_path)
    search_space_config = load_config(search_space_config_path)

    weekly = load_weekly(data_config)
    features = build_features(weekly, features_config)
    windows = make_windows(weeks_index(features), experiment_config)

    return Setup(
        features=features,
        windows=windows,
        data_config=data_config,
        features_config=features_config,
        experiment_config=experiment_config,
        search_space_config=search_space_config,
    )
