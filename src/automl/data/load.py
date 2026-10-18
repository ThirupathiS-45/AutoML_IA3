"""Read raw mandi CSV files from ``data/raw`` and standardize their columns.

The CEDA portal returns short column names (``t``, ``p_modal`` and so on). Everything
downstream expects the canonical names defined in :data:`CANONICAL_COLUMNS`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from automl.config import resolve_path

LOGGER = logging.getLogger(__name__)

#: Raw column name -> canonical column name.
COLUMN_RENAMES: dict[str, str] = {
    "t": "date",
    "cmdty": "commodity",
    "p_min": "min_price",
    "p_max": "max_price",
    "p_modal": "modal_price",
}

#: Columns every loaded frame is guaranteed to have, in order.
CANONICAL_COLUMNS: list[str] = [
    "date",
    "market",
    "commodity",
    "min_price",
    "max_price",
    "modal_price",
]


def load_raw_frame(path: str | Path) -> pd.DataFrame:
    """Read one raw CSV and rename its columns to the canonical names.

    Args:
        path: Path to a raw CSV written by ``automl.data.fetch``.

    Returns:
        A dataframe with the columns in :data:`CANONICAL_COLUMNS`, dates parsed and
        prices coerced to float.

    Raises:
        ValueError: If an expected column is missing after renaming.
    """
    resolved = resolve_path(path)
    frame = pd.read_csv(resolved)
    frame = frame.rename(columns=COLUMN_RENAMES)

    missing = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{resolved.name} is missing columns {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("min_price", "max_price", "modal_price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["market"] = frame["market"].astype("string").str.strip()
    frame["commodity"] = frame["commodity"].astype("string").str.strip()

    frame = frame[CANONICAL_COLUMNS]
    LOGGER.info("loaded %s: %d rows", resolved.name, len(frame))
    return frame


def load_raw_directory(config: dict[str, Any]) -> pd.DataFrame:
    """Read and concatenate every raw onion CSV named in the download manifest.

    Args:
        config: Parsed ``configs/data.yaml``.

    Returns:
        One dataframe holding every market, sorted by market then date.

    Raises:
        FileNotFoundError: If no raw CSV files are present.
    """
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    paths = sorted(raw_dir.glob(f"{config['crop']}_*_daily.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no raw CSVs in {raw_dir}. Run: uv run python -m automl.data.fetch"
        )
    frames = [load_raw_frame(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["market", "date"], ignore_index=True)
    LOGGER.info("combined %d files into %d rows", len(paths), len(combined))
    return combined
