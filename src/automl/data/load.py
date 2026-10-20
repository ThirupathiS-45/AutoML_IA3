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
    "crop",
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

    required = [c for c in CANONICAL_COLUMNS if c != "crop"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{resolved.name} is missing columns {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("min_price", "max_price", "modal_price"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("market", "commodity", "crop"):
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()
    if "crop" not in frame.columns:
        # Files written before the multi-crop fetch carried no crop column; the
        # portal's own commodity label is the right fallback.
        frame["crop"] = frame["commodity"]

    frame = frame[CANONICAL_COLUMNS]
    LOGGER.info("loaded %s: %d rows", resolved.name, len(frame))
    return frame


def load_raw_directory(config: dict[str, Any]) -> pd.DataFrame:
    """Read and concatenate the raw CSV of every configured crop.

    Args:
        config: Parsed ``configs/data.yaml``.

    Returns:
        One dataframe holding every crop and market, sorted by crop, market, date.

    Raises:
        FileNotFoundError: If no raw CSV files are present.
    """
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    slugs = [str(crop["slug"]) for crop in config["crops"]]
    paths = sorted(path for slug in slugs for path in raw_dir.glob(f"{slug}_*_daily.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no raw CSVs in {raw_dir}. Run: uv run python -m automl.data.fetch"
        )
    frames = [load_raw_frame(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["crop", "market", "date"], ignore_index=True)
    LOGGER.info(
        "combined %d files into %d rows across %d crops",
        len(paths),
        len(combined),
        combined["crop"].nunique(),
    )
    return combined
