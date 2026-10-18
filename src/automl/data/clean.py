"""Clean daily mandi records: duplicates, market names and implausible prices.

Two outlier rules are applied, and both are deliberately conservative because a real
onion price spike is a signal we want the model to see, not noise to be smoothed away:

1. **Hard bounds.** A modal price outside ``clean.min_price`` .. ``clean.max_price``
   (rupees per quintal) is treated as a data-entry error and dropped.
2. **Robust z-score.** Within one market, the log price is compared against that
   market's median using the median absolute deviation (MAD). Points beyond
   ``clean.outlier_log_mad_threshold`` MADs are *flagged* in an ``is_outlier`` column
   but kept, so the choice can be revisited without re-downloading anything.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: Scale factor making the MAD a consistent estimator of the standard deviation
#: for normally distributed data.
MAD_TO_SIGMA = 1.4826


def standardize_market_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse whitespace and casing differences in the ``market`` column.

    Args:
        frame: Daily records with a ``market`` column.

    Returns:
        A copy whose market names are stripped and title-cased consistently.
    """
    out = frame.copy()
    out["market"] = (
        out["market"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.title()
    )
    return out


def drop_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated (market, date) rows, keeping the first occurrence.

    Args:
        frame: Daily records.

    Returns:
        A copy with at most one row per market and date.
    """
    before = len(frame)
    out = frame.drop_duplicates(subset=["market", "date"], keep="first").reset_index(drop=True)
    if before != len(out):
        LOGGER.info("dropped %d duplicate (market, date) rows", before - len(out))
    return out


def drop_implausible_prices(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Drop rows whose modal price is missing or outside the configured hard bounds.

    Args:
        frame: Daily records with a ``modal_price`` column.
        config: Parsed ``configs/data.yaml``.

    Returns:
        A copy containing only rows with a usable modal price.
    """
    clean_cfg = config["clean"]
    price = frame["modal_price"]
    keep = price.notna() & price.between(clean_cfg["min_price"], clean_cfg["max_price"])
    dropped = int((~keep).sum())
    if dropped:
        LOGGER.info("dropped %d rows outside price bounds or missing", dropped)
    return frame.loc[keep].reset_index(drop=True)


def flag_outliers(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Flag per-market price outliers using a robust z-score on the log price.

    Args:
        frame: Daily records with ``market`` and ``modal_price`` columns.
        config: Parsed ``configs/data.yaml``.

    Returns:
        A copy with an added boolean ``is_outlier`` column. Rows are kept either way.
    """
    threshold = float(config["clean"]["outlier_log_mad_threshold"])
    out = frame.copy()
    log_price = np.log(out["modal_price"])

    def _robust_z(series: pd.Series) -> pd.Series:
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - median).abs() / (MAD_TO_SIGMA * mad)

    scores = log_price.groupby(out["market"], observed=True).transform(_robust_z)
    out["is_outlier"] = scores > threshold
    LOGGER.info(
        "flagged %d of %d rows as outliers (threshold %.1f MAD on log price)",
        int(out["is_outlier"].sum()),
        len(out),
        threshold,
    )
    return out


def clean_daily(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Run the full cleaning pipeline over raw daily records.

    Args:
        frame: Concatenated raw records from :mod:`automl.data.load`.
        config: Parsed ``configs/data.yaml``.

    Returns:
        Cleaned daily records sorted by market then date, with an ``is_outlier`` column.
    """
    out = standardize_market_names(frame)
    out = drop_duplicates(out)
    out = drop_implausible_prices(out, config)
    out = flag_outliers(out, config)
    out = out.sort_values(["market", "date"], ignore_index=True)
    LOGGER.info("cleaned frame: %d rows across %d markets", len(out), out["market"].nunique())
    return out
