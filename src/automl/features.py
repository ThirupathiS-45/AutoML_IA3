"""Build past-only features for weekly mandi price forecasting.

**Indexing convention (this is what makes leakage testable).** Every row is indexed
by the *target* week ``T``. The label is the price in week ``T``. Every feature must
be computable strictly before week ``T`` begins:

- ``lag_k`` is the price in week ``T - k``. The smallest ``k`` allowed is the forecast
  horizon, so a 1-week-ahead model may use ``lag_1`` (last week) but never week ``T``.
- ``roll_mean_w`` is the mean price over weeks ``T - w`` .. ``T - 1`` inclusive.
- Calendar features describe week ``T`` itself, which is legitimate: the month and
  season of a future week are known today without observing any price.

Everything is computed per **series**, where a series is one crop in one market. Onion
in Gujarat and tomato in Gujarat are separate histories, and a lag must never reach
across either boundary.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

#: India Meteorological Department season for each calendar month.
MONTH_TO_SEASON: dict[int, str] = {
    1: "winter",
    2: "winter",
    3: "summer",
    4: "summer",
    5: "summer",
    6: "monsoon",
    7: "monsoon",
    8: "monsoon",
    9: "monsoon",
    10: "post_monsoon",
    11: "post_monsoon",
    12: "winter",
}

#: The price level in the target week. Always present; every metric is measured here.
TARGET_COLUMN = "target_price"
#: What the model is actually fitted on. Equals TARGET_COLUMN in "level" mode, or the
#: week-on-week change in "delta" mode.
MODEL_TARGET_COLUMN = "target_model"
PRICE_COLUMN = "modal_price"
#: Baseline column a delta prediction is added back onto.
ANCHOR_COLUMN = "lag_1"
#: Records how the model was parameterised, so serving never has to guess.
MODE_COLUMN = "target_mode"
#: Bookkeeping column naming the crop. Never a model input: a per-crop model sees a
#: single constant value, so feeding it in would add a column of no information.
CROP_COLUMN = "crop"


def season_of_month(month: int) -> str:
    """Map a calendar month to an Indian meteorological season.

    Args:
        month: Month number, 1 through 12.

    Returns:
        One of ``winter``, ``summer``, ``monsoon`` or ``post_monsoon``.

    Raises:
        KeyError: If ``month`` is not between 1 and 12.
    """
    return MONTH_TO_SEASON[month]


def _validate_lags(lags: list[int], horizon: int) -> None:
    """Fail loudly if any lag would reach into the forecast horizon.

    Args:
        lags: Requested lag offsets in weeks.
        horizon: Forecast horizon in weeks.

    Raises:
        ValueError: If any lag is smaller than the horizon.
    """
    bad = [lag for lag in lags if lag < horizon]
    if bad:
        raise ValueError(
            f"lags {bad} are smaller than the forecast horizon {horizon}; "
            "they would use data that is not available at prediction time"
        )


def series_keys(frame: pd.DataFrame) -> list[str]:
    """Return the columns that together identify one price series.

    Args:
        frame: A weekly or feature frame.

    Returns:
        ``["crop", "market"]`` when a crop column is present, else ``["market"]``.
    """
    return [CROP_COLUMN, "market"] if CROP_COLUMN in frame.columns else ["market"]


def _add_feature_columns(weekly: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Attach every past-only feature column to a long weekly frame.

    Shared by :func:`build_features` and :func:`build_forecast_rows` so that a row
    served in production is built by exactly the same code as a row used in training.

    Args:
        weekly: Long frame with ``week_start``, ``market`` and ``modal_price``.
        config: Parsed ``configs/features.yaml``.

    Returns:
        A sorted copy with the feature columns added.
    """
    lags = [int(lag) for lag in config["lags_weeks"]]
    windows = [int(window) for window in config["rolling_mean_windows_weeks"]]

    keys = series_keys(weekly)
    frame = weekly.sort_values([*keys, "week_start"], ignore_index=True).copy()
    grouped = frame.groupby(keys, observed=True)[PRICE_COLUMN]

    for lag in lags:
        frame[f"lag_{lag}"] = grouped.shift(lag)

    for window in windows:
        # shift(1) first so the window ends at T-1 and never touches week T.
        frame[f"roll_mean_{window}"] = grouped.transform(
            lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).mean()
        )
        frame[f"roll_std_{window}"] = grouped.transform(
            lambda series, w=window: series.shift(1).rolling(window=w, min_periods=w).std()
        )

    if config["calendar"].get("month", True):
        frame["month"] = frame["week_start"].dt.month
    if config["calendar"].get("season", True):
        frame["season"] = frame["week_start"].dt.month.map(MONTH_TO_SEASON).astype("category")

    for column in config.get("categorical", []):
        frame[column] = frame[column].astype("category")

    return frame


def build_forecast_rows(weekly: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build the feature row for the next, not-yet-observed week of each market.

    The training table only contains weeks whose price is known, so its last row
    predicts a week that already happened. Serving a real forecast needs one extra
    row per market, dated one horizon ahead, whose lags and rolling statistics come
    from the observed history. No target exists for these rows, by definition.

    Args:
        weekly: The long weekly frame.
        config: Parsed ``configs/features.yaml``.

    Returns:
        One row per market for the next unobserved week.
    """
    horizon = int(config["target"]["horizon_weeks"])
    keys = series_keys(weekly)
    extended = weekly.sort_values([*keys, "week_start"], ignore_index=True).copy()

    future = []
    for combo, group in extended.groupby(keys, observed=True):
        combo = combo if isinstance(combo, tuple) else (combo,)
        last_week = group["week_start"].max()
        for step in range(1, horizon + 1):
            future.append(
                {
                    **dict(zip(keys, combo, strict=True)),
                    "week_start": last_week + pd.Timedelta(weeks=step),
                    PRICE_COLUMN: np.nan,
                    "n_days": 0,
                }
            )
    extended = pd.concat([extended, pd.DataFrame(future)], ignore_index=True)

    built = _add_feature_columns(extended, config)
    built[MODE_COLUMN] = str(config["target"].get("mode", "level"))
    horizon_week = built.groupby(keys, observed=True)["week_start"].transform("max")
    rows = built[built["week_start"] == horizon_week].reset_index(drop=True)
    LOGGER.info("built %d forecast rows for week(s) after the last observation", len(rows))
    return rows


def build_features(weekly: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Turn a long weekly price frame into a supervised learning table.

    Args:
        weekly: Long frame with ``week_start``, ``market`` and ``modal_price`` columns,
            on a continuous weekly grid per market.
        config: Parsed ``configs/features.yaml``.

    Returns:
        One row per (market, target week) with the feature columns, the
        ``target_price`` label, and the bookkeeping columns ``week_start`` and
        ``market``. Rows whose label is missing are dropped; rows with missing
        features are kept, because LightGBM handles them natively.

    Raises:
        ValueError: If a requested lag is shorter than the forecast horizon.
    """
    horizon = int(config["target"]["horizon_weeks"])
    _validate_lags([int(lag) for lag in config["lags_weeks"]], horizon)

    frame = _add_feature_columns(weekly, config)

    frame[TARGET_COLUMN] = frame[PRICE_COLUMN]

    mode = str(config["target"].get("mode", "level"))
    frame[MODE_COLUMN] = mode
    if mode == "level":
        frame[MODEL_TARGET_COLUMN] = frame[TARGET_COLUMN]
    elif mode == "delta":
        if ANCHOR_COLUMN not in frame.columns:
            raise ValueError("target.mode='delta' requires a lag_1 feature to anchor on")
        frame[MODEL_TARGET_COLUMN] = frame[TARGET_COLUMN] - frame[ANCHOR_COLUMN]
    else:
        raise ValueError(f"unknown target.mode {mode!r}; expected 'level' or 'delta'")

    # The price label must exist. In delta mode the anchor must exist too, because a
    # change relative to a missing week is undefined.
    required = [TARGET_COLUMN, MODEL_TARGET_COLUMN]
    frame = frame.dropna(subset=required).reset_index(drop=True)

    LOGGER.info(
        "built %d feature rows, %d feature columns, %d series",
        len(frame),
        len(feature_columns(frame)),
        len(frame.groupby(series_keys(frame), observed=True)),
    )
    return frame


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """List the model input columns of a feature frame.

    Args:
        frame: Output of :func:`build_features`.

    Returns:
        Column names excluding the label and the bookkeeping columns.
    """
    excluded = {
        TARGET_COLUMN,
        MODEL_TARGET_COLUMN,
        MODE_COLUMN,
        CROP_COLUMN,
        PRICE_COLUMN,
        "week_start",
        "n_days",
    }
    return [column for column in frame.columns if column not in excluded]


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate a feature frame into the model matrix and the fitting label.

    Args:
        frame: Output of :func:`build_features`.

    Returns:
        A tuple of (features, the column the model is fitted on). In ``delta`` mode
        that label is a week-on-week change, not a price.
    """
    label = MODEL_TARGET_COLUMN if MODEL_TARGET_COLUMN in frame.columns else TARGET_COLUMN
    return frame[feature_columns(frame)], frame[label]


def target_mode_of(frame: pd.DataFrame) -> str:
    """Determine how a frame's model target was parameterised.

    Prefers the explicit :data:`MODE_COLUMN` stamp, which is the only thing available
    on forecast rows, and falls back to comparing the label columns on a training
    frame.

    Args:
        frame: A feature or forecast frame.

    Returns:
        Either ``"level"`` or ``"delta"``.
    """
    if MODE_COLUMN in frame.columns and len(frame):
        return str(frame[MODE_COLUMN].iloc[0])
    return "delta" if is_delta_frame(frame) else "level"


def to_price_level(frame: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    """Convert raw model output back into a price in rupees per quintal.

    In ``level`` mode this is the identity. In ``delta`` mode the predicted change is
    added back onto last week's price, so every metric is reported on the same scale
    regardless of how the model was trained.

    Args:
        frame: The rows the predictions correspond to, in the same order.
        predictions: Raw model output.

    Returns:
        Predicted prices.
    """
    values = np.asarray(predictions, dtype=float).ravel()
    if target_mode_of(frame) != "delta":
        return values
    return values + np.asarray(frame[ANCHOR_COLUMN], dtype=float)


def is_delta_frame(frame: pd.DataFrame) -> bool:
    """Detect whether a feature frame was built in delta mode.

    Args:
        frame: Output of :func:`build_features`.

    Returns:
        ``True`` when the fitting label differs from the price level.
    """
    if MODEL_TARGET_COLUMN not in frame.columns:
        return False
    difference = (frame[MODEL_TARGET_COLUMN] - frame[TARGET_COLUMN]).abs()
    return bool(difference.max() > 1e-9)
