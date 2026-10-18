"""Split the weekly timeline into ordered train / validation / test windows.

A *window* is one snapshot of a deployed model's life: it is fitted on the training
weeks, its hyperparameters are scored on the validation weeks, and it is finally
judged on the test weeks that come after both. Windows march forward through time,
so window 2's test block starts where window 1's ended.

Three invariants hold by construction and are checked in ``tests/test_split.py``:

1. Inside a window, ``train < valid < test`` in time, with no overlap.
2. Across windows, the test blocks are disjoint and strictly increasing.
3. No window ever trains on a week that a previous window has not yet reached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Window:
    """One time-ordered train / validation / test split.

    Attributes:
        window_id: One-based index of the window.
        train_start: First training week, inclusive.
        train_end: Last training week, inclusive.
        valid_start: First validation week, inclusive.
        valid_end: Last validation week, inclusive.
        test_start: First test week, inclusive.
        test_end: Last test week, inclusive.
    """

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def as_record(self) -> dict[str, Any]:
        """Flatten the window into a dictionary for tabular display.

        Returns:
            A mapping of field name to value, with dates as ``datetime.date``.
        """
        return {
            "window_id": self.window_id,
            "train_start": self.train_start.date(),
            "train_end": self.train_end.date(),
            "valid_start": self.valid_start.date(),
            "valid_end": self.valid_end.date(),
            "test_start": self.test_start.date(),
            "test_end": self.test_end.date(),
            "train_weeks": self.n_train_weeks,
        }

    @property
    def n_train_weeks(self) -> int:
        """Number of distinct weeks in the training block.

        Returns:
            The count of weekly periods from ``train_start`` to ``train_end``.
        """
        return int((self.train_end - self.train_start).days // 7) + 1


def make_windows(weeks: pd.DatetimeIndex, config: dict[str, Any]) -> list[Window]:
    """Lay out the evaluation windows over a sorted weekly index.

    Args:
        weeks: Unique, sorted weekly timestamps covering the whole dataset.
        config: Parsed ``configs/experiment.yaml``; the ``splits`` block is used.

    Returns:
        A list of :class:`Window` objects ordered by time.

    Raises:
        ValueError: If the timeline is too short for the requested layout.
    """
    splits = config["splits"]
    n_windows = int(splits["n_windows"])
    valid_weeks = int(splits["valid_weeks"])
    test_weeks = int(splits["test_weeks"])
    min_train = int(splits.get("min_train_weeks", 52))
    scheme = str(splits.get("scheme", "expanding"))
    train_weeks = splits.get("train_weeks")

    total = len(weeks)
    needed = min_train + valid_weeks + n_windows * test_weeks
    if total < needed:
        raise ValueError(
            f"timeline has {total} weeks but the layout needs at least {needed} "
            f"({min_train} train + {valid_weeks} valid + {n_windows}x{test_weeks} test)"
        )

    windows: list[Window] = []
    for index in range(n_windows):
        test_stop = total - (n_windows - 1 - index) * test_weeks
        test_begin = test_stop - test_weeks
        valid_begin = test_begin - valid_weeks
        if scheme == "rolling":
            if train_weeks is None:
                raise ValueError("splits.scheme='rolling' requires splits.train_weeks")
            train_begin = max(0, valid_begin - int(train_weeks))
        else:
            train_begin = 0
        if valid_begin - train_begin < min_train:
            raise ValueError(
                f"window {index + 1} would train on {valid_begin - train_begin} weeks, "
                f"below splits.min_train_weeks={min_train}"
            )
        windows.append(
            Window(
                window_id=index + 1,
                train_start=weeks[train_begin],
                train_end=weeks[valid_begin - 1],
                valid_start=weeks[valid_begin],
                valid_end=weeks[test_begin - 1],
                test_start=weeks[test_begin],
                test_end=weeks[test_stop - 1],
            )
        )
    LOGGER.info("built %d windows over %d weeks (scheme=%s)", len(windows), total, scheme)
    return windows


def weeks_index(frame: pd.DataFrame, column: str = "week_start") -> pd.DatetimeIndex:
    """Extract the sorted unique weekly index from a frame.

    Args:
        frame: Any frame with a weekly timestamp column.
        column: Name of that column.

    Returns:
        A sorted :class:`pandas.DatetimeIndex` of unique weeks.
    """
    return pd.DatetimeIndex(sorted(frame[column].unique()))


def slice_window(
    frame: pd.DataFrame, window: Window, part: str, column: str = "week_start"
) -> pd.DataFrame:
    """Select the rows of one part of a window.

    Args:
        frame: Feature frame with a weekly timestamp column.
        window: The window to slice.
        part: One of ``train``, ``valid`` or ``test``.
        column: Name of the weekly timestamp column.

    Returns:
        The matching rows, in their original order.

    Raises:
        ValueError: If ``part`` is not a recognised block name.
    """
    bounds = {
        "train": (window.train_start, window.train_end),
        "valid": (window.valid_start, window.valid_end),
        "test": (window.test_start, window.test_end),
    }
    if part not in bounds:
        raise ValueError(f"unknown window part {part!r}; expected train, valid or test")
    start, end = bounds[part]
    mask = (frame[column] >= start) & (frame[column] <= end)
    return frame.loc[mask]


def windows_table(windows: list[Window]) -> pd.DataFrame:
    """Render a list of windows as a dataframe for printing.

    Args:
        windows: The windows to display.

    Returns:
        One row per window with all boundary dates.
    """
    return pd.DataFrame([window.as_record() for window in windows])
