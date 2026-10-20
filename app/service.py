"""Live data and model access for the API.

Everything here is computed on demand from the processed dataset on disk. Nothing is
pre-baked: a prediction is produced by building the forecast feature row at request
time and running the registered model over it, and the backtest fits nothing new but
genuinely re-runs the model across past weeks so the chart shows real errors.

The dataset and the models are cached in memory and reloaded automatically when the
files on disk change, so re-running ``automl.data.aggregate`` or ``automl.registry``
is picked up without restarting the server.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl.config import load_config, resolve_path
from automl.features import (
    TARGET_COLUMN,
    build_features,
    build_forecast_rows,
    feature_columns,
    split_xy,
    target_mode_of,
    to_price_level,
)
from automl.metrics import all_metrics
from automl.registry import card_path, load_card, load_index, load_model, model_path

LOGGER = logging.getLogger(__name__)


class DataUnavailable(RuntimeError):
    """Raised when the processed dataset or the registry is missing."""


class UnknownCrop(KeyError):
    """Raised when a request names a crop the registry does not hold."""


class UnknownMarket(KeyError):
    """Raised when a request names a market the crop's data does not cover."""


@dataclass
class _Cached:
    """A value plus the file signature it was computed from."""

    signature: tuple
    value: Any


class PriceService:
    """Loads data and models lazily, and recomputes when files on disk change."""

    def __init__(self, data_config_path: str | Path = "configs/data.yaml") -> None:
        """Set up an empty service.

        Args:
            data_config_path: Path to the data config.
        """
        self._data_config_path = data_config_path
        self._lock = threading.Lock()
        self._weekly: _Cached | None = None
        self._models: dict[str, _Cached] = {}
        self._cards: dict[str, _Cached] = {}
        self._index: _Cached | None = None

    # --- plumbing -------------------------------------------------------------

    @staticmethod
    def _signature(path: Path) -> tuple:
        """Return a cheap change signature for a file.

        Args:
            path: The file to stamp.

        Returns:
            A tuple of (exists, mtime, size).
        """
        try:
            stat = path.stat()
            return (True, stat.st_mtime_ns, stat.st_size)
        except OSError:
            return (False, 0, 0)

    @property
    def data_config(self) -> dict[str, Any]:
        """The parsed data config."""
        return load_config(self._data_config_path)

    def _weekly_path(self) -> Path:
        """Resolved path of the weekly price table the service should read.

        The processed parquet is preferred when present, which is the case in
        development. A deployed container instead carries ``models/weekly_snapshot.csv``
        and no parquet engine, so that is the fallback.

        Returns:
            The path to read, whether or not it exists.
        """
        parquet = resolve_path(self.data_config["paths"]["weekly_parquet"])
        if parquet.is_file():
            return parquet
        snapshot = resolve_path("models/weekly_snapshot.csv")
        return snapshot if snapshot.is_file() else parquet

    def weekly(self) -> pd.DataFrame:
        """Return the processed weekly price table, reloading it if the file changed.

        Returns:
            The long weekly frame.

        Raises:
            DataUnavailable: If Phase 1 has not been run.
        """
        path = self._weekly_path()
        signature = self._signature(path)
        if not signature[0]:
            raise DataUnavailable(
                f"{path} not found. Build it with: uv run python -m automl.data.aggregate"
            )
        with self._lock:
            if self._weekly is None or self._weekly.signature != signature:
                if path.suffix == ".csv":
                    frame = pd.read_csv(path, parse_dates=["week_start"])
                else:
                    frame = pd.read_parquet(path)
                if "crop" not in frame.columns:
                    frame["crop"] = str(self.data_config["crops"][0]["name"])
                self._weekly = _Cached(signature, frame)
                LOGGER.info("loaded %d weekly rows from %s", len(frame), path.name)
            return self._weekly.value

    def index(self) -> dict[str, Any]:
        """Return the registry index, reloading it if the file changed.

        Returns:
            The index contents.

        Raises:
            DataUnavailable: If no model has been registered.
        """
        path = resolve_path("models/index.json")
        signature = self._signature(path)
        if not signature[0]:
            raise DataUnavailable(
                "models/index.json not found. Register models with: "
                "uv run python -m automl.registry"
            )
        with self._lock:
            if self._index is None or self._index.signature != signature:
                self._index = _Cached(signature, load_index(path))
            return self._index.value

    def _slug(self, crop: str) -> str:
        """Resolve a crop name to its registered slug.

        Args:
            crop: Crop name or slug, matched case-insensitively.

        Returns:
            The slug.

        Raises:
            UnknownCrop: If the registry holds no such crop.
        """
        for entry in self.index()["crops"]:
            if crop.lower() in {str(entry["crop"]).lower(), str(entry["slug"]).lower()}:
                return str(entry["slug"])
        known = [entry["crop"] for entry in self.index()["crops"]]
        raise UnknownCrop(f"unknown crop {crop!r}; registered crops are {known}")

    def crop_name(self, crop: str) -> str:
        """Resolve a crop name or slug to its display name.

        Args:
            crop: Crop name or slug.

        Returns:
            The display name.
        """
        slug = self._slug(crop)
        for entry in self.index()["crops"]:
            if entry["slug"] == slug:
                return str(entry["crop"])
        return slug

    def model(self, crop: str) -> Any:
        """Return one crop's fitted model, reloading if the pickle changed.

        Args:
            crop: Crop name or slug.

        Returns:
            The fitted estimator.
        """
        slug = self._slug(crop)
        path = model_path(slug)
        signature = self._signature(path)
        with self._lock:
            cached = self._models.get(slug)
            if cached is None or cached.signature != signature:
                self._models[slug] = _Cached(signature, load_model(path))
                LOGGER.info("loaded model for %s", slug)
            return self._models[slug].value

    def card(self, crop: str) -> dict[str, Any]:
        """Return one crop's model card, reloading if the file changed.

        Args:
            crop: Crop name or slug.

        Returns:
            The card contents.
        """
        slug = self._slug(crop)
        path = card_path(slug)
        signature = self._signature(path)
        with self._lock:
            cached = self._cards.get(slug)
            if cached is None or cached.signature != signature:
                self._cards[slug] = _Cached(signature, load_card(path))
            return self._cards[slug].value

    # --- derived views --------------------------------------------------------

    def crop_frame(self, crop: str) -> pd.DataFrame:
        """Return the weekly rows for one crop.

        Args:
            crop: Crop name or slug.

        Returns:
            The crop's weekly rows, sorted.
        """
        name = self.crop_name(crop)
        frame = self.weekly()
        block = frame[frame["crop"].astype(str).str.lower() == name.lower()]
        return block.sort_values(["market", "week_start"], ignore_index=True)

    def markets(self, crop: str) -> list[str]:
        """List the markets available for one crop.

        Args:
            crop: Crop name or slug.

        Returns:
            Sorted market names.
        """
        return sorted(str(name) for name in self.crop_frame(crop)["market"].unique())

    def meta(self) -> dict[str, Any]:
        """Describe everything the UI needs to populate its controls.

        Returns:
            Crops with their markets, model versions and measured test error, plus the
            overall data range and the provenance of the dataset.
        """
        frame = self.weekly()
        observed = frame.dropna(subset=["modal_price"])
        crops = []
        for entry in self.index()["crops"]:
            name = str(entry["crop"])
            try:
                markets = self.markets(name)
            except Exception:  # noqa: BLE001 - a crop with no data is simply skipped
                markets = []
            crops.append(
                {
                    "crop": name,
                    "slug": entry["slug"],
                    "markets": markets,
                    "model_version": entry.get("version"),
                    "test_mae": entry.get("test_mae"),
                    "horizon_weeks": entry.get("horizon_weeks", 1),
                }
            )
        return {
            "crops": [c for c in crops if c["markets"]],
            "data_start": str(observed["week_start"].min().date()),
            "data_end": str(observed["week_start"].max().date()),
            "n_weekly_rows": int(len(frame)),
            "price_unit": "INR per quintal",
            "entity_level": str(self.data_config["source"]["entity_level"]),
            "source": "CEDA Agmarknet portal, Ashoka University (data: Agmarknet, GoI)",
            "registry_updated_utc": self.index().get("updated_utc"),
            "served_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    # --- the live computations ------------------------------------------------

    def predict(self, crop: str, market: str) -> dict[str, Any]:
        """Forecast the next unobserved week for one crop in one market.

        The feature row is built here, at request time, from the dataset on disk --
        not read from a stored answer.

        Args:
            crop: Crop name or slug.
            market: Market name.

        Returns:
            The forecast with the context it was computed in.

        Raises:
            UnknownMarket: If the crop has no data for that market.
        """
        name = self.crop_name(crop)
        frame = self.crop_frame(crop)
        if market not in set(frame["market"].astype(str)):
            raise UnknownMarket(
                f"no {name} data for market {market!r}; available: {self.markets(crop)}"
            )

        features_config = load_config("configs/features.yaml")
        rows = build_forecast_rows(frame.reset_index(drop=True), features_config)
        row = rows[rows["market"].astype(str) == market]
        if row.empty:
            raise UnknownMarket(f"could not build a forecast row for {market!r}")

        columns = feature_columns(row)
        raw = self.model(crop).predict(row[columns])
        predicted = float(to_price_level(row, raw)[0])
        anchor = float(row["lag_1"].iloc[0])
        forecast_week = pd.Timestamp(row["week_start"].iloc[0])

        return {
            "crop": name,
            "market": market,
            "predicted_price": round(predicted, 2),
            "last_observed_price": round(anchor, 2),
            "change": round(predicted - anchor, 2),
            "change_pct": round((predicted - anchor) / anchor * 100.0, 2) if anchor else None,
            "for_week_starting": str(forecast_week.date()),
            "based_on_week_starting": str((forecast_week - pd.Timedelta(weeks=1)).date()),
            "target_mode": target_mode_of(row),
            "model_version": self.card(crop).get("version"),
            "computed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def history(self, crop: str, market: str, weeks: int = 104) -> dict[str, Any]:
        """Return the real observed weekly prices for one series.

        Args:
            crop: Crop name or slug.
            market: Market name.
            weeks: How many recent weeks to return.

        Returns:
            Week/price pairs, oldest first, with missing weeks omitted.

        Raises:
            UnknownMarket: If the crop has no data for that market.
        """
        frame = self.crop_frame(crop)
        block = frame[frame["market"].astype(str) == market].dropna(subset=["modal_price"])
        if block.empty:
            raise UnknownMarket(f"no data for {market!r}")
        block = block.sort_values("week_start").tail(max(1, weeks))
        return {
            "crop": self.crop_name(crop),
            "market": market,
            "points": [
                {"week": str(pd.Timestamp(w).date()), "price": round(float(p), 2)}
                for w, p in zip(block["week_start"], block["modal_price"], strict=True)
            ],
        }

    def backtest(self, crop: str, market: str, weeks: int = 26) -> dict[str, Any]:
        """Re-run the registered model across recent weeks and report real errors.

        This is the honest answer to "is the model any good?". The model is not
        refitted; it predicts each of the last ``weeks`` weeks from features built
        only from earlier data, and the result is compared against what the price
        actually turned out to be. The naive "same as last week" forecast is scored on
        exactly the same rows so the comparison is fair.

        Args:
            crop: Crop name or slug.
            market: Market name.
            weeks: How many recent weeks to score.

        Returns:
            Per-week actual and predicted prices, plus summary metrics for the model
            and for the naive baseline.

        Raises:
            UnknownMarket: If the crop has no usable rows for that market.
        """
        frame = self.crop_frame(crop)
        features_config = load_config("configs/features.yaml")
        built = build_features(frame.reset_index(drop=True), features_config)
        block = built[built["market"].astype(str) == market]
        if block.empty:
            raise UnknownMarket(f"no usable rows for {market!r}")
        block = block.sort_values("week_start").tail(max(2, weeks)).reset_index(drop=True)

        x_block, _ = split_xy(block)
        raw = self.model(crop).predict(x_block)
        predicted = to_price_level(block, raw)
        actual = block[TARGET_COLUMN].to_numpy(dtype=float)
        naive = block["lag_1"].to_numpy(dtype=float)

        usable = np.isfinite(naive)
        model_metrics = all_metrics(actual, predicted)
        naive_metrics = all_metrics(actual[usable], naive[usable])

        return {
            "crop": self.crop_name(crop),
            "market": market,
            "weeks_scored": int(len(block)),
            "model": {key: round(value, 2) for key, value in model_metrics.items()},
            "naive_last_week": {key: round(value, 2) for key, value in naive_metrics.items()},
            "model_beats_naive": bool(model_metrics["mae"] < naive_metrics["mae"]),
            "points": [
                {
                    "week": str(pd.Timestamp(week).date()),
                    "actual": round(float(a), 2),
                    "predicted": round(float(p), 2),
                    "naive": round(float(n), 2) if np.isfinite(n) else None,
                }
                for week, a, p, n in zip(block["week_start"], actual, predicted, naive, strict=True)
            ],
            "computed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }


SERVICE = PriceService()
