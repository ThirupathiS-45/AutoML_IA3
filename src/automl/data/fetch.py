"""Download daily crop price series from the CEDA Agmarknet portal into ``data/raw``.

One CSV is written per (crop, state) pair. Requests are issued a few at a time rather
than one by one, because the portal takes roughly forty seconds per year of daily data
and a full crop-by-state sweep would otherwise take hours. Concurrency is kept
deliberately low (``source.max_workers``) out of courtesy to a free public endpoint.

A ``provenance.json`` records exactly what was requested and when, so the raw layer can
be audited later. Files already on disk are skipped, which makes the download
resumable: interrupt it and run it again.

Run with::

    uv run python -m automl.data.fetch --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from automl.config import ensure_parent, load_config, resolve_path, setup_logging

LOGGER = logging.getLogger(__name__)

_LOCAL = threading.local()


def _session() -> requests.Session:
    """Return a per-thread HTTP session.

    ``requests.Session`` is not documented as thread-safe, so each worker thread keeps
    its own while still reusing connections.

    Returns:
        A session owned by the calling thread.
    """
    existing = getattr(_LOCAL, "session", None)
    if existing is None:
        existing = requests.Session()
        existing.headers.update({"Content-Type": "application/json"})
        _LOCAL.session = existing
    return existing


def _year_chunks(start: date, end: date, chunk_years: int) -> list[tuple[date, date]]:
    """Split a date range into chunks of at most ``chunk_years`` calendar years.

    Args:
        start: First date to include.
        end: Last date to include.
        chunk_years: Maximum number of calendar years per chunk.

    Returns:
        A list of inclusive ``(chunk_start, chunk_end)`` pairs covering the range.
    """
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        stop_year = cursor.year + chunk_years - 1
        chunk_end = min(date(stop_year, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = date(stop_year + 1, 1, 1)
    return chunks


def fetch_chunk(
    source: dict[str, Any],
    commodity_id: int,
    market_id: int,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """Fetch one (crop, state, date-range) chunk of daily prices, retrying on failure.

    Args:
        source: The ``source`` block of the data config.
        commodity_id: CEDA commodity id.
        market_id: Census state id of the market entity.
        start: First date of the chunk.
        end: Last date of the chunk.

    Returns:
        The list of raw records returned by the portal, possibly empty.

    Raises:
        RuntimeError: If every retry attempt failed.
    """
    url = f"{source['base_url']}{source['prices_path']}"
    payload = {
        "state_id": market_id,
        "commodity_id": commodity_id,
        "district_id": None,
        "calculation_type": source["calculation_type"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    last_error: Exception | None = None
    for attempt in range(1, int(source["retries"]) + 1):
        try:
            response = _session().post(
                url, json=payload, timeout=float(source["request_timeout_s"])
            )
            response.raise_for_status()
            body = response.json()
            if "data" not in body:
                raise RuntimeError(f"unexpected response body: {str(body)[:200]}")
            return list(body["data"])
        except Exception as error:  # noqa: BLE001 - retried and re-raised below
            last_error = error
            LOGGER.warning(
                "attempt %d/%s failed for commodity=%s market=%s %s..%s: %s",
                attempt,
                source["retries"],
                commodity_id,
                market_id,
                start,
                end,
                error,
            )
            time.sleep(float(source["sleep_between_s"]) * attempt)
    raise RuntimeError(
        f"all retries failed for commodity={commodity_id} market={market_id} {start}..{end}"
    ) from last_error


def fetch_series(
    source: dict[str, Any],
    crop: dict[str, Any],
    market: dict[str, Any],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch the full date range for one crop in one state, chunk by chunk.

    Args:
        source: The ``source`` block of the data config.
        crop: Mapping with ``id``, ``name`` and ``slug`` keys.
        market: Mapping with ``id`` and ``name`` keys.
        start: First date to request.
        end: Last date to request.

    Returns:
        A dataframe of raw records with ``market`` and ``crop`` columns added.
    """
    records: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _year_chunks(start, end, int(source["chunk_years"])):
        chunk = fetch_chunk(source, int(crop["id"]), int(market["id"]), chunk_start, chunk_end)
        records.extend(chunk)
        time.sleep(float(source["sleep_between_s"]))
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["market"] = market["name"]
        frame["crop"] = crop["name"]
    return frame


def _raw_path(raw_dir: Path, crop: dict[str, Any], market: dict[str, Any]) -> Path:
    """Build the raw CSV path for one crop and state.

    Args:
        raw_dir: Directory raw CSVs live in.
        crop: Crop mapping.
        market: Market mapping.

    Returns:
        The destination path.
    """
    market_slug = str(market["name"]).lower().replace(" ", "_")
    return raw_dir / f"{crop['slug']}_{market_slug}_daily.csv"


def run(config_path: str | Path) -> Path:
    """Download every (crop, state) series and write provenance alongside them.

    Args:
        config_path: Path to ``configs/data.yaml``.

    Returns:
        The directory the raw CSVs were written to.
    """
    config = load_config(config_path)
    source = config["source"]
    start = datetime.strptime(config["date_range"]["start"], "%Y-%m-%d").date()
    end = datetime.strptime(config["date_range"]["end"], "%Y-%m-%d").date()
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (crop, market) for crop in config["crops"] for market in config["candidate_markets"]
    ]
    pending = [(c, m) for c, m in jobs if not _raw_path(raw_dir, c, m).exists()]
    LOGGER.info(
        "%d series requested, %d already on disk, %d to download",
        len(jobs),
        len(jobs) - len(pending),
        len(pending),
    )

    manifest: list[dict[str, Any]] = []
    done = 0

    def _work(crop: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
        out_path = _raw_path(raw_dir, crop, market)
        frame = fetch_series(source, crop, market, start, end)
        frame.to_csv(out_path, index=False)
        return {
            "crop": crop["name"],
            "crop_id": crop["id"],
            "market": market["name"],
            "market_id": market["id"],
            "file": out_path.name,
            "rows": int(len(frame)),
        }

    with ThreadPoolExecutor(max_workers=int(source.get("max_workers", 3))) as pool:
        futures = {pool.submit(_work, crop, market): (crop, market) for crop, market in pending}
        for future in as_completed(futures):
            crop, market = futures[future]
            done += 1
            try:
                record = future.result()
                manifest.append(record)
                LOGGER.info(
                    "[%d/%d] %s / %s -> %d rows",
                    done,
                    len(pending),
                    record["crop"],
                    record["market"],
                    record["rows"],
                )
            except Exception:
                LOGGER.exception(
                    "[%d/%d] %s / %s failed; it will be retried on the next run",
                    done,
                    len(pending),
                    crop["name"],
                    market["name"],
                )

    for crop, market in jobs:
        path = _raw_path(raw_dir, crop, market)
        if path.exists() and not any(entry["file"] == path.name for entry in manifest):
            manifest.append(
                {
                    "crop": crop["name"],
                    "crop_id": crop["id"],
                    "market": market["name"],
                    "market_id": market["id"],
                    "file": path.name,
                    "cached": True,
                }
            )

    provenance = {
        "source": (
            "CEDA Agmarknet portal, Centre for Economic Data and Analysis, Ashoka University"
        ),
        "portal_url": source["base_url"],
        "endpoint": f"{source['base_url']}{source['prices_path']}",
        "underlying_data": (
            "Agmarknet, Ministry of Agriculture and Farmers Welfare, Government of India"
        ),
        "entity_level": source["entity_level"],
        "crops": config["crops"],
        "markets": config["candidate_markets"],
        "granularity": "daily",
        "price_unit": "INR per quintal",
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": sorted(manifest, key=lambda entry: entry["file"]),
    }
    provenance_path = ensure_parent(config["paths"]["provenance"])
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("provenance written to %s", provenance_path)
    return raw_dir


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    run(args.config)


if __name__ == "__main__":
    main()
