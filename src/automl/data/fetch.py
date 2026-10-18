"""Download daily onion price series from the CEDA Agmarknet portal into ``data/raw``.

The portal's public JSON endpoint is queried once per (market, year) chunk and each
response is written verbatim as a CSV. A ``provenance.json`` records exactly what was
requested and when, so the raw layer can be audited later.

Run with::

    uv run python -m automl.data.fetch --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from automl.config import ensure_parent, load_config, resolve_path, setup_logging

LOGGER = logging.getLogger(__name__)


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
    session: requests.Session,
    source: dict[str, Any],
    market_id: int,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """Fetch one (market, date-range) chunk of daily prices, retrying on failure.

    Args:
        session: Reusable HTTP session.
        source: The ``source`` block of the data config.
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
        "commodity_id": source["commodity_id"],
        "district_id": None,
        "calculation_type": source["calculation_type"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    last_error: Exception | None = None
    for attempt in range(1, int(source["retries"]) + 1):
        try:
            response = session.post(url, json=payload, timeout=float(source["request_timeout_s"]))
            response.raise_for_status()
            body = response.json()
            if "data" not in body:
                raise RuntimeError(f"unexpected response body: {str(body)[:200]}")
            return list(body["data"])
        except Exception as error:  # noqa: BLE001 - retried and re-raised below
            last_error = error
            LOGGER.warning(
                "attempt %d/%s failed for market=%s %s..%s: %s",
                attempt,
                source["retries"],
                market_id,
                start,
                end,
                error,
            )
            time.sleep(float(source["sleep_between_s"]) * attempt)
    raise RuntimeError(f"all retries failed for market={market_id} {start}..{end}") from last_error


def fetch_market(
    session: requests.Session,
    source: dict[str, Any],
    market: dict[str, Any],
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch the full date range for a single market, chunk by chunk.

    Args:
        session: Reusable HTTP session.
        source: The ``source`` block of the data config.
        market: Mapping with ``id`` and ``name`` keys.
        start: First date to request.
        end: Last date to request.

    Returns:
        A dataframe of raw records with a ``market`` column added.
    """
    records: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _year_chunks(start, end, int(source["chunk_years"])):
        chunk = fetch_chunk(session, source, int(market["id"]), chunk_start, chunk_end)
        LOGGER.info(
            "market=%s %s..%s -> %d records", market["name"], chunk_start, chunk_end, len(chunk)
        )
        records.extend(chunk)
        time.sleep(float(source["sleep_between_s"]))
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["market"] = market["name"]
    return frame


def run(config_path: str | Path) -> Path:
    """Download every candidate market and write one CSV per market plus provenance.

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

    manifest: list[dict[str, Any]] = []
    with requests.Session() as session:
        session.headers.update({"Content-Type": "application/json"})
        for market in config["candidate_markets"]:
            slug = str(market["name"]).lower().replace(" ", "_")
            out_path = raw_dir / f"onion_{slug}_daily.csv"
            if out_path.exists():
                LOGGER.info("skipping %s, already downloaded", out_path.name)
                manifest.append({"market": market["name"], "file": out_path.name, "cached": True})
                continue
            frame = fetch_market(session, source, market, start, end)
            frame.to_csv(out_path, index=False)
            LOGGER.info("wrote %s (%d rows)", out_path.name, len(frame))
            manifest.append(
                {
                    "market": market["name"],
                    "market_id": market["id"],
                    "file": out_path.name,
                    "rows": int(len(frame)),
                    "cached": False,
                }
            )

    provenance = {
        "source": (
            "CEDA Agmarknet portal, Centre for Economic Data and Analysis, Ashoka University"
        ),
        "endpoint": f"{source['base_url']}{source['prices_path']}",
        "underlying_data": (
            "Agmarknet, Ministry of Agriculture and Farmers Welfare, Government of India"
        ),
        "entity_level": source["entity_level"],
        "commodity": config["crop"],
        "commodity_id": source["commodity_id"],
        "granularity": "daily",
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "downloaded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": manifest,
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
