"""The ``web-start`` entry point: bring up everything the web app needs.

A single command so there is nothing to remember. It checks the prerequisites in
order, tells you the exact command to run for anything missing, and then starts the
server. Backend and frontend are one process by design: the API and the page it
serves live behind the same port, so there is no second server to start, no CORS
configuration, and nothing that can drift out of sync.

Run with::

    uv run web-start                       # http://127.0.0.1:8000
    uv run web-start --port 9000 --reload   # pick a port, reload on edit
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from automl.config import load_config, resolve_path, setup_logging

LOGGER = logging.getLogger(__name__)

BANNER = r"""
  mandi-automl  .  crop price forecaster
  ---------------------------------------
"""


def preflight() -> tuple[list[str], list[str]]:
    """Check what the app needs to serve, and what it would need to rebuild.

    Serving needs only two things: a registry index and a readable price table. Both
    are committed to the repository, so a fresh clone can start the app without
    downloading anything. The raw CSVs and the processed parquet are needed to
    *rebuild* the dataset, not to serve it, so their absence is a note rather than a
    blocker.

    Returns:
        A tuple of (blockers, notes). A non-empty blockers list means the app cannot
        usefully serve.
    """
    blockers: list[str] = []
    notes: list[str] = []

    config = load_config("configs/data.yaml")
    root = resolve_path(".")
    weekly = resolve_path(config["paths"]["weekly_parquet"])
    snapshot = resolve_path("models/weekly_snapshot.csv")
    raw_dir = resolve_path(config["paths"]["raw_dir"])
    index = resolve_path("models/index.json")

    if not index.is_file():
        blockers.append(
            "No registered models (models/index.json is missing).\n"
            "      Fix: uv run python -m automl.registry"
        )
    if not weekly.is_file() and not snapshot.is_file():
        blockers.append(
            "No price table to serve from.\n"
            "      Fix: uv run python -m automl.data.fetch\n"
            "       and: uv run python -m automl.data.aggregate\n"
            "       and: uv run python -m automl.registry"
        )

    if not weekly.is_file() and snapshot.is_file():
        notes.append(
            f"serving from {snapshot.relative_to(root)} "
            "(the committed snapshot; no local dataset build found)"
        )
    if not any(raw_dir.glob("*_daily.csv")):
        notes.append(
            "no raw CSVs in data/raw -- only needed to refresh the dataset "
            "(uv run python -m automl.data.fetch)"
        )
    return blockers, notes


def main() -> None:
    """Command-line entry point for ``web-start``."""
    parser = argparse.ArgumentParser(
        prog="web-start", description="Start the mandi-automl web app (API + UI)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on source edits")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="start even if data or models are missing (the app will report it)",
    )
    args = parser.parse_args()

    setup_logging()
    print(BANNER)

    blockers, notes = preflight()
    if blockers:
        print("  Not ready yet:\n")
        for blocker in blockers:
            print(f"   !  {blocker}\n")
        if not args.skip_checks:
            print("  Run the fixes above, or pass --skip-checks to start anyway.")
            sys.exit(1)
        print("  --skip-checks given; starting anyway.\n")
    else:
        print("  models and price data: ready")
    for note in notes:
        print(f"  note: {note}")

    url = f"http://{args.host}:{args.port}"
    print(f"\n  UI   {url}")
    print(f"  API  {url}/docs")
    print(f"  ping {url}/api/health\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
