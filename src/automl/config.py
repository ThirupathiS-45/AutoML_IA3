"""Load YAML configuration files and resolve repository-relative paths."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a dictionary.

    Args:
        path: Path to the YAML file, absolute or relative to the repository root.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"config file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle) or {}
    LOGGER.debug("loaded config %s", resolved)
    return config


def resolve_path(path: str | Path) -> Path:
    """Turn a possibly repository-relative path into an absolute one.

    Args:
        path: An absolute path, or one relative to the repository root.

    Returns:
        The absolute path. Nothing is created on disk.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def ensure_parent(path: str | Path) -> Path:
    """Resolve a path and create its parent directory if needed.

    Args:
        path: The file path whose parent directory should exist.

    Returns:
        The resolved absolute path.
    """
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, with a consistent format.

    Args:
        level: Logging level for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
