"""Persist tuning trials to JSON so a search can be resumed or warm-started.

Every evaluated configuration is recorded with the vector that produced it, the score
it earned, and which window and seed it belonged to. Two things depend on this file:

- **Resume.** A search interrupted after 8 of 15 trials reloads its history and carries
  on, rather than starting over.
- **Warm start.** When the market drifts and a new window opens, the trials from
  earlier windows are loaded into the surrogate *before* the first new trial, so the
  search begins already knowing which regions were hopeless last time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from automl.config import ensure_parent, resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass
class Trial:
    """One evaluated hyperparameter configuration.

    Attributes:
        trial_id: Zero-based index within its own search.
        vector: The unit-box encoding of ``params``.
        params: Hyperparameters in natural units.
        score: Validation error. Lower is better.
        window_id: Which evaluation window the trial belongs to.
        seed: The seed the search was run with.
        source: How the point was chosen, ``random``, ``bo`` or ``warm``.
        timestamp: UTC ISO-8601 time the trial finished.
        extra: Any additional metrics worth keeping.
    """

    trial_id: int
    vector: list[float]
    params: dict[str, Any]
    score: float
    window_id: int
    seed: int
    source: str = "bo"
    timestamp: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Stamp the trial with the current time if none was supplied."""
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat(timespec="seconds")


class TrialHistory:
    """An ordered collection of trials, with JSON persistence."""

    def __init__(self, trials: list[Trial] | None = None) -> None:
        """Create a history.

        Args:
            trials: Existing trials to start from. Defaults to empty.
        """
        self.trials: list[Trial] = list(trials or [])

    def __len__(self) -> int:
        """Number of trials recorded."""
        return len(self.trials)

    def add(self, trial: Trial) -> None:
        """Append a trial.

        Args:
            trial: The finished trial to record.
        """
        self.trials.append(trial)

    def vectors(self) -> np.ndarray:
        """Stack every trial vector.

        Returns:
            An ``(n_trials, n_dimensions)`` array, empty if there are no trials.
        """
        if not self.trials:
            return np.empty((0, 0))
        return np.array([trial.vector for trial in self.trials], dtype=float)

    def scores(self) -> np.ndarray:
        """Collect every trial score.

        Returns:
            A 1-D array of validation errors.
        """
        return np.array([trial.score for trial in self.trials], dtype=float)

    def best(self) -> Trial | None:
        """Return the trial with the lowest score.

        Returns:
            The best trial, or ``None`` if the history is empty. Finite scores are
            preferred over failed trials recorded as infinity.
        """
        finite = [trial for trial in self.trials if np.isfinite(trial.score)]
        pool = finite or self.trials
        return min(pool, key=lambda trial: trial.score) if pool else None

    def best_score(self) -> float:
        """Lowest score in the history.

        Returns:
            The best score, or ``inf`` if the history is empty.
        """
        best = self.best()
        return float(best.score) if best is not None else float("inf")

    def top_k(self, k: int) -> list[Trial]:
        """Return the ``k`` best trials, best first.

        Args:
            k: How many trials to return.

        Returns:
            Up to ``k`` trials sorted by ascending score, skipping non-finite ones.
        """
        finite = [trial for trial in self.trials if np.isfinite(trial.score)]
        return sorted(finite, key=lambda trial: trial.score)[: max(0, k)]

    def for_windows(self, window_ids: list[int]) -> TrialHistory:
        """Filter to trials belonging to the given windows.

        Args:
            window_ids: Windows to keep.

        Returns:
            A new history containing only the matching trials.
        """
        wanted = set(window_ids)
        return TrialHistory([t for t in self.trials if t.window_id in wanted])

    def save(self, path: str | Path, space_names: list[str] | None = None) -> Path:
        """Write the history to JSON.

        Args:
            path: Destination file. Parent directories are created.
            space_names: Dimension names in vector order, stored for auditability.

        Returns:
            The path written to.
        """
        resolved = ensure_parent(path)
        payload = {
            "space_names": space_names or [],
            "n_trials": len(self.trials),
            "saved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "trials": [asdict(trial) for trial in self.trials],
        }
        resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("saved %d trials to %s", len(self.trials), resolved)
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> TrialHistory:
        """Read a history back from JSON.

        Args:
            path: A file previously written by :meth:`save`.

        Returns:
            The reconstructed history. A missing file yields an empty history, so a
            first run and a resumed run take the same code path.
        """
        resolved = resolve_path(path)
        if not resolved.is_file():
            LOGGER.info("no history at %s; starting empty", resolved)
            return cls()
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        trials = [Trial(**record) for record in payload.get("trials", [])]
        LOGGER.info("loaded %d trials from %s", len(trials), resolved)
        return cls(trials)
