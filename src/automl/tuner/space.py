"""Convert hyperparameter dictionaries to and from numeric vectors.

A Gaussian Process cannot read ``{"learning_rate": 0.05, "num_leaves": 31}``. It needs
a point in a numeric box. This module is the translator, and it has one firm rule:
**every dimension is normalised to the unit interval [0, 1]**, so no single
hyperparameter dominates the kernel's distance calculation just because its raw
numbers happen to be large.

Two details matter:

- *Log scale.* Learning rates are searched on a log axis, because the step from 0.005
  to 0.01 matters as much as the step from 0.1 to 0.2. Encoding takes the logarithm
  before normalising; decoding exponentiates afterwards.
- *Integers.* Parameters such as ``num_leaves`` are treated as continuous during the
  search and rounded only when a real model is built. That keeps the surrogate smooth
  while still producing valid LightGBM arguments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dimension:
    """One tunable hyperparameter.

    Attributes:
        name: Hyperparameter name as LightGBM expects it.
        kind: Either ``float`` or ``int``.
        low: Inclusive lower bound in natural units.
        high: Inclusive upper bound in natural units.
        log: Whether to search on a logarithmic axis.
    """

    name: str
    kind: str
    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        """Validate the bounds.

        Raises:
            ValueError: If the bounds are inverted, or a log axis touches zero.
        """
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high ({self.high}) must exceed low ({self.low})")
        if self.log and self.low <= 0:
            raise ValueError(f"{self.name}: log axis needs a strictly positive lower bound")
        if self.kind not in {"float", "int"}:
            raise ValueError(f"{self.name}: unknown type {self.kind!r}")

    def encode(self, value: float) -> float:
        """Map a natural-unit value onto [0, 1].

        Args:
            value: The hyperparameter value in natural units.

        Returns:
            The normalised coordinate, clipped to [0, 1].
        """
        if self.log:
            position = (np.log(value) - np.log(self.low)) / (np.log(self.high) - np.log(self.low))
        else:
            position = (value - self.low) / (self.high - self.low)
        return float(np.clip(position, 0.0, 1.0))

    def decode(self, position: float) -> float | int:
        """Map a [0, 1] coordinate back to natural units.

        Args:
            position: The normalised coordinate.

        Returns:
            The hyperparameter value, rounded to an ``int`` for integer dimensions.
        """
        clipped = float(np.clip(position, 0.0, 1.0))
        if self.log:
            value = float(
                np.exp(np.log(self.low) + clipped * (np.log(self.high) - np.log(self.low)))
            )
        else:
            value = float(self.low + clipped * (self.high - self.low))
        if self.kind == "int":
            return int(round(value))
        return value


class SearchSpace:
    """An ordered collection of dimensions, with vector encode and decode."""

    def __init__(self, dimensions: list[Dimension]) -> None:
        """Store the dimensions in a fixed order.

        Args:
            dimensions: The tunable hyperparameters.

        Raises:
            ValueError: If no dimensions are supplied.
        """
        if not dimensions:
            raise ValueError("a search space needs at least one dimension")
        self.dimensions = dimensions

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SearchSpace:
        """Build a space from the parsed ``configs/search_space.yaml``.

        Args:
            config: Mapping with a ``hyperparameters`` block.

        Returns:
            The configured search space, with dimensions in sorted name order so
            the vector layout is stable across runs.
        """
        block = config["hyperparameters"]
        dimensions = [
            Dimension(
                name=name,
                kind=str(spec["type"]),
                low=float(spec["low"]),
                high=float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
            for name, spec in sorted(block.items())
        ]
        LOGGER.info(
            "search space: %d dimensions (%s)",
            len(dimensions),
            ", ".join(d.name for d in dimensions),
        )
        return cls(dimensions)

    @property
    def names(self) -> list[str]:
        """Dimension names in vector order."""
        return [dimension.name for dimension in self.dimensions]

    def __len__(self) -> int:
        """Number of dimensions."""
        return len(self.dimensions)

    def to_vector(self, params: dict[str, Any]) -> np.ndarray:
        """Encode a hyperparameter dictionary as a unit-box vector.

        Args:
            params: Hyperparameters in natural units. Extra keys are ignored.

        Returns:
            A 1-D array of length ``len(self)`` with values in [0, 1].

        Raises:
            KeyError: If a dimension is missing from ``params``.
        """
        missing = [d.name for d in self.dimensions if d.name not in params]
        if missing:
            raise KeyError(f"params missing dimensions: {missing}")
        return np.array(
            [dimension.encode(float(params[dimension.name])) for dimension in self.dimensions],
            dtype=float,
        )

    def from_vector(self, vector: np.ndarray) -> dict[str, float | int]:
        """Decode a unit-box vector back into hyperparameters.

        Args:
            vector: A 1-D array of length ``len(self)``.

        Returns:
            Hyperparameters in natural units, ready for LightGBM.

        Raises:
            ValueError: If the vector has the wrong length.
        """
        array = np.asarray(vector, dtype=float).ravel()
        if array.size != len(self):
            raise ValueError(f"expected a vector of length {len(self)}, got {array.size}")
        return {
            dimension.name: dimension.decode(array[index])
            for index, dimension in enumerate(self.dimensions)
        }

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        """Draw uniform random points from the unit box.

        Args:
            rng: Seeded random generator.
            n: How many points to draw.

        Returns:
            An ``(n, len(self))`` array of candidate vectors.
        """
        return rng.random((n, len(self)))
