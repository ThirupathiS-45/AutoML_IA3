"""A Gaussian Process surrogate: the tuner's cheap stand-in for real model training.

Fitting one LightGBM model on real data takes seconds; we only get 15 of those per
strategy. So after each real trial we fit a much cheaper model that maps a
hyperparameter vector to the validation error it produced, and we ask *that* model
where to look next. The Gaussian Process is the usual choice because it returns not
just a guess but an uncertainty, which is exactly what the acquisition function needs.

The kernel is Matern with ``nu=2.5``: it assumes the error surface is smooth but not
absurdly so. A plain squared-exponential kernel assumes infinite smoothness, which
tends to make the surrogate overconfident between observed points.

Scores are standardised before fitting. A GP with a zero mean prior would otherwise
believe that unexplored regions have an error of zero rupees, and chase them forever.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

LOGGER = logging.getLogger(__name__)


class GaussianProcessSurrogate:
    """Wraps scikit-learn's GP with the normalisation the tuner needs."""

    def __init__(self, n_dimensions: int, seed: int = 0, noise_level: float = 1e-3) -> None:
        """Configure the kernel.

        Args:
            n_dimensions: Width of the hyperparameter vectors.
            seed: Random seed for the kernel's restarts.
            noise_level: Starting value for the learned observation noise. A nonzero
                value keeps the covariance matrix invertible when two trials land on
                nearly the same configuration.
        """
        self.n_dimensions = n_dimensions
        self.seed = seed
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(n_dimensions),
            length_scale_bounds=(1e-2, 1e4),
            nu=2.5,
        ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1e1))
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=seed,
        )
        self._mean = 0.0
        self._scale = 1.0
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called with usable data."""
        return self._fitted

    def fit(self, vectors: np.ndarray, scores: np.ndarray) -> GaussianProcessSurrogate:
        """Fit the surrogate on the trial history.

        Args:
            vectors: An ``(n_trials, n_dimensions)`` array of unit-box vectors.
            scores: The validation error of each trial. Lower is better.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            ValueError: If the shapes disagree with the configured width.
        """
        x = np.atleast_2d(np.asarray(vectors, dtype=float))
        y = np.asarray(scores, dtype=float).ravel()
        if x.shape[1] != self.n_dimensions:
            raise ValueError(f"expected {self.n_dimensions} columns, got {x.shape[1]}")
        if x.shape[0] != y.size:
            raise ValueError(f"{x.shape[0]} vectors but {y.size} scores")
        if y.size < 2:
            LOGGER.debug("not enough trials to fit a surrogate (%d)", y.size)
            self._fitted = False
            return self

        self._mean = float(np.mean(y))
        spread = float(np.std(y))
        self._scale = spread if spread > 1e-12 else 1.0
        with warnings.catch_warnings():
            # A length scale pinned at its bound means the surrogate sees that
            # dimension as nearly flat. That is information, not a failure.
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._model.fit(x, (y - self._mean) / self._scale)
        self._fitted = True
        LOGGER.debug("surrogate fitted on %d trials; kernel=%s", y.size, self._model.kernel_)
        return self

    def predict(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the error and its uncertainty at new points.

        Args:
            vectors: An ``(n_points, n_dimensions)`` array of unit-box vectors.

        Returns:
            A tuple of (mean error, standard deviation), both in the original score
            units. Before the surrogate is fitted, it returns the trivial prediction
            of zero mean and unit spread, which makes every point equally attractive.
        """
        x = np.atleast_2d(np.asarray(vectors, dtype=float))
        if not self._fitted:
            return np.zeros(x.shape[0]), np.ones(x.shape[0])
        mean, std = self._model.predict(x, return_std=True)
        return mean * self._scale + self._mean, np.maximum(std * self._scale, 1e-12)
