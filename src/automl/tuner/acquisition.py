"""Expected Improvement: how the tuner decides which configuration to try next.

The surrogate tells us, for any untried configuration, a predicted error ``mu`` and an
uncertainty ``sigma``. Expected Improvement asks a single question: *on average, how
many rupees of error would we save against the best result so far if we tried this
point?* Points win either by looking good (low ``mu``) or by being unexplored (high
``sigma``), which is how one formula balances exploiting and exploring.

Written out, with ``f_best`` the lowest error seen so far and ``xi`` a small constant
that nudges the search towards exploration::

    improvement = f_best - mu - xi
    z           = improvement / sigma
    EI          = improvement * Phi(z) + sigma * phi(z)

``Phi`` is the standard normal cumulative distribution (the probability of landing
below a value) and ``phi`` is its density. The first term is the size of the
improvement weighted by how likely it is; the second rewards uncertainty on its own.
Expected Improvement is never negative, because a configuration that turns out worse
than the current best simply gets discarded rather than counted against us.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best_score: float, xi: float = 0.01
) -> np.ndarray:
    """Compute Expected Improvement for a minimisation problem.

    Args:
        mean: Surrogate mean prediction at each candidate point.
        std: Surrogate standard deviation at each candidate point.
        best_score: Lowest validation error observed so far.
        xi: Exploration margin in score units. Larger values push the search away
            from the current best.

    Returns:
        A non-negative array of the same shape as ``mean``. Points with zero
        uncertainty score exactly zero, because there is nothing left to learn there.
    """
    mu = np.asarray(mean, dtype=float).ravel()
    sigma = np.asarray(std, dtype=float).ravel()

    improvement = best_score - mu - xi
    result = np.zeros_like(mu)

    usable = sigma > 1e-12
    z = np.zeros_like(mu)
    z[usable] = improvement[usable] / sigma[usable]
    result[usable] = improvement[usable] * norm.cdf(z[usable]) + sigma[usable] * norm.pdf(z[usable])

    return np.maximum(result, 0.0)
