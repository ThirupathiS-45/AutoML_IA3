"""The Bayesian optimization search loop, in cold-start and warm-start modes.

The loop is deliberately plain, because the point of the project is to compare how it
is *started*, not to invent a cleverer optimiser:

1. Evaluate a few configurations drawn at random, to give the surrogate something to
   learn from.
2. Fit the Gaussian Process on every trial so far.
3. Score a large pool of random candidate vectors with Expected Improvement and keep
   the single best one.
4. Evaluate that configuration for real, record it, and go back to step 2.
5. Stop when the trial budget is spent.

Three start-up modes share that loop:

``cold``
    Nothing is known. Step 1 runs in full.

``warm_all``
    Every past trial is loaded into the surrogate before the first new evaluation.
    The random start-up is skipped entirely, so all of the budget goes to informed
    proposals. Past trials do **not** count against the budget, and their scores are
    not eligible to be reported as this window's result -- they were measured on a
    different window's data.

``warm_topk``
    Only the ``k`` best past configurations are reused, and they are re-evaluated on
    the new window's data as the initial design. This costs ``k`` trials from the
    budget but every score is measured on current data.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from automl.tuner.acquisition import expected_improvement
from automl.tuner.history import Trial, TrialHistory
from automl.tuner.space import SearchSpace
from automl.tuner.surrogate import GaussianProcessSurrogate

LOGGER = logging.getLogger(__name__)

#: Signature of the function that trains a model and returns its validation error.
ObjectiveFn = Callable[[dict[str, Any]], float]

COLD = "cold"
WARM_ALL = "warm_all"
WARM_TOPK = "warm_topk"
VALID_MODES = (COLD, WARM_ALL, WARM_TOPK)


class BOTuner:
    """A Gaussian-Process Bayesian optimizer over a :class:`SearchSpace`."""

    def __init__(
        self,
        space: SearchSpace,
        objective: ObjectiveFn,
        seed: int = 0,
        n_initial_random: int = 5,
        n_candidates: int = 2048,
        xi: float = 0.01,
    ) -> None:
        """Configure the search.

        Args:
            space: The hyperparameter space to search.
            objective: Callable taking natural-unit hyperparameters and returning the
                validation error to minimise.
            seed: Random seed controlling the initial design and the candidate pool.
            n_initial_random: Random trials before the surrogate is first trusted.
            n_candidates: Size of the random pool that Expected Improvement ranks each
                iteration. This replaces a gradient-based maximisation of the
                acquisition function and is simple enough to be obviously correct.
            xi: Exploration margin passed to Expected Improvement.
        """
        self.space = space
        self.objective = objective
        self.seed = seed
        self.n_initial_random = n_initial_random
        self.n_candidates = n_candidates
        self.xi = xi
        self.history = TrialHistory()
        self.prior = TrialHistory()
        self.trace: list[dict[str, Any]] = []

    def evaluate_vector(self, vector: np.ndarray, window_id: int, source: str) -> Trial:
        """Run the objective on one vector and record the result.

        Args:
            vector: Unit-box hyperparameter vector.
            window_id: Window the trial belongs to.
            source: How the point was chosen.

        Returns:
            The recorded trial. A crashing objective is recorded with an infinite
            score rather than aborting the whole search.
        """
        params = self.space.from_vector(vector)
        started = time.perf_counter()
        try:
            score = float(self.objective(params))
            if not np.isfinite(score):
                score = float("inf")
        except Exception as error:  # noqa: BLE001 - a bad config must not kill the search
            LOGGER.warning("objective failed for %s: %s", params, error)
            score = float("inf")
        elapsed = time.perf_counter() - started

        trial = Trial(
            trial_id=len(self.history),
            vector=[float(value) for value in np.asarray(vector, dtype=float).ravel()],
            params=params,
            score=score,
            window_id=window_id,
            seed=self.seed,
            source=source,
            extra={"seconds": round(elapsed, 4)},
        )
        self.history.add(trial)
        self.trace.append(
            {
                "trial": trial.trial_id,
                "source": source,
                "score": score,
                "best_so_far": self.history.best_score(),
                "seconds": elapsed,
            }
        )
        LOGGER.info(
            "trial %02d [%s] score=%.2f best=%.2f (%.2fs)",
            trial.trial_id,
            source,
            score,
            self.history.best_score(),
            elapsed,
        )
        return trial

    def _fit_surrogate(self) -> GaussianProcessSurrogate:
        """Fit the surrogate on the prior plus the current history.

        Returns:
            A fitted surrogate, or an unfitted one when fewer than two usable trials
            exist.
        """
        surrogate = GaussianProcessSurrogate(len(self.space), seed=self.seed)
        vectors_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        for source in (self.prior, self.history):
            if len(source):
                vectors_list.append(source.vectors())
                scores_list.append(source.scores())
        if not vectors_list:
            return surrogate
        vectors = np.vstack(vectors_list)
        scores = np.concatenate(scores_list)
        finite = np.isfinite(scores)
        if finite.sum() >= 2:
            surrogate.fit(vectors[finite], scores[finite])
        return surrogate

    def _propose(self, surrogate: GaussianProcessSurrogate, rng: np.random.Generator) -> np.ndarray:
        """Pick the next vector by maximising Expected Improvement over a random pool.

        Args:
            surrogate: The fitted surrogate.
            rng: Seeded random generator.

        Returns:
            The winning candidate vector.
        """
        candidates = self.space.sample(rng, self.n_candidates)
        mean, std = surrogate.predict(candidates)
        reference = min(self.history.best_score(), self.prior.best_score())
        if not np.isfinite(reference):
            reference = float(np.min(mean)) if mean.size else 0.0
        scores = expected_improvement(mean, std, reference, xi=self.xi)
        return candidates[int(np.argmax(scores))]

    def run(
        self,
        budget: int,
        window_id: int = 1,
        mode: str = COLD,
        prior: TrialHistory | None = None,
        top_k: int = 5,
        resume_from: TrialHistory | None = None,
    ) -> TrialHistory:
        """Execute the search.

        Args:
            budget: Total number of real objective evaluations allowed.
            window_id: Window the trials belong to.
            mode: One of ``cold``, ``warm_all`` or ``warm_topk``.
            prior: Trials from earlier windows, used by the warm modes.
            top_k: How many past configurations ``warm_topk`` re-evaluates.
            resume_from: Trials already completed for *this* window, which count
                against the budget. Used to continue an interrupted search.

        Returns:
            The history of trials run on this window.

        Raises:
            ValueError: If ``mode`` is unknown, or a warm mode is given no prior.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
        if mode in (WARM_ALL, WARM_TOPK) and (prior is None or len(prior) == 0):
            raise ValueError(f"mode {mode!r} needs a non-empty prior history")

        rng = np.random.default_rng(self.seed)
        self.history = resume_from or TrialHistory()
        self.prior = TrialHistory()
        self.trace = []
        if resume_from is not None and len(resume_from):
            LOGGER.info("resuming with %d completed trials", len(resume_from))

        initial_vectors: list[tuple[np.ndarray, str]] = []
        if mode == WARM_ALL:
            self.prior = prior if prior is not None else TrialHistory()
            LOGGER.info("warm start: %d prior trials seeded into the surrogate", len(self.prior))
        elif mode == WARM_TOPK:
            assert prior is not None
            for trial in prior.top_k(top_k):
                initial_vectors.append((np.asarray(trial.vector, dtype=float), "warm"))
            LOGGER.info("warm start: re-evaluating the top %d prior configs", len(initial_vectors))
        else:
            n_random = max(0, self.n_initial_random - len(self.history))
            for vector in self.space.sample(rng, n_random):
                initial_vectors.append((vector, "random"))

        for vector, source in initial_vectors:
            if len(self.history) >= budget:
                break
            self.evaluate_vector(vector, window_id, source)

        while len(self.history) < budget:
            surrogate = self._fit_surrogate()
            if not surrogate.is_fitted:
                self.evaluate_vector(self.space.sample(rng, 1)[0], window_id, "random")
                continue
            self.evaluate_vector(self._propose(surrogate, rng), window_id, "bo")

        LOGGER.info(
            "search finished: %d trials, best score %.2f",
            len(self.history),
            self.history.best_score(),
        )
        return self.history


def random_search(
    space: SearchSpace,
    objective: ObjectiveFn,
    budget: int,
    seed: int = 0,
    window_id: int = 1,
) -> TrialHistory:
    """Run a pure random search, the control that Bayesian optimization must beat.

    Args:
        space: The hyperparameter space.
        objective: Callable returning the validation error.
        budget: Number of evaluations.
        seed: Random seed.
        window_id: Window the trials belong to.

    Returns:
        The history of trials.
    """
    tuner = BOTuner(space, objective, seed=seed, n_initial_random=budget)
    rng = np.random.default_rng(seed)
    tuner.history = TrialHistory()
    for vector in space.sample(rng, budget):
        tuner.evaluate_vector(vector, window_id, "random")
    return tuner.history
