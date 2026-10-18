"""Tests for the custom Bayesian optimizer: encoding, acquisition, search and resume."""

from __future__ import annotations

import numpy as np
import pytest

from automl.tuner.acquisition import expected_improvement
from automl.tuner.bo_tuner import COLD, WARM_ALL, WARM_TOPK, BOTuner, random_search
from automl.tuner.history import Trial, TrialHistory
from automl.tuner.space import Dimension, SearchSpace
from automl.tuner.surrogate import GaussianProcessSurrogate

SPACE_CONFIG = {
    "hyperparameters": {
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
        "num_leaves": {"type": "int", "low": 7, "high": 127},
        "reg_lambda": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    }
}


@pytest.fixture
def space() -> SearchSpace:
    return SearchSpace.from_config(SPACE_CONFIG)


def quadratic_objective(params: dict) -> float:
    """A cheap deterministic stand-in for training a model."""
    return (
        (np.log10(params["learning_rate"]) + 1.5) ** 2
        + ((params["num_leaves"] - 40) / 40.0) ** 2
        + (params["subsample"] - 0.8) ** 2
    )


# --- space -------------------------------------------------------------------


def test_vector_round_trip_recovers_the_parameters(space):
    original = {"learning_rate": 0.05, "num_leaves": 63, "reg_lambda": 0.5, "subsample": 0.75}
    recovered = space.from_vector(space.to_vector(original))
    assert recovered["learning_rate"] == pytest.approx(original["learning_rate"])
    assert recovered["reg_lambda"] == pytest.approx(original["reg_lambda"])
    assert recovered["subsample"] == pytest.approx(original["subsample"])
    assert recovered["num_leaves"] == original["num_leaves"]


def test_round_trip_is_stable_for_random_vectors(space):
    rng = np.random.default_rng(0)
    for vector in space.sample(rng, 50):
        params = space.from_vector(vector)
        # Decoding is idempotent: re-encoding a decoded point and decoding again
        # must land on exactly the same parameters, integers included.
        assert space.from_vector(space.to_vector(params)) == params


def test_encoding_lands_in_the_unit_box(space):
    rng = np.random.default_rng(1)
    for vector in space.sample(rng, 25):
        encoded = space.to_vector(space.from_vector(vector))
        assert np.all(encoded >= 0.0) and np.all(encoded <= 1.0)


def test_bounds_map_to_zero_and_one(space):
    low = {d.name: d.low for d in space.dimensions}
    high = {d.name: d.high for d in space.dimensions}
    assert np.allclose(space.to_vector(low), 0.0)
    assert np.allclose(space.to_vector(high), 1.0)


def test_log_dimension_is_geometric_at_the_midpoint():
    dimension = Dimension("learning_rate", "float", 0.001, 0.1, log=True)
    assert dimension.decode(0.5) == pytest.approx(np.sqrt(0.001 * 0.1))


def test_integer_dimension_always_decodes_to_int():
    dimension = Dimension("num_leaves", "int", 7, 127)
    for position in np.linspace(0, 1, 30):
        assert isinstance(dimension.decode(float(position)), int)


def test_invalid_dimensions_are_rejected():
    with pytest.raises(ValueError, match="must exceed"):
        Dimension("bad", "float", 1.0, 0.5)
    with pytest.raises(ValueError, match="strictly positive"):
        Dimension("bad", "float", 0.0, 1.0, log=True)
    with pytest.raises(ValueError, match="unknown type"):
        Dimension("bad", "categorical", 0.0, 1.0)


def test_missing_parameter_is_reported(space):
    with pytest.raises(KeyError, match="missing dimensions"):
        space.to_vector({"learning_rate": 0.05})


def test_wrong_length_vector_is_rejected(space):
    with pytest.raises(ValueError, match="length"):
        space.from_vector(np.zeros(2))


# --- acquisition ---------------------------------------------------------------


def test_expected_improvement_is_never_negative():
    rng = np.random.default_rng(2)
    mean = rng.normal(10, 5, 500)
    std = np.abs(rng.normal(2, 1, 500))
    assert np.all(expected_improvement(mean, std, best_score=8.0) >= 0.0)


def test_expected_improvement_keeps_the_input_shape():
    mean = np.zeros(7)
    std = np.ones(7)
    assert expected_improvement(mean, std, 1.0).shape == (7,)


def test_zero_uncertainty_gives_zero_improvement():
    result = expected_improvement(np.array([0.5]), np.array([0.0]), best_score=10.0)
    assert result[0] == 0.0


def test_lower_predicted_error_is_preferred_at_equal_uncertainty():
    std = np.array([1.0, 1.0])
    mean = np.array([5.0, 9.0])
    result = expected_improvement(mean, std, best_score=10.0)
    assert result[0] > result[1]


def test_more_uncertainty_is_preferred_at_equal_prediction():
    mean = np.array([10.0, 10.0])
    std = np.array([0.5, 3.0])
    result = expected_improvement(mean, std, best_score=10.0)
    assert result[1] > result[0]


# --- surrogate -----------------------------------------------------------------


def test_surrogate_predicts_near_an_observed_point():
    rng = np.random.default_rng(3)
    vectors = rng.random((25, 3))
    scores = vectors.sum(axis=1)
    surrogate = GaussianProcessSurrogate(3, seed=0).fit(vectors, scores)
    mean, std = surrogate.predict(vectors[:5])
    assert np.allclose(mean, scores[:5], atol=0.2)
    assert np.all(std >= 0)


def test_unfitted_surrogate_returns_neutral_predictions():
    surrogate = GaussianProcessSurrogate(3, seed=0)
    mean, std = surrogate.predict(np.zeros((4, 3)))
    assert not surrogate.is_fitted
    assert np.allclose(mean, 0.0)
    assert np.allclose(std, 1.0)


def test_surrogate_rejects_mismatched_shapes():
    surrogate = GaussianProcessSurrogate(3, seed=0)
    with pytest.raises(ValueError, match="columns"):
        surrogate.fit(np.zeros((5, 2)), np.zeros(5))
    with pytest.raises(ValueError, match="scores"):
        surrogate.fit(np.zeros((5, 3)), np.zeros(4))


# --- history -------------------------------------------------------------------


def _history(n: int = 6) -> TrialHistory:
    history = TrialHistory()
    for index in range(n):
        history.add(
            Trial(
                trial_id=index,
                vector=[index / 10, 0.5, 0.25, 0.75],
                params={"a": index},
                score=float(10 - index),
                window_id=1,
                seed=0,
            )
        )
    return history


def test_history_round_trips_through_json(tmp_path):
    original = _history()
    path = original.save(tmp_path / "history.json", space_names=["a", "b", "c", "d"])
    reloaded = TrialHistory.load(path)
    assert len(reloaded) == len(original)
    for before, after in zip(original.trials, reloaded.trials, strict=True):
        assert before == after


def test_loading_a_missing_file_gives_an_empty_history(tmp_path):
    assert len(TrialHistory.load(tmp_path / "nope.json")) == 0


def test_best_and_top_k_order_by_ascending_score():
    history = _history()
    assert history.best().score == 5.0
    assert [t.score for t in history.top_k(3)] == [5.0, 6.0, 7.0]


def test_best_ignores_failed_trials():
    history = _history(3)
    history.add(
        Trial(trial_id=3, vector=[0, 0, 0, 0], params={}, score=float("inf"), window_id=1, seed=0)
    )
    assert np.isfinite(history.best().score)


def test_filtering_by_window():
    history = _history(4)
    history.trials[0].window_id = 2
    assert len(history.for_windows([2])) == 1
    assert len(history.for_windows([1])) == 3


# --- search loop ---------------------------------------------------------------


def test_search_respects_the_budget(space):
    tuner = BOTuner(space, quadratic_objective, seed=0, n_initial_random=3)
    history = tuner.run(budget=8, window_id=1, mode=COLD)
    assert len(history) == 8


def test_search_is_reproducible_under_a_fixed_seed(space):
    first = BOTuner(space, quadratic_objective, seed=7, n_initial_random=3).run(budget=8)
    second = BOTuner(space, quadratic_objective, seed=7, n_initial_random=3).run(budget=8)
    assert [t.score for t in first.trials] == [t.score for t in second.trials]


def test_search_improves_on_its_own_random_start(space):
    tuner = BOTuner(space, quadratic_objective, seed=1, n_initial_random=4)
    history = tuner.run(budget=16)
    start_best = min(t.score for t in history.trials[:4])
    final_best = history.best_score()
    assert final_best <= start_best


def test_bo_beats_random_search_on_average(space):
    bo_scores, random_scores = [], []
    for seed in range(5):
        bo_scores.append(
            BOTuner(space, quadratic_objective, seed=seed, n_initial_random=4)
            .run(budget=15)
            .best_score()
        )
        random_scores.append(random_search(space, quadratic_objective, 15, seed=seed).best_score())
    assert np.mean(bo_scores) <= np.mean(random_scores)


def test_resume_continues_rather_than_restarting(space):
    partial = BOTuner(space, quadratic_objective, seed=3, n_initial_random=3).run(budget=5)
    resumed = BOTuner(space, quadratic_objective, seed=3, n_initial_random=3).run(
        budget=10, resume_from=partial
    )
    assert len(resumed) == 10
    for before, after in zip(partial.trials[:5], resumed.trials[:5], strict=True):
        assert before.score == after.score
        assert before.vector == after.vector


def test_resume_from_a_saved_file_is_identical(tmp_path, space):
    partial = BOTuner(space, quadratic_objective, seed=4, n_initial_random=3).run(budget=5)
    path = partial.save(tmp_path / "h.json", space_names=space.names)
    reloaded = TrialHistory.load(path)
    resumed = BOTuner(space, quadratic_objective, seed=4, n_initial_random=3).run(
        budget=9, resume_from=reloaded
    )
    assert [t.score for t in resumed.trials[:5]] == [t.score for t in partial.trials]


def test_warm_all_spends_the_whole_budget_on_informed_proposals(space):
    prior = BOTuner(space, quadratic_objective, seed=0, n_initial_random=4).run(
        budget=12, window_id=1
    )
    tuner = BOTuner(space, quadratic_objective, seed=0, n_initial_random=4)
    history = tuner.run(budget=6, window_id=2, mode=WARM_ALL, prior=prior)
    assert len(history) == 6
    assert all(trial.source == "bo" for trial in history.trials)
    assert all(trial.window_id == 2 for trial in history.trials)


def test_warm_topk_reevaluates_past_configurations(space):
    prior = BOTuner(space, quadratic_objective, seed=0, n_initial_random=4).run(
        budget=12, window_id=1
    )
    tuner = BOTuner(space, quadratic_objective, seed=0, n_initial_random=4)
    history = tuner.run(budget=8, window_id=2, mode=WARM_TOPK, prior=prior, top_k=3)
    assert len(history) == 8
    assert [t.source for t in history.trials[:3]] == ["warm"] * 3
    best_prior_vectors = [t.vector for t in prior.top_k(3)]
    assert [t.vector for t in history.trials[:3]] == best_prior_vectors


def test_warm_modes_require_a_prior(space):
    tuner = BOTuner(space, quadratic_objective, seed=0)
    with pytest.raises(ValueError, match="non-empty prior"):
        tuner.run(budget=5, mode=WARM_ALL, prior=None)


def test_unknown_mode_is_rejected(space):
    tuner = BOTuner(space, quadratic_objective, seed=0)
    with pytest.raises(ValueError, match="unknown mode"):
        tuner.run(budget=5, mode="telepathy")


def test_a_crashing_objective_does_not_kill_the_search(space):
    def explodes(params: dict) -> float:
        if params["num_leaves"] > 60:
            raise RuntimeError("boom")
        return quadratic_objective(params)

    history = BOTuner(space, explodes, seed=2, n_initial_random=4).run(budget=10)
    assert len(history) == 10
    assert np.isfinite(history.best_score())
