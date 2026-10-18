"""Tests for the drift detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.drift import detect_input_drift, error_drift_verdict, holm_bonferroni


def _frame(mean: float, n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "lag_1": rng.normal(mean, 100, n),
            "lag_2": rng.normal(mean, 100, n),
            "roll_mean_4": rng.normal(mean, 80, n),
        }
    )


def test_identical_distributions_do_not_trigger_drift():
    result = detect_input_drift(_frame(1000, seed=1), _frame(1000, seed=2))
    assert result.input_drift is False
    assert result.n_drifted_features == 0
    assert result.n_features_tested == 3


def test_a_large_shift_triggers_drift():
    result = detect_input_drift(_frame(1000, seed=1), _frame(2500, seed=2))
    assert result.input_drift is True
    assert result.n_drifted_features == 3
    assert result.min_p_value < 0.05


def test_tiny_samples_are_skipped_rather_than_guessed():
    reference = _frame(1000, n=3)
    result = detect_input_drift(reference, reference)
    assert result.n_features_tested == 0
    assert result.input_drift is False


def test_drift_result_flattens_for_csv():
    record = detect_input_drift(_frame(1000, seed=1), _frame(1000, seed=2)).as_record()
    assert set(record) == {
        "input_drift",
        "n_drifted_features",
        "n_features_tested",
        "min_p_value",
        "error_ratio",
        "error_drift",
    }


def test_holm_bonferroni_is_stricter_than_the_raw_threshold():
    p_values = {"a": 0.04, "b": 0.045, "c": 0.049}
    flags = holm_bonferroni(p_values, alpha=0.05)
    assert not any(flags.values())  # each p < 0.05 alone, none survives correction


def test_holm_bonferroni_keeps_a_clearly_significant_feature():
    flags = holm_bonferroni({"a": 1e-8, "b": 0.6, "c": 0.9}, alpha=0.05)
    assert flags["a"] is True
    assert flags["b"] is False


def test_holm_bonferroni_handles_no_tests():
    assert holm_bonferroni({}, alpha=0.05) == {}


def test_error_drift_fires_above_the_margin():
    ratio, decayed = error_drift_verdict(100.0, 130.0, margin=0.10)
    assert ratio == pytest.approx(1.3)
    assert decayed is True


def test_error_drift_stays_quiet_inside_the_margin():
    _, decayed = error_drift_verdict(100.0, 105.0, margin=0.10)
    assert decayed is False


def test_error_drift_is_undefined_without_a_reference():
    ratio, decayed = error_drift_verdict(0.0, 120.0)
    assert np.isnan(ratio)
    assert decayed is False
