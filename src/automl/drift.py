"""Detect that the world has changed, in two independent ways.

**Input drift** asks whether the features themselves look different. For each numeric
feature we run a two-sample Kolmogorov-Smirnov test comparing the reference window
against the current one. The KS test measures the largest gap between two cumulative
distributions and reports how surprising that gap would be if both samples came from
the same source. A small p-value means the distributions genuinely differ. Because we
test many features at once, the p-values are corrected with the Holm-Bonferroni
procedure, which keeps the chance of at least one false alarm at the stated level.

**Error drift** asks the more practical question: has the deployed model actually got
worse? That is a plain comparison of test error before and after.

Both are reported. Input drift can fire while the model stays accurate, and a model
can decay while its inputs look unchanged, so neither alone is the whole story.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

LOGGER = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """The outcome of one drift check.

    Attributes:
        input_drift: Whether any feature's distribution shifted significantly.
        n_drifted_features: How many features were flagged after correction.
        n_features_tested: How many features were tested.
        min_p_value: The smallest raw p-value observed.
        per_feature: Raw p-value and KS statistic for each feature.
        error_ratio: Current error divided by reference error, if supplied.
        error_drift: Whether the error rose by more than the configured margin.
    """

    input_drift: bool
    n_drifted_features: int
    n_features_tested: int
    min_p_value: float
    per_feature: dict[str, dict[str, float]] = field(default_factory=dict)
    error_ratio: float | None = None
    error_drift: bool | None = None

    def as_record(self) -> dict[str, Any]:
        """Flatten into a row for the results CSV.

        Returns:
            A mapping of scalar fields, excluding the per-feature detail.
        """
        return {
            "input_drift": self.input_drift,
            "n_drifted_features": self.n_drifted_features,
            "n_features_tested": self.n_features_tested,
            "min_p_value": self.min_p_value,
            "error_ratio": self.error_ratio,
            "error_drift": self.error_drift,
        }


def holm_bonferroni(p_values: dict[str, float], alpha: float) -> dict[str, bool]:
    """Apply the Holm-Bonferroni step-down correction for multiple testing.

    Testing twelve features at the 5% level would raise a false alarm about half the
    time by chance alone. Holm-Bonferroni sorts the p-values and uses a progressively
    looser threshold, which controls that risk without being as blunt as dividing
    alpha by the number of tests for every feature.

    Args:
        p_values: Raw p-value per feature name.
        alpha: Desired family-wise error rate.

    Returns:
        A mapping from feature name to whether it is significant after correction.
    """
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    rejected: dict[str, bool] = {}
    still_rejecting = True
    for rank, (name, p_value) in enumerate(ordered):
        threshold = alpha / (total - rank)
        if still_rejecting and p_value <= threshold:
            rejected[name] = True
        else:
            still_rejecting = False
            rejected[name] = False
    return rejected


def detect_input_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    alpha: float = 0.05,
) -> DriftResult:
    """Compare two feature blocks with per-feature KS tests.

    Args:
        reference: Feature rows from the earlier period.
        current: Feature rows from the later period.
        columns: Numeric columns to test. Defaults to every numeric column shared
            by both frames.
        alpha: Family-wise significance level.

    Returns:
        The drift verdict, with per-feature detail.
    """
    if columns is None:
        numeric = reference.select_dtypes(include=[np.number]).columns
        columns = [column for column in numeric if column in current.columns]

    per_feature: dict[str, dict[str, float]] = {}
    p_values: dict[str, float] = {}
    for column in columns:
        left = reference[column].dropna().to_numpy(dtype=float)
        right = current[column].dropna().to_numpy(dtype=float)
        if left.size < 5 or right.size < 5:
            continue
        statistic, p_value = ks_2samp(left, right)
        per_feature[column] = {"ks_statistic": float(statistic), "p_value": float(p_value)}
        p_values[column] = float(p_value)

    flags = holm_bonferroni(p_values, alpha)
    n_drifted = sum(flags.values())
    result = DriftResult(
        input_drift=n_drifted > 0,
        n_drifted_features=int(n_drifted),
        n_features_tested=len(p_values),
        min_p_value=min(p_values.values()) if p_values else float("nan"),
        per_feature=per_feature,
    )
    LOGGER.info(
        "input drift: %d/%d features flagged (min p=%.3g)",
        result.n_drifted_features,
        result.n_features_tested,
        result.min_p_value,
    )
    return result


def error_drift_verdict(
    reference_error: float, current_error: float, margin: float = 0.10
) -> tuple[float, bool]:
    """Decide whether a deployed model's error has risen enough to count as decay.

    Args:
        reference_error: Error of the deployed model on the previous window.
        current_error: Error of the same model on the current window.
        margin: Relative rise that counts as decay. ``0.10`` means ten percent worse.

    Returns:
        A tuple of (error ratio, whether the rise exceeds the margin).
    """
    if reference_error and np.isfinite(reference_error) and reference_error > 0:
        ratio = float(current_error / reference_error)
    else:
        ratio = float("nan")
    return ratio, bool(np.isfinite(ratio) and ratio > 1.0 + margin)


def add_error_drift(
    result: DriftResult, reference_error: float, current_error: float, margin: float = 0.10
) -> DriftResult:
    """Attach an error-rise verdict to an existing input-drift result.

    Args:
        result: The result returned by :func:`detect_input_drift`.
        reference_error: Error of the deployed model on the previous window.
        current_error: Error of the same model on the current window.
        margin: Relative rise that counts as decay. ``0.10`` means ten percent worse.

    Returns:
        The same result object, with ``error_ratio`` and ``error_drift`` filled in.
    """
    ratio, decayed = error_drift_verdict(reference_error, current_error, margin)
    result.error_ratio = ratio
    result.error_drift = decayed
    LOGGER.info("error drift: ratio=%.3f -> %s", ratio, result.error_drift)
    return result
