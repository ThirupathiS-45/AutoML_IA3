"""Phase 5 reporting: turn the experiment CSVs into tables and plots.

Everything written here is derived arithmetic over ``results/drift_experiment.csv``
and ``results/drift_traces.csv``. No number is invented, and the summary deliberately
separates what was measured from what it might mean.

Run with::

    uv run python -m automl.experiments.summarize
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from automl.config import ensure_parent, resolve_path, setup_logging

LOGGER = logging.getLogger(__name__)

STRATEGY_ORDER = ["no_retrain", "refit", "cold_bo", "warm_bo_all", "warm_bo_topk"]


def pair_index(results: pd.DataFrame) -> list[str]:
    """Return the columns that identify one matched run.

    Strategies are compared on identical (crop, seed, window) combinations. Leaving
    crop out of the index would average four different price regimes into one pair and
    silently destroy the matching.

    Args:
        results: The results table.

    Returns:
        ``["crop", "seed", "window_id"]`` when a crop column is present, else
        ``["seed", "window_id"]``.
    """
    return (["crop"] if "crop" in results.columns else []) + ["seed", "window_id"]


def load_results(path: str | Path) -> pd.DataFrame:
    """Read the drift experiment results.

    Args:
        path: Path to ``results/drift_experiment.csv``.

    Returns:
        The results table.

    Raises:
        FileNotFoundError: If the experiment has not been run.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{resolved} not found. Run: uv run python -m automl.experiments.run_drift"
        )
    return pd.read_csv(resolved)


def summarise_by_strategy(results: pd.DataFrame) -> pd.DataFrame:
    """Average test error per strategy across every post-drift window and seed.

    Window 1 is excluded because it is the shared starting point, not a comparison.

    Args:
        results: The results table.

    Returns:
        One row per strategy with mean and standard deviation of each metric.
    """
    later = results[results["window_id"] > 1]
    grouped = later.groupby("strategy").agg(
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mape_mean=("mape", "mean"),
        trials_mean=("trials_used", "mean"),
        seconds_mean=("seconds", "mean"),
        n_runs=("mae", "size"),
    )
    order = [name for name in STRATEGY_ORDER if name in grouped.index]
    return grouped.loc[order].round(2).reset_index()


def summarise_by_crop(results: pd.DataFrame) -> pd.DataFrame:
    """Mean post-drift test MAE per crop and strategy.

    A single pooled average can hide the fact that different crops prefer different
    strategies, which is exactly what happens here, so this breakdown is reported
    before any pooled number.

    Args:
        results: The results table.

    Returns:
        A wide table, crops as rows and strategies as columns, plus the winning
        strategy for each crop.
    """
    if "crop" not in results.columns:
        return pd.DataFrame()
    later = results[results["window_id"] > 1]
    pivot = later.pivot_table(index="crop", columns="strategy", values="mae", aggfunc="mean")
    columns = [name for name in STRATEGY_ORDER if name in pivot.columns]
    table = pivot[columns].round(1)
    table["best"] = table.idxmin(axis=1)
    return table.reset_index()


def summarise_by_window(results: pd.DataFrame) -> pd.DataFrame:
    """Average test MAE per strategy for each window.

    Args:
        results: The results table.

    Returns:
        A wide table, windows as rows and strategies as columns.
    """
    pivot = results.pivot_table(index="window_id", columns="strategy", values="mae", aggfunc="mean")
    columns = [name for name in STRATEGY_ORDER if name in pivot.columns]
    return pivot[columns].round(1).reset_index()


def summarise_trials_to_target(results: pd.DataFrame) -> pd.DataFrame:
    """Report how many trials each search strategy needed to match cold's final score.

    Args:
        results: The results table.

    Returns:
        One row per search strategy, with the mean trial count and how often the
        target was reached at all.
    """
    searches = results[(results["window_id"] > 1) & results["strategy"].str.contains("bo")]
    rows = []
    for strategy, group in searches.groupby("strategy"):
        reached = group["trials_to_cold_best"].notna()
        rows.append(
            {
                "strategy": strategy,
                "reached_target": f"{int(reached.sum())}/{len(group)}",
                "mean_trials_when_reached": round(
                    float(group.loc[reached, "trials_to_cold_best"].mean()), 2
                )
                if reached.any()
                else float("nan"),
                "median_trials_when_reached": group.loc[reached, "trials_to_cold_best"].median(),
            }
        )
    frame = pd.DataFrame(rows)
    order = [name for name in STRATEGY_ORDER if name in set(frame["strategy"])]
    return frame.set_index("strategy").loc[order].reset_index()


def summarise_validation(results: pd.DataFrame) -> pd.DataFrame:
    """Report the validation error each search actually achieved.

    This is the quantity the tuner optimises, so it isolates search quality from the
    separate question of whether a validation gain survives onto held-out weeks.

    Args:
        results: The results table.

    Returns:
        One row per search strategy.
    """
    searches = results[(results["window_id"] > 1) & results["valid_mae"].notna()]
    return (
        searches.groupby("strategy")["valid_mae"]
        .agg(["mean", "std", "count"])
        .round(2)
        .reset_index()
    )


def head_to_head(results: pd.DataFrame, reference: str = "cold_bo") -> pd.DataFrame:
    """Count how often each strategy beat a reference strategy, run for run.

    Means hide how often a strategy wins. Two methods with identical averages can
    differ in whether one wins narrowly most of the time or loses badly sometimes.

    Args:
        results: The results table.
        reference: The strategy every other is compared against.

    Returns:
        One row per strategy with its win count against the reference.
    """
    later = results[results["window_id"] > 1]
    pivot = later.pivot_table(index=pair_index(results), columns="strategy", values="mae")
    if reference not in pivot.columns:
        return pd.DataFrame()
    rows = []
    for strategy in pivot.columns:
        if strategy == reference:
            continue
        wins = int((pivot[strategy] < pivot[reference]).sum())
        rows.append(
            {
                "strategy": strategy,
                f"beat_{reference}": f"{wins}/{len(pivot)}",
                "mean_mae_difference": round(float((pivot[strategy] - pivot[reference]).mean()), 2),
            }
        )
    frame = pd.DataFrame(rows)
    order = [name for name in STRATEGY_ORDER if name in set(frame["strategy"])]
    return frame.set_index("strategy").loc[order].reset_index()


def paired_tests(results: pd.DataFrame) -> pd.DataFrame:
    """Compare strategies pairwise on the runs they share, with a signed-rank test.

    The strategies were run on identical seeds and identical windows, so each
    (seed, window) is a *matched pair* rather than an independent sample. Comparing
    group means throws that pairing away and inflates the apparent spread: the
    standard deviations here are dominated by how hard each window is, which affects
    every strategy equally. A paired test removes that shared difficulty and asks only
    whether one strategy is consistently better on the same problem.

    The Wilcoxon signed-rank test is used rather than a paired t-test because twenty
    differences are too few to lean on a normality assumption.

    Because several comparisons are made at once, the p-values also carry a
    Holm-Bonferroni correction. Both columns are reported: the raw p-value, and
    whether it survives correction. Reading only the raw column would overstate the
    evidence.

    Args:
        results: The results table.

    Returns:
        One row per comparison, ordered by raw p-value.
    """
    from scipy.stats import wilcoxon

    from automl.drift import holm_bonferroni

    later = results[results["window_id"] > 1]
    pivot = later.pivot_table(index=pair_index(results), columns="strategy", values="mae")
    comparisons = [
        ("warm_bo_all", "cold_bo"),
        ("warm_bo_all", "refit"),
        ("warm_bo_all", "no_retrain"),
        ("warm_bo_topk", "cold_bo"),
        ("cold_bo", "refit"),
        ("cold_bo", "no_retrain"),
        ("refit", "no_retrain"),
    ]
    rows, raw = [], {}
    for left, right in comparisons:
        if left not in pivot.columns or right not in pivot.columns:
            continue
        difference = (pivot[left] - pivot[right]).dropna()
        if len(difference) < 3:
            continue
        _, p_value = wilcoxon(difference)
        label = f"{left} vs {right}"
        raw[label] = float(p_value)
        rows.append(
            {
                "comparison": label,
                "n_pairs": int(len(difference)),
                "wins": f"{int((difference < 0).sum())}/{len(difference)}",
                "mean_mae_difference": round(float(difference.mean()), 2),
                "median_mae_difference": round(float(difference.median()), 2),
                "p_value": round(float(p_value), 4),
            }
        )
    if not rows:
        LOGGER.warning("not enough matched runs for a paired test")
        return pd.DataFrame(
            columns=[
                "comparison",
                "n_pairs",
                "wins",
                "mean_mae_difference",
                "median_mae_difference",
                "p_value",
                "significant_after_holm",
            ]
        )
    flags = holm_bonferroni(raw, alpha=0.05)
    for row in rows:
        row["significant_after_holm"] = bool(flags.get(row["comparison"], False))
    return pd.DataFrame(rows).sort_values("p_value", ignore_index=True)


def paired_tests_by_crop(
    results: pd.DataFrame, left: str = "warm_bo_all", right: str = "refit"
) -> pd.DataFrame:
    """Run the same paired comparison separately within each crop.

    Args:
        results: The results table.
        left: The strategy under test.
        right: The baseline it is compared against.

    Returns:
        One row per crop. The p-values carry a Holm-Bonferroni correction across crops.
    """
    from scipy.stats import wilcoxon

    from automl.drift import holm_bonferroni

    if "crop" not in results.columns:
        return pd.DataFrame()
    later = results[results["window_id"] > 1]
    rows, raw = [], {}
    for crop, group in later.groupby("crop"):
        pivot = group.pivot_table(index=["seed", "window_id"], columns="strategy", values="mae")
        if left not in pivot.columns or right not in pivot.columns:
            continue
        difference = (pivot[left] - pivot[right]).dropna()
        if len(difference) < 3:
            continue
        _, p_value = wilcoxon(difference)
        raw[str(crop)] = float(p_value)
        rows.append(
            {
                "crop": crop,
                "n_pairs": int(len(difference)),
                f"{left}_wins": f"{int((difference < 0).sum())}/{len(difference)}",
                "mean_mae_difference": round(float(difference.mean()), 2),
                "p_value": round(float(p_value), 4),
                "favours": left if difference.mean() < 0 else right,
            }
        )
    if not rows:
        return pd.DataFrame()
    flags = holm_bonferroni(raw, alpha=0.05)
    for row in rows:
        row["significant_after_holm"] = bool(flags.get(str(row["crop"]), False))
    return pd.DataFrame(rows).sort_values("p_value", ignore_index=True)


def summarise_drift(results: pd.DataFrame) -> pd.DataFrame:
    """Report the drift verdict recorded at each window boundary.

    Args:
        results: The results table.

    Returns:
        One row per window with the input-drift and error-drift findings.
    """
    later = results[results["window_id"] > 1]
    grouped = later.groupby("window_id").agg(
        n_drifted_features=("n_drifted_features", "max"),
        n_features_tested=("n_features_tested", "max"),
        input_drift=("input_drift", "max"),
        error_ratio=("error_ratio", "mean"),
        error_drift=("error_drift", "max"),
    )
    return grouped.round(3).reset_index()


def plot_error_vs_trial(traces_path: str | Path, output: str | Path) -> Path | None:
    """Plot the running-best validation error against trial number.

    Args:
        traces_path: Path to ``results/drift_traces.csv``.
        output: Destination image path.

    Returns:
        The saved path, or ``None`` if the trace file or matplotlib is missing.
    """
    resolved = resolve_path(traces_path)
    if not resolved.is_file():
        LOGGER.warning("no traces at %s; skipping plot", resolved)
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib unavailable; skipping plot")
        return None

    traces = pd.read_csv(resolved)
    traces = traces[traces["window_id"] > 1]
    if traces.empty:
        LOGGER.warning("no post-window-1 traces to plot")
        return None

    curves = traces.groupby(["strategy", "trial"])["best_so_far"].mean().reset_index()
    path = ensure_parent(output)
    fig, axis = plt.subplots(figsize=(9, 5))
    for strategy, group in curves.groupby("strategy"):
        axis.plot(group["trial"] + 1, group["best_so_far"], marker="o", ms=3, label=strategy)
    axis.set_xlabel("Trial number")
    axis.set_ylabel("Best validation MAE so far (Rs/quintal)")
    axis.set_title("Search progress after drift, averaged over windows and seeds")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    LOGGER.info("wrote %s", path)
    return path


def plot_strategy_comparison(results: pd.DataFrame, output: str | Path) -> Path | None:
    """Plot mean test MAE per strategy for each post-drift window.

    Args:
        results: The results table.
        output: Destination image path.

    Returns:
        The saved path, or ``None`` if matplotlib is missing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib unavailable; skipping plot")
        return None

    later = results[results["window_id"] > 1]
    pivot = later.pivot_table(index="window_id", columns="strategy", values="mae", aggfunc="mean")
    columns = [name for name in STRATEGY_ORDER if name in pivot.columns]
    path = ensure_parent(output)
    axis = pivot[columns].plot(kind="bar", figsize=(10, 5), rot=0)
    axis.set_xlabel("Window")
    axis.set_ylabel("Test MAE (Rs/quintal)")
    axis.set_title("Test error by retraining strategy")
    axis.grid(alpha=0.3, axis="y")
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=120)
    plt.close(axis.figure)
    LOGGER.info("wrote %s", path)
    return path


def write_summary(results: pd.DataFrame, output: str | Path) -> Path:
    """Write the Markdown summary of the experiment.

    Args:
        results: The results table.
        output: Destination Markdown path.

    Returns:
        The saved path.
    """
    by_strategy = summarise_by_strategy(results)
    by_window = summarise_by_window(results)
    to_target = summarise_trials_to_target(results)
    drift = summarise_drift(results)

    best_row = by_strategy.loc[by_strategy["mae_mean"].idxmin()]
    refit_row = by_strategy[by_strategy["strategy"] == "refit"]
    refit_mae = float(refit_row["mae_mean"].iloc[0]) if not refit_row.empty else float("nan")
    validation = summarise_validation(results)
    versus_cold = head_to_head(results, "cold_bo")
    paired = paired_tests(results)
    by_crop = summarise_by_crop(results)
    per_crop_paired = paired_tests_by_crop(results)

    warm_rows = to_target[to_target["strategy"].str.startswith("warm")]
    cold_trials = to_target.loc[to_target["strategy"] == "cold_bo", "median_trials_when_reached"]
    cold_median = float(cold_trials.iloc[0]) if not cold_trials.empty else float("nan")
    warm_note = ""
    if not warm_rows.empty:
        fastest = warm_rows.loc[warm_rows["median_trials_when_reached"].idxmin()]
        warm_note = (
            f"- When **{fastest['strategy']}** did reach the cold search's final "
            f"validation score, it needed a median of "
            f"{fastest['median_trials_when_reached']:.0f} trials, against the "
            f"{cold_median:.0f} the cold search itself took. It reached that bar in "
            f"{fastest['reached_target']} runs."
        )

    lines = [
        "# Drift experiment summary",
        "",
        "All numbers below were produced by `uv run python -m automl.experiments.run_drift`.",
        "",
        *(
            [f"- Crops: {sorted(results['crop'].unique().tolist())}"]
            if "crop" in results.columns
            else []
        ),
        f"- Seeds: {sorted(results['seed'].unique().tolist())}",
        f"- Windows: {sorted(results['window_id'].unique().tolist())}",
        f"- Trial budget per search: {int(results['trials_used'].max())}",
        "",
        "## Facts: test error by strategy (windows 2 onward)",
        "",
        by_strategy.to_markdown(index=False),
        "",
        *(
            [
                "## Facts: the same thing, split by crop",
                "",
                "Pooling hides disagreement between crops, so this comes first.",
                "",
                by_crop.to_markdown(index=False),
                "",
                "### Warm start against a plain refit, within each crop",
                "",
                per_crop_paired.to_markdown(index=False),
                "",
            ]
            if not by_crop.empty
            else []
        ),
        "## Facts: validation error reached by each search",
        "",
        "This is what the tuner optimises. Test error is a separate question.",
        "",
        validation.to_markdown(index=False),
        "",
        "## Facts: mean test MAE per window",
        "",
        by_window.to_markdown(index=False),
        "",
        "## Facts: head-to-head against the cold search",
        "",
        versus_cold.to_markdown(index=False),
        "",
        "## Facts: paired significance tests",
        "",
        "Every strategy ran on the same seeds and the same windows, so each "
        "(seed, window) is a matched pair. A paired signed-rank test removes the "
        "shared difficulty of each window, which is what the large standard "
        "deviations above are mostly measuring. `wins` counts the pairs where the "
        "left strategy had the lower test MAE; a negative difference favours it.",
        "",
        paired.to_markdown(index=False),
        "",
        "Seven comparisons are made, so `significant_after_holm` is the column that "
        "matters. A raw p-value below 0.05 that does not survive correction is "
        "suggestive, not conclusive.",
        "",
        "## Facts: trials needed to match the cold search's final validation score",
        "",
        "**Read `cold_bo` with care.** The target *is* the cold search's own final "
        "score, so cold reaches it by definition in every run. Only the warm rows "
        "carry information; the cold row is there to show how long cold took to "
        "arrive at the score the warm runs are being asked to match.",
        "",
        to_target.to_markdown(index=False),
        "",
        "## Facts: drift detected at each window boundary",
        "",
        drift.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"- The lowest mean test MAE across post-drift windows came from "
        f"**{best_row['strategy']}** at {best_row['mae_mean']:.1f} "
        f"(SD {best_row['mae_std']:.1f}) rupees per quintal.",
        f"- The plain refit baseline scored {refit_mae:.1f}.",
        warm_note,
        "",
        "Unpaired standard deviations here are roughly a third of the means and every "
        "strategy's interval overlaps every other's, which is why the paired table "
        "above is the one to read: it removes the window-difficulty variance that "
        "those intervals are mostly made of. Treat all of it as evidence about this "
        "dataset and this budget, not as a general finding.",
        "",
    ]
    path = ensure_parent(output)
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("wrote %s", path)
    return path


def run(results_path: str | Path, traces_path: str | Path, outdir: str | Path) -> pd.DataFrame:
    """Produce every summary artefact.

    Args:
        results_path: Path to the results CSV.
        traces_path: Path to the per-trial traces CSV.
        outdir: Directory for the summary and plots.

    Returns:
        The by-strategy summary table.
    """
    results = load_results(results_path)
    out = Path(outdir)
    by_strategy = summarise_by_strategy(results)
    write_summary(results, out / "summary.md")
    plot_error_vs_trial(traces_path, out / "error_vs_trial.png")
    plot_strategy_comparison(results, out / "strategy_comparison.png")
    return by_strategy


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/drift_experiment.csv")
    parser.add_argument("--traces", default="results/drift_traces.csv")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()
    setup_logging()
    table = run(args.results, args.traces, args.outdir)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
