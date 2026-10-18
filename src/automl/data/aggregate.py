"""Aggregate cleaned daily records into a weekly mean modal price series.

This is the Phase 1 entry point. It loads the raw CSVs, cleans them, keeps the
best-covered markets, resamples to weekly buckets, writes
``data/processed/weekly.parquet`` and produces a data-quality report.

Run with::

    uv run python -m automl.data.aggregate --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from automl.config import ensure_parent, load_config, setup_logging
from automl.data.clean import clean_daily
from automl.data.load import load_raw_directory

LOGGER = logging.getLogger(__name__)


def to_weekly(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Resample cleaned daily records into weekly buckets, one row per market-week.

    A week is kept only if it contains at least ``aggregate.min_days_per_week``
    observed days; otherwise its price is recorded as missing. Every market is
    reindexed onto the same continuous weekly grid, so gaps are explicit rather
    than silently absent.

    Args:
        frame: Cleaned daily records.
        config: Parsed ``configs/data.yaml``.

    Returns:
        A long dataframe with columns ``week_start``, ``market``, ``modal_price``
        and ``n_days``, sorted by market then week.
    """
    agg_cfg = config["aggregate"]
    frequency = str(agg_cfg["frequency"])
    min_days = int(agg_cfg["min_days_per_week"])

    how = str(agg_cfg["agg"])
    indexed = frame.set_index("date")
    # label/closed="left" makes the bucket label the Monday the week STARTS on.
    # Pandas defaults to labelling a W-MON bucket by the Monday it ends on, which
    # would silently shift every observation forward by a week.
    grouped = indexed.groupby("market", observed=True)["modal_price"].resample(
        frequency, label="left", closed="left"
    )
    weekly = grouped.agg([how, "count"]).reset_index()
    weekly = weekly.rename(columns={how: "modal_price", "count": "n_days", "date": "week_start"})

    weekly.loc[weekly["n_days"] < min_days, "modal_price"] = pd.NA

    full_index = pd.date_range(
        weekly["week_start"].min(), weekly["week_start"].max(), freq=frequency
    )
    markets = sorted(weekly["market"].dropna().unique())
    grid = pd.MultiIndex.from_product([markets, full_index], names=["market", "week_start"])
    weekly = (
        weekly.set_index(["market", "week_start"])
        .reindex(grid)
        .reset_index()
        .sort_values(["market", "week_start"], ignore_index=True)
    )
    weekly["n_days"] = weekly["n_days"].fillna(0).astype(int)
    weekly["modal_price"] = pd.to_numeric(weekly["modal_price"], errors="coerce")
    LOGGER.info(
        "weekly grid: %d weeks x %d markets = %d rows",
        len(full_index),
        len(markets),
        len(weekly),
    )
    return weekly


def coverage_table(weekly: pd.DataFrame) -> pd.DataFrame:
    """Summarise how complete each market's weekly series is.

    Args:
        weekly: Output of :func:`to_weekly`.

    Returns:
        One row per market with week counts, the missing fraction and the date range.
    """
    rows = []
    for market, group in weekly.groupby("market", observed=True):
        total = len(group)
        present = int(group["modal_price"].notna().sum())
        rows.append(
            {
                "market": market,
                "weeks_total": total,
                "weeks_present": present,
                "weeks_missing": total - present,
                "missing_fraction": (total - present) / total if total else 1.0,
                "first_week": group["week_start"].min(),
                "last_week": group["week_start"].max(),
                "mean_price": group["modal_price"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_fraction", ignore_index=True)


def select_markets(
    weekly: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the ``n_markets`` best-covered markets.

    Args:
        weekly: Output of :func:`to_weekly`.
        config: Parsed ``configs/data.yaml``.

    Returns:
        A tuple of (filtered weekly frame, coverage table for the full candidate set).
    """
    coverage = coverage_table(weekly)
    tolerance = float(config.get("coverage_tie_tolerance", 0.0))
    best = float(coverage["missing_fraction"].min())

    # Coverage decides, but only when the differences are real. Every candidate here
    # is ~99% complete, so ranking on a two-week difference out of four hundred would
    # be noise dressed up as a criterion. Markets within `tolerance` of the best are
    # treated as equally well covered and ordered by the config's own listing, which
    # puts the economically important onion markets first.
    preference = {
        str(entry["name"]): rank for rank, entry in enumerate(config.get("candidate_markets", []))
    }
    coverage = coverage.assign(
        tied=(coverage["missing_fraction"] - best) <= tolerance,
        preference=coverage["market"].map(preference).fillna(len(preference)),
    ).sort_values(
        ["tied", "preference", "missing_fraction"],
        ascending=[False, True, True],
        ignore_index=True,
    )

    keep = coverage.head(int(config["n_markets"]))["market"].tolist()
    LOGGER.info(
        "keeping markets: %s (tie tolerance %.1f%% around best coverage %.2f%%)",
        ", ".join(keep),
        tolerance * 100,
        best * 100,
    )
    return weekly[weekly["market"].isin(keep)].reset_index(drop=True), coverage


def check_stop_rule(
    coverage: pd.DataFrame, selected: list[str], config: dict[str, Any]
) -> list[str]:
    """Apply the project stop rule on missing weeks.

    Args:
        coverage: Coverage table from :func:`coverage_table`.
        selected: Market names that were kept.
        config: Parsed ``configs/data.yaml``.

    Returns:
        A list of human-readable violation messages. Empty means the rule passed.
    """
    limit = float(config["max_missing_week_fraction"])
    chosen = coverage[coverage["market"].isin(selected)]
    violations = [
        f"{row.market}: {row.missing_fraction:.1%} of weeks missing (limit {limit:.0%})"
        for row in chosen.itertuples()
        if row.missing_fraction > limit
    ]
    return violations


def write_quality_report(
    coverage: pd.DataFrame,
    selected: pd.DataFrame,
    violations: list[str],
    config: dict[str, Any],
) -> Path:
    """Write a Markdown data-quality report.

    Args:
        coverage: Coverage table for all candidate markets.
        selected: The filtered weekly frame that was saved.
        violations: Stop-rule messages, empty if the rule passed.
        config: Parsed ``configs/data.yaml``.

    Returns:
        The path the report was written to.
    """
    path = ensure_parent(config["paths"]["quality_report"])
    display = coverage.drop(columns=["tied", "preference"], errors="ignore").copy()
    display["missing_fraction"] = display["missing_fraction"].map("{:.2%}".format)
    display["mean_price"] = display["mean_price"].map("{:.0f}".format)
    for column in ("first_week", "last_week"):
        display[column] = pd.to_datetime(display[column]).dt.date

    lines = [
        "# Data quality report",
        "",
        f"- Commodity: **{config['crop']}**",
        f"- Entity level: **{config['source']['entity_level']}** "
        "(state-wide aggregate of mandi modal prices, not a single mandi)",
        "- Source: CEDA Agmarknet portal, Ashoka University (underlying data: Agmarknet, GoI)",
        f"- Requested range: {config['date_range']['start']} to {config['date_range']['end']}",
        f"- Weekly frequency: `{config['aggregate']['frequency']}`",
        "",
        "## Coverage by candidate market",
        "",
        display.to_markdown(index=False),
        "",
        "## Stop rule",
        "",
    ]
    limit = float(config["max_missing_week_fraction"])
    if violations:
        lines.append(f"**FAILED** - missing weeks above the {limit:.0%} limit:")
        lines.extend(f"- {violation}" for violation in violations)
    else:
        lines.append(f"**PASSED** - every selected market is within the {limit:.0%} missing limit.")
    lines += [
        "",
        "## Saved series",
        "",
        f"- Markets kept: {', '.join(sorted(selected['market'].unique()))}",
        f"- Rows: {len(selected)}",
        f"- Weeks per market: {selected.groupby('market', observed=True).size().iloc[0]}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("quality report written to %s", path)
    return path


def plot_series(weekly: pd.DataFrame, config: dict[str, Any]) -> Path | None:
    """Plot the weekly series for each selected market.

    Args:
        weekly: The filtered weekly frame.
        config: Parsed ``configs/data.yaml``.

    Returns:
        The path of the saved figure, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not installed (research group); skipping plot")
        return None

    path = ensure_parent(config["paths"]["quality_plot"])
    fig, axis = plt.subplots(figsize=(12, 5))
    for market, group in weekly.groupby("market", observed=True):
        axis.plot(group["week_start"], group["modal_price"], linewidth=1.2, label=market)
    axis.set_xlabel("Week")
    axis.set_ylabel("Modal price (Rs/quintal)")
    axis.set_title(f"Weekly {config['crop']} modal price by state")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    LOGGER.info("plot written to %s", path)
    return path


def run(config_path: str | Path) -> pd.DataFrame:
    """Build the weekly series and write all Phase 1 outputs.

    Args:
        config_path: Path to ``configs/data.yaml``.

    Returns:
        The weekly dataframe that was saved.
    """
    config = load_config(config_path)
    daily = clean_daily(load_raw_directory(config), config)

    daily_path = ensure_parent(config["paths"]["daily_csv"])
    daily.to_csv(daily_path, index=False)

    weekly_all = to_weekly(daily, config)
    weekly, coverage = select_markets(weekly_all, config)
    selected_names = sorted(weekly["market"].unique())
    violations = check_stop_rule(coverage, selected_names, config)

    parquet_path = ensure_parent(config["paths"]["weekly_parquet"])
    weekly.to_parquet(parquet_path, index=False)
    LOGGER.info("weekly series written to %s", parquet_path)

    write_quality_report(coverage, weekly, violations, config)
    plot_series(weekly, config)

    if violations:
        LOGGER.error("STOP RULE TRIGGERED: %s", "; ".join(violations))
    return weekly


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    setup_logging()
    run(args.config)


if __name__ == "__main__":
    main()
