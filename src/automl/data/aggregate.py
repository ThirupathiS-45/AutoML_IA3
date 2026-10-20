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
from automl.data.clean import clean_daily, series_keys
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
    keys = series_keys(frame)
    indexed = frame.set_index("date")
    # label/closed="left" makes the bucket label the Monday the week STARTS on.
    # Pandas defaults to labelling a W-MON bucket by the Monday it ends on, which
    # would silently shift every observation forward by a week.
    grouped = indexed.groupby(keys, observed=True)["modal_price"].resample(
        frequency, label="left", closed="left"
    )
    weekly = grouped.agg([how, "count"]).reset_index()
    weekly = weekly.rename(columns={how: "modal_price", "count": "n_days", "date": "week_start"})

    weekly.loc[weekly["n_days"] < min_days, "modal_price"] = pd.NA

    # Every series is reindexed onto one shared weekly grid, so a gap is an explicit
    # missing value rather than an absent row that later code could silently skip.
    full_index = pd.date_range(
        weekly["week_start"].min(), weekly["week_start"].max(), freq=frequency
    )
    series = sorted(
        {tuple(row) for row in weekly[keys].dropna().itertuples(index=False, name=None)}
    )
    grid = pd.MultiIndex.from_tuples(
        [(*combo, week) for combo in series for week in full_index],
        names=[*keys, "week_start"],
    )
    weekly = (
        weekly.set_index([*keys, "week_start"])
        .reindex(grid)
        .reset_index()
        .sort_values([*keys, "week_start"], ignore_index=True)
    )
    weekly["n_days"] = weekly["n_days"].fillna(0).astype(int)
    weekly["modal_price"] = pd.to_numeric(weekly["modal_price"], errors="coerce")
    LOGGER.info(
        "weekly grid: %d weeks x %d series = %d rows",
        len(full_index),
        len(series),
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
    keys = series_keys(weekly)
    rows = []
    for combo, group in weekly.groupby(keys, observed=True):
        combo = combo if isinstance(combo, tuple) else (combo,)
        total = len(group)
        present = int(group["modal_price"].notna().sum())
        rows.append(
            {
                **dict(zip(keys, combo, strict=True)),
                "weeks_total": total,
                "weeks_present": present,
                "weeks_missing": total - present,
                "missing_fraction": (total - present) / total if total else 1.0,
                "first_week": group["week_start"].min(),
                "last_week": group["week_start"].max(),
                "mean_price": group["modal_price"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [*[k for k in keys if k != "market"], "missing_fraction"], ignore_index=True
    )


def select_markets(
    weekly: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the ``n_markets`` best-covered markets, chosen separately for each crop.

    Coverage decides, but only when the differences are real. Candidates here are all
    roughly 99% complete, so ranking on a two-week difference out of four hundred would
    be noise dressed up as a criterion. Markets within ``coverage_tie_tolerance`` of the
    best are treated as equally well covered and ordered by the config's own listing,
    which puts the economically important markets first.

    Args:
        weekly: Output of :func:`to_weekly`.
        config: Parsed ``configs/data.yaml``.

    Returns:
        A tuple of (filtered weekly frame, annotated coverage table for all candidates).
    """
    coverage = coverage_table(weekly)
    tolerance = float(config.get("coverage_tie_tolerance", 0.0))
    preference = {
        str(entry["name"]): rank for rank, entry in enumerate(config.get("candidate_markets", []))
    }
    keys = series_keys(weekly)
    group_keys = [key for key in keys if key != "market"]

    best = (
        coverage.groupby(group_keys, observed=True)["missing_fraction"].transform("min")
        if group_keys
        else coverage["missing_fraction"].min()
    )
    coverage = coverage.assign(
        tied=(coverage["missing_fraction"] - best) <= tolerance,
        preference=coverage["market"].map(preference).fillna(len(preference)),
    ).sort_values(
        [*group_keys, "tied", "preference", "missing_fraction"],
        ascending=[*[True] * len(group_keys), False, True, True],
        ignore_index=True,
    )

    n_markets = int(config["n_markets"])
    kept = (
        coverage.groupby(group_keys, observed=True, sort=False).head(n_markets)
        if group_keys
        else coverage.head(n_markets)
    )
    wanted = {tuple(row) for row in kept[keys].itertuples(index=False, name=None)}
    mask = [tuple(row) in wanted for row in weekly[keys].itertuples(index=False, name=None)]
    for combo, group in kept.groupby(group_keys, observed=True) if group_keys else []:
        LOGGER.info("%s: keeping %s", combo, ", ".join(group["market"]))
    if not group_keys:
        LOGGER.info("keeping markets: %s", ", ".join(kept["market"]))
    return weekly[mask].reset_index(drop=True), coverage


def drop_violating_series(
    weekly: pd.DataFrame, coverage: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Remove series whose missing-week fraction breaks the stop rule.

    The project spec's stop rule exists so a badly covered series is never modelled
    silently. With one crop that meant halting. With several crops, halting on one bad
    series would block every good one, so the default is to **exclude and report**:
    the offending series are dropped, named in the quality report, and logged as
    errors. Set ``on_stop_rule_violation: halt`` in the data config to get the
    original halting behaviour instead.

    Args:
        weekly: The selected weekly frame.
        coverage: Coverage table from :func:`coverage_table`.
        config: Parsed ``configs/data.yaml``.

    Returns:
        A tuple of (frame with violating series removed, violation messages).
    """
    violations = check_stop_rule(coverage, weekly, config)
    if not violations:
        return weekly, []

    keys = series_keys(weekly)
    limit = float(config["max_missing_week_fraction"])
    bad = {
        tuple(getattr(row, key) for key in keys)
        for row in coverage.itertuples()
        if row.missing_fraction > limit
    }
    mask = [tuple(row) not in bad for row in weekly[keys].itertuples(index=False, name=None)]
    for message in violations:
        LOGGER.error("STOP RULE: %s -- series excluded", message)
    return weekly[mask].reset_index(drop=True), violations


def check_stop_rule(
    coverage: pd.DataFrame, selected: pd.DataFrame, config: dict[str, Any]
) -> list[str]:
    """Apply the project stop rule on missing weeks.

    Args:
        coverage: Coverage table from :func:`coverage_table`.
        selected: The filtered weekly frame that was kept.
        config: Parsed ``configs/data.yaml``.

    Returns:
        A list of human-readable violation messages. Empty means the rule passed.
    """
    limit = float(config["max_missing_week_fraction"])
    keys = series_keys(selected)
    unique = selected[keys].drop_duplicates()
    wanted = {tuple(row) for row in unique.itertuples(index=False, name=None)}
    violations = []
    for row in coverage.itertuples():
        combo = tuple(getattr(row, key) for key in keys)
        if combo in wanted and row.missing_fraction > limit:
            violations.append(
                f"{' / '.join(str(part) for part in combo)}: "
                f"{row.missing_fraction:.1%} of weeks missing (limit {limit:.0%})"
            )
    return violations


def write_quality_report(
    coverage: pd.DataFrame,
    selected: pd.DataFrame,
    violations: list[str],
    config: dict[str, Any],
) -> Path:
    """Write a Markdown data-quality report.

    Args:
        coverage: Coverage table for all candidate series.
        selected: The filtered weekly frame that was saved.
        violations: Stop-rule messages, empty if the rule passed.
        config: Parsed ``configs/data.yaml``.

    Returns:
        The path the report was written to.
    """
    path = ensure_parent(config["paths"]["quality_report"])
    keys = series_keys(selected)
    display = coverage.drop(columns=["tied", "preference"], errors="ignore").copy()
    display["missing_fraction"] = display["missing_fraction"].map("{:.2%}".format)
    display["mean_price"] = display["mean_price"].map("{:.0f}".format)
    for column in ("first_week", "last_week"):
        display[column] = pd.to_datetime(display[column]).dt.date

    kept = selected[keys].drop_duplicates().sort_values(keys)
    limit = float(config["max_missing_week_fraction"])

    lines = [
        "# Data quality report",
        "",
        f"- Crops: **{', '.join(str(crop['name']) for crop in config['crops'])}**",
        f"- Entity level: **{config['source']['entity_level']}** "
        "(state-wide aggregate of mandi modal prices, not a single mandi)",
        "- Source: CEDA Agmarknet portal, Ashoka University "
        "(underlying data: Agmarknet, Ministry of Agriculture and Farmers Welfare, GoI)",
        f"- Requested range: {config['date_range']['start']} to {config['date_range']['end']}",
        f"- Weekly frequency: `{config['aggregate']['frequency']}`",
        "- Price unit: rupees per quintal",
        "",
        "## Coverage by candidate series",
        "",
        display.to_markdown(index=False),
        "",
        "## Stop rule",
        "",
    ]
    if violations:
        lines.append(
            f"**{len(violations)} series excluded** for having more than "
            f"{limit:.0%} of weeks missing. They are not modelled and do not appear "
            "in the saved series below:"
        )
        lines.extend(f"- {violation}" for violation in violations)
    else:
        lines.append(f"**PASSED** - every selected series is within the {limit:.0%} missing limit.")

    lines += ["", "## Saved series", "", kept.to_markdown(index=False), ""]
    weeks = selected.groupby(keys, observed=True).size()
    lines += [
        f"- Series kept: {len(kept)}",
        f"- Rows: {len(selected)}",
        f"- Weeks per series: {int(weeks.iloc[0]) if len(weeks) else 0}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("quality report written to %s", path)
    return path


def plot_series(weekly: pd.DataFrame, config: dict[str, Any]) -> Path | None:
    """Plot the weekly series, one panel per crop.

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
    crops = sorted(weekly["crop"].unique()) if "crop" in weekly.columns else [None]
    fig, axes = plt.subplots(
        len(crops), 1, figsize=(13, 3.2 * len(crops)), sharex=True, squeeze=False
    )
    for axis, crop in zip(axes[:, 0], crops, strict=True):
        block = weekly if crop is None else weekly[weekly["crop"] == crop]
        for market, group in block.groupby("market", observed=True):
            axis.plot(group["week_start"], group["modal_price"], linewidth=1.0, label=market)
        axis.set_title(f"{crop or config.get('crop', 'crop')} - weekly modal price")
        axis.set_ylabel("Rs/quintal")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, ncol=3)
    axes[-1, 0].set_xlabel("Week")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
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

    mode = str(config.get("on_stop_rule_violation", "exclude"))
    if mode == "halt":
        violations = check_stop_rule(coverage, weekly, config)
    else:
        weekly, violations = drop_violating_series(weekly, coverage, config)

    parquet_path = ensure_parent(config["paths"]["weekly_parquet"])
    weekly.to_parquet(parquet_path, index=False)
    LOGGER.info("weekly series written to %s", parquet_path)

    write_quality_report(coverage, weekly, violations, config)
    plot_series(weekly, config)

    if violations and mode == "halt":
        LOGGER.error("STOP RULE TRIGGERED, halting: %s", "; ".join(violations))
        raise SystemExit(1)
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
