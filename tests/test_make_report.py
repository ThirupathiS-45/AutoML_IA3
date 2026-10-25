"""Tests for the report generator.

The report's whole purpose is that its numbers come from the CSVs rather than from
prose, so these tests check exactly that: the generator must build without a dataset,
must not leave unresolved placeholders, and must report "not run yet" rather than
inventing a figure.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from automl.experiments.make_report import _table, render


@pytest.fixture
def frame():
    """A small results table with one all-missing column."""
    return pd.DataFrame(
        {
            "crop": ["Onion", "Onion", "Potato", "Potato"],
            "method": ["naive", "model", "naive", "model"],
            "mae": [60.6251, 56.8758, 30.1234, 29.0001],
            "lags_kept": [None, None, None, None],
        }
    )


def test_table_keeps_only_requested_columns_that_exist(frame):
    rendered = _table(frame, ["crop", "method", "mae", "absent_column"])
    assert "crop" in rendered and "mae" in rendered
    assert "absent_column" not in rendered


def test_table_drops_an_all_missing_column(frame):
    """An entirely empty column carries no information and would only add noise."""
    rendered = _table(frame, ["crop", "method", "mae", "lags_kept"])
    assert "lags_kept" not in rendered
    assert "nan" not in rendered.lower()


def test_table_rounds_numbers(frame):
    rendered = _table(frame, ["crop", "method", "mae"])
    assert "60.63" in rendered
    assert "60.6251" not in rendered


def test_table_can_aggregate_by_group(frame):
    rendered = _table(frame, ["crop", "mae"], group=["crop"])
    assert rendered.count("Onion") == 1
    assert rendered.count("Potato") == 1


def test_table_says_so_when_nothing_was_run():
    assert "Not run yet" in _table(pd.DataFrame(), ["mae"])


def test_table_survives_a_missing_crop_column(frame):
    """An older results CSV has no crop column; the report must still build."""
    rendered = _table(frame.drop(columns=["crop"]), ["crop", "method", "mae"])
    assert "method" in rendered


def test_report_builds_with_no_results_at_all(monkeypatch):
    """The generator must work on a fresh clone before any experiment has run."""
    import automl.experiments.make_report as module

    monkeypatch.setattr(module, "_read", lambda name: pd.DataFrame())
    monkeypatch.setattr(module, "_windows_table", lambda: pd.DataFrame())
    text = module.render(module.gather())
    assert "Not run yet" in text
    assert "## 12. Conclusion" in text


def test_report_leaves_no_unresolved_placeholder(monkeypatch):
    """A stray brace would mean a number never got interpolated."""
    import automl.experiments.make_report as module

    monkeypatch.setattr(module, "_read", lambda name: pd.DataFrame())
    monkeypatch.setattr(module, "_windows_table", lambda: pd.DataFrame())
    text = module.render(module.gather())
    stray = re.findall(r"\{[A-Za-z_][A-Za-z0-9_\[\]'\"\.]*\}", text)
    assert not stray, f"unresolved placeholders: {stray[:5]}"


def test_report_covers_every_course_module(monkeypatch):
    import automl.experiments.make_report as module

    monkeypatch.setattr(module, "_read", lambda name: pd.DataFrame())
    monkeypatch.setattr(module, "_windows_table", lambda: pd.DataFrame())
    text = module.render(module.gather())
    for module_number in range(1, 7):
        assert f"Module {module_number}" in text


def test_report_states_the_honest_framing(monkeypatch):
    """The specification requires this sentence to appear; a test keeps it there."""
    import automl.experiments.make_report as module

    monkeypatch.setattr(module, "_read", lambda name: pd.DataFrame())
    monkeypatch.setattr(module, "_windows_table", lambda: pd.DataFrame())
    text = module.render(module.gather())
    assert "not a new algorithm" in text.lower()
    assert "Preprint only" in text


def test_report_writes_to_disk(tmp_path, monkeypatch):
    import automl.experiments.make_report as module

    monkeypatch.setattr(module, "_read", lambda name: pd.DataFrame())
    monkeypatch.setattr(module, "_windows_table", lambda: pd.DataFrame())
    path = module.run(tmp_path / "report.md")
    assert path.is_file()
    assert path.read_text().startswith("# Drift-Aware AutoML")


def test_generated_report_matches_the_committed_results():
    """Regenerating must reproduce the committed report byte for byte.

    This needs the processed dataset, because one section reports the window layout
    computed from it. The dataset is deliberately git-ignored, so in a clean CI
    checkout this check is skipped rather than failed: it would otherwise break every
    build for a reason that says nothing about the code.
    """
    from automl.config import resolve_path
    from automl.experiments.make_report import gather

    committed = resolve_path("docs/report.md")
    if not committed.is_file():
        pytest.skip("report has not been generated yet")
    if not resolve_path("data/processed/weekly.parquet").is_file():
        pytest.skip("processed dataset absent; run automl.data.aggregate to check this")
    assert render(gather()) == committed.read_text(encoding="utf-8")


def test_report_is_self_consistent_without_the_dataset():
    """Whatever the report says about windows, it must still build and stay coherent.

    The dataset-dependent section is the only part that varies, so this is the check
    that genuinely runs in CI: the report builds, keeps its structure, and never
    silently emits an empty table without the stand-in note.
    """
    from automl.experiments.make_report import gather

    text = render(gather())
    assert text.startswith("# Drift-Aware AutoML")
    assert "## 12. Conclusion and future work" in text
    assert "|  |" not in text, "an empty table leaked into the report"
