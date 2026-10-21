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
    """Regenerating must reproduce the committed report byte for byte."""
    from automl.config import resolve_path

    committed = resolve_path("docs/report.md")
    if not committed.is_file():
        pytest.skip("report has not been generated yet")
    assert render(
        __import__("automl.experiments.make_report", fromlist=["gather"]).gather()
    ) == committed.read_text(encoding="utf-8")
