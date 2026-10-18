"""Phase 0 smoke test: the package imports and the repo skeleton is in place."""

from pathlib import Path

import automl

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert automl.__version__ == "0.1.0"


def test_expected_directories_exist():
    for rel in ["configs", "src/automl", "tests", "app", "results", "models"]:
        assert (REPO_ROOT / rel).is_dir(), f"missing directory: {rel}"


def test_configs_are_present():
    for name in ["data.yaml", "features.yaml", "search_space.yaml", "experiment.yaml"]:
        assert (REPO_ROOT / "configs" / name).is_file(), f"missing config: {name}"
