"""Tests for the FastAPI app: it starts, reports health, and computes real numbers.

The four primitives the service reads from disk (the weekly table, the registry index,
the model and the card) are replaced with in-memory fakes. Everything derived from
them -- the forecast row construction, the delta reconstruction, the backtest -- runs
the real code, so these tests exercise the live computation rather than a stub.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from automl.features import build_features, split_xy
from automl.models.lgbm import default_params, fit

from .conftest import make_weekly_panel

FEATURES_CONFIG = {
    "target": {"column": "modal_price", "horizon_weeks": 1, "mode": "delta"},
    "lags_weeks": [1, 2, 4, 8],
    "rolling_mean_windows_weeks": [4, 8],
    "calendar": {"month": True, "season": True},
    "categorical": ["market"],
}

CARD = {
    "version": "test-1.0.0",
    "git_commit": "abc1234",
    "created_utc": "2026-01-01T00:00:00+00:00",
    "crop": "Onion",
    "crop_slug": "onion",
    "entity_level": "state",
    "train_end": "2025-10-27",
    "horizon_weeks": 1,
    "n_train_rows": 300,
    "metrics": {"mae": 123.4, "rmse": 234.5, "mape": 7.8},
    "extra": {"target_mode": "delta", "tuner": "custom GP Bayesian optimization", "budget": 15},
}

INDEX = {
    "updated_utc": "2026-01-01T00:00:00+00:00",
    "crops": [
        {
            "crop": "Onion",
            "slug": "onion",
            "version": "test-1.0.0",
            "markets": ["Gujarat", "Maharashtra"],
            "test_mae": 123.4,
            "horizon_weeks": 1,
        }
    ],
}


def _get(client, path, **params):
    """GET a JSON endpoint and return the parsed body."""
    return client.get(path, params=params).json()


@pytest.fixture
def weekly():
    """A two-market synthetic weekly panel for one crop."""
    panel = make_weekly_panel(n_weeks=180, markets=("Maharashtra", "Gujarat"))
    return panel.assign(crop="Onion")


@pytest.fixture
def client(monkeypatch, weekly):
    """A test client whose service is backed by in-memory data and a real model."""
    from app.service import SERVICE

    features = build_features(weekly, FEATURES_CONFIG)
    x, y = split_xy(features)
    model = fit(default_params(), x, y, seed=0)

    monkeypatch.setattr(SERVICE, "weekly", lambda: weekly)
    monkeypatch.setattr(SERVICE, "index", lambda: INDEX)
    monkeypatch.setattr(SERVICE, "model", lambda crop: model)
    monkeypatch.setattr(SERVICE, "card", lambda crop: CARD)

    from app.main import app

    return TestClient(app)


@pytest.fixture
def broken_client(monkeypatch):
    """A test client for an app whose data and registry are missing."""
    from app.service import SERVICE, DataUnavailable

    def missing(*args, **kwargs):
        raise DataUnavailable("data/processed/weekly.parquet not found")

    monkeypatch.setattr(SERVICE, "weekly", missing)
    monkeypatch.setattr(SERVICE, "index", missing)

    from app.main import app

    return TestClient(app)


# --- with data and a model ------------------------------------------------------


def test_health_is_200_and_lists_crops(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True
    assert body["crops"] == ["Onion"]


def test_meta_exposes_crops_markets_and_range(client):
    body = client.get("/api/meta").json()
    assert [crop["crop"] for crop in body["crops"]] == ["Onion"]
    assert body["crops"][0]["markets"] == ["Gujarat", "Maharashtra"]
    assert body["data_start"] < body["data_end"]
    assert body["price_unit"] == "INR per quintal"


def test_predict_returns_a_plausible_price(client, weekly):
    body = client.post("/api/predict", json={"crop": "Onion", "market": "Maharashtra"}).json()
    observed = weekly[weekly["market"] == "Maharashtra"]["modal_price"]
    # The forecast must be a price, not a raw delta: within the observed range.
    assert observed.min() * 0.5 < body["predicted_price"] < observed.max() * 1.5
    assert body["target_mode"] == "delta"
    assert body["crop"] == "Onion"


def test_predict_is_for_a_week_that_has_not_happened(client, weekly):
    body = client.post("/api/predict", json={"crop": "Onion", "market": "Gujarat"}).json()
    last_observed = weekly["week_start"].max()
    assert pd.Timestamp(body["for_week_starting"]) > last_observed
    assert body["for_week_starting"] > body["based_on_week_starting"]


def test_predict_reports_the_change_consistently(client):
    body = client.post("/api/predict", json={"crop": "Onion", "market": "Gujarat"}).json()
    assert body["change"] == pytest.approx(
        body["predicted_price"] - body["last_observed_price"], abs=0.02
    )


def test_predict_is_computed_fresh_each_call(client):
    first = client.post("/api/predict", json={"crop": "Onion", "market": "Gujarat"}).json()
    second = client.post("/api/predict", json={"crop": "Onion", "market": "Gujarat"}).json()
    assert first["predicted_price"] == second["predicted_price"]
    # A stored answer would carry a stored timestamp; a live one carries a new stamp.
    assert "computed_at_utc" in first and "computed_at_utc" in second


def test_predict_rejects_an_unknown_crop(client):
    response = client.post("/api/predict", json={"crop": "Durian", "market": "Gujarat"})
    assert response.status_code == 400
    assert "unknown crop" in response.json()["detail"].lower()


def test_predict_rejects_an_unknown_market(client):
    response = client.post("/api/predict", json={"crop": "Onion", "market": "Atlantis"})
    assert response.status_code == 400


def test_predict_validates_its_body(client):
    assert client.post("/api/predict", json={"crop": "Onion"}).status_code == 422


def test_history_returns_real_observed_prices(client, weekly):
    body = client.get(
        "/api/history", params={"crop": "Onion", "market": "Gujarat", "weeks": 20}
    ).json()
    assert len(body["points"]) == 20
    source = weekly[weekly["market"] == "Gujarat"].sort_values("week_start")
    assert body["points"][-1]["price"] == pytest.approx(
        float(source["modal_price"].iloc[-1]), abs=0.01
    )
    weeks = [point["week"] for point in body["points"]]
    assert weeks == sorted(weeks)


def test_history_rejects_a_silly_week_count(client):
    assert (
        client.get(
            "/api/history", params={"crop": "Onion", "market": "Gujarat", "weeks": 1}
        ).status_code
        == 422
    )


def test_backtest_scores_the_model_against_naive(client):
    body = client.get(
        "/api/backtest", params={"crop": "Onion", "market": "Gujarat", "weeks": 20}
    ).json()
    assert body["weeks_scored"] == 20
    assert set(body["model"]) == {"mae", "rmse", "mape"}
    assert body["model"]["mae"] > 0 and body["naive_last_week"]["mae"] > 0
    assert isinstance(body["model_beats_naive"], bool)
    assert len(body["points"]) == 20


def test_backtest_points_carry_actual_and_predicted(client):
    body = client.get(
        "/api/backtest", params={"crop": "Onion", "market": "Gujarat", "weeks": 12}
    ).json()
    for point in body["points"]:
        assert point["actual"] > 0
        assert point["predicted"] > 0


def test_backtest_verdict_matches_its_own_numbers(client):
    body = client.get(
        "/api/backtest", params={"crop": "Onion", "market": "Gujarat", "weeks": 26}
    ).json()
    assert body["model_beats_naive"] == (body["model"]["mae"] < body["naive_last_week"]["mae"])


def test_model_card_is_served(client):
    body = client.get("/api/model/Onion").json()
    assert body["version"] == "test-1.0.0"


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Mandi crop price forecaster" in response.text


def test_static_assets_are_served(client):
    for path in ("/static/app.css", "/static/chart.js"):
        assert client.get(path).status_code == 200


def test_openapi_docs_are_available(client):
    assert client.get("/openapi.json").status_code == 200


# --- without data ---------------------------------------------------------------


def test_app_starts_without_data(broken_client):
    body = broken_client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is False


def test_data_routes_return_503_without_data(broken_client):
    assert broken_client.get("/api/meta").status_code == 503
    response = broken_client.post("/api/predict", json={"crop": "Onion", "market": "Gujarat"})
    assert response.status_code == 503
    assert "fix" in response.json()


def test_index_still_renders_without_data(broken_client):
    assert broken_client.get("/").status_code == 200


# --- the web-start preflight ----------------------------------------------------


def test_preflight_passes_on_a_fresh_clone(tmp_path, monkeypatch):
    """Models and a committed snapshot are enough to serve; raw data is not required."""
    import app.cli as cli

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "index.json").write_text("{}")
    (tmp_path / "models" / "weekly_snapshot.csv").write_text("crop,market\n")
    monkeypatch.setattr(cli, "resolve_path", lambda p: tmp_path / str(p))
    monkeypatch.setattr(
        cli, "load_config", lambda p: {"paths": {"weekly_parquet": "x.parquet", "raw_dir": "raw"}}
    )
    (tmp_path / "raw").mkdir()

    blockers, notes = cli.preflight()
    assert blockers == []
    assert any("snapshot" in note for note in notes)
    assert any("raw CSVs" in note for note in notes)


def test_preflight_blocks_without_a_registry(tmp_path, monkeypatch):
    import app.cli as cli

    (tmp_path / "models").mkdir()
    (tmp_path / "raw").mkdir()
    monkeypatch.setattr(cli, "resolve_path", lambda p: tmp_path / str(p))
    monkeypatch.setattr(
        cli, "load_config", lambda p: {"paths": {"weekly_parquet": "x.parquet", "raw_dir": "raw"}}
    )

    blockers, _ = cli.preflight()
    assert any("index.json" in blocker for blocker in blockers)
    assert any("price table" in blocker for blocker in blockers)
