"""Tests for the Flask app: it starts, reports health, and returns real numbers."""

from __future__ import annotations

import pytest

from automl.features import build_features, feature_columns, split_xy
from automl.models.lgbm import default_params, fit

from .conftest import make_weekly_panel

FEATURES_CONFIG = {
    "target": {"column": "modal_price", "horizon_weeks": 1},
    "lags_weeks": [1, 2, 4, 8],
    "rolling_mean_windows_weeks": [4, 8],
    "calendar": {"month": True, "season": True},
    "categorical": ["market"],
}

CARD = {
    "version": "test-0.0.1",
    "git_commit": "abc1234",
    "created_utc": "2026-01-01T00:00:00+00:00",
    "entity_level": "state",
    "train_end": "2025-10-27",
    "metrics": {"mae": 123.4, "rmse": 234.5, "mape": 7.8},
}


@pytest.fixture
def trained_bits():
    """A fitted model plus the single serving row per market."""
    features = build_features(make_weekly_panel(n_weeks=160, markets=("A", "B")), FEATURES_CONFIG)
    x, y = split_xy(features)
    model = fit(default_params(), x, y, seed=0)
    serving = (
        features.sort_values("week_start")
        .groupby("market", observed=True)
        .tail(1)
        .reset_index(drop=True)
    )
    return model, serving


@pytest.fixture
def client(monkeypatch, trained_bits):
    """A test client for an app that has a model loaded."""
    import app.app as app_module

    model, serving = trained_bits
    monkeypatch.setattr(app_module, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(app_module, "load_card", lambda *a, **k: CARD)
    monkeypatch.setattr(app_module, "load_serving_features", lambda *a, **k: serving)
    flask_app = app_module.create_app()
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


@pytest.fixture
def modelless_client(monkeypatch):
    """A test client for an app that found no registered model."""
    import app.app as app_module

    def missing(*args, **kwargs):
        raise FileNotFoundError("models/model.pkl not found")

    monkeypatch.setattr(app_module, "load_model", missing)
    monkeypatch.setattr(app_module, "load_card", missing)
    monkeypatch.setattr(app_module, "load_serving_features", missing)
    flask_app = app_module.create_app()
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


# --- with a model ---------------------------------------------------------------


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    assert response.get_json()["model_loaded"] is True


def test_index_lists_the_markets(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "A" in body and "B" in body


def test_predict_returns_a_number(client):
    response = client.post("/predict", json={"market": "A"})
    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload["predicted_price_rs_per_quintal"], float)
    assert payload["market"] == "A"
    assert payload["for_week_starting"] > payload["based_on_week_starting"]


def test_predict_rejects_an_unknown_market(client):
    response = client.post("/predict", json={"market": "Atlantis"})
    assert response.status_code == 400
    assert "markets" in response.get_json()


def test_predict_requires_a_market(client):
    response = client.post("/predict", json={})
    assert response.status_code == 400


def test_model_card_is_served(client):
    response = client.get("/model")
    assert response.status_code == 200
    assert response.get_json()["version"] == "test-0.0.1"


def test_form_submission_renders_a_price(client):
    response = client.post("/", data={"market": "B"})
    assert response.status_code == 200
    assert "per quintal" in response.get_data(as_text=True)


def test_serving_row_carries_every_model_feature(trained_bits):
    model, serving = trained_bits
    assert set(model.feature_name_) == set(feature_columns(serving))


# --- without a model ------------------------------------------------------------


def test_app_starts_without_a_model(modelless_client):
    response = modelless_client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is False


def test_predict_is_unavailable_without_a_model(modelless_client):
    assert modelless_client.post("/predict", json={"market": "A"}).status_code == 503


def test_model_card_is_unavailable_without_a_model(modelless_client):
    assert modelless_client.get("/model").status_code == 503


def test_index_still_renders_without_a_model(modelless_client):
    assert modelless_client.get("/").status_code == 200
