"""Flask application serving next week's onion price forecast.

Routes:

``GET /``
    A form listing the available markets. Submitting it shows the forecast.
``POST /predict``
    JSON API. Send ``{"market": "Maharashtra"}``, receive the predicted price.
``GET /health``
    Liveness probe for the deployment pipeline. Always cheap, never touches the model.
``GET /model``
    The model card, so a caller can tell which model answered them.

The app loads the model once at start-up. If no model has been registered, it still
starts and ``/health`` still returns 200 -- the deployment should come up and report
honestly rather than crash-loop -- but prediction routes return 503.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from flask import Flask, jsonify, render_template, request

from automl.features import feature_columns, target_mode_of, to_price_level
from automl.registry import load_card, load_model, load_serving_features

LOGGER = logging.getLogger(__name__)

MODEL_UNAVAILABLE = {
    "error": "no model registered",
    "hint": "run: uv run python -m automl.registry",
}


def create_app() -> Flask:
    """Build the Flask application.

    Returns:
        The configured app, with the model loaded if one is available.
    """
    app = Flask(__name__)
    app.config["MODEL"] = None
    app.config["CARD"] = None
    app.config["SERVING"] = None

    try:
        app.config["MODEL"] = load_model()
        app.config["CARD"] = load_card()
        app.config["SERVING"] = load_serving_features()
        LOGGER.info("model loaded: version %s", app.config["CARD"].get("version"))
    except FileNotFoundError as error:
        LOGGER.warning("starting without a model: %s", error)

    def markets() -> list[str]:
        """List the markets the loaded model can serve.

        Returns:
            Sorted market names, empty when no model is loaded.
        """
        serving = app.config["SERVING"]
        if serving is None:
            return []
        return sorted(str(name) for name in serving["market"].unique())

    def predict_for(market: str) -> dict[str, Any]:
        """Predict next week's price for one market.

        Args:
            market: The market name.

        Returns:
            A mapping with the prediction and the context it was made in.

        Raises:
            KeyError: If the market is unknown to the loaded model.
        """
        serving: pd.DataFrame = app.config["SERVING"]
        row = serving[serving["market"].astype(str) == market]
        if row.empty:
            raise KeyError(market)
        columns = feature_columns(serving)
        raw = app.config["MODEL"].predict(row[columns])
        # In delta mode the model outputs a week-on-week change, which has to be
        # added back onto last week's price before anyone sees it.
        prediction = float(to_price_level(row, raw)[0])
        forecast_week = pd.Timestamp(row["week_start"].iloc[0])
        return {
            "market": market,
            "predicted_price_rs_per_quintal": round(prediction, 2),
            "last_observed_price_rs_per_quintal": round(float(row["lag_1"].iloc[0]), 2),
            "target_mode": target_mode_of(serving),
            "for_week_starting": str(forecast_week.date()),
            "based_on_week_starting": str((forecast_week - pd.Timedelta(weeks=1)).date()),
            "model_version": (app.config["CARD"] or {}).get("version"),
        }

    @app.get("/health")
    def health():
        """Liveness probe.

        Returns:
            A JSON body reporting service status and whether a model is loaded.
        """
        return jsonify({"status": "ok", "model_loaded": app.config["MODEL"] is not None}), 200

    @app.get("/model")
    def model_card():
        """Expose the model card.

        Returns:
            The card as JSON, or 503 if no model is registered.
        """
        if app.config["CARD"] is None:
            return jsonify(MODEL_UNAVAILABLE), 503
        return jsonify(app.config["CARD"]), 200

    @app.post("/predict")
    def predict():
        """JSON prediction endpoint.

        Returns:
            The forecast as JSON, 400 for a missing or unknown market, or 503 when
            no model is registered.
        """
        if app.config["MODEL"] is None:
            return jsonify(MODEL_UNAVAILABLE), 503
        payload = request.get_json(silent=True) or {}
        market = payload.get("market") or request.form.get("market")
        if not market:
            return jsonify({"error": "field 'market' is required", "markets": markets()}), 400
        try:
            return jsonify(predict_for(str(market))), 200
        except KeyError:
            return jsonify({"error": f"unknown market {market!r}", "markets": markets()}), 400

    @app.route("/", methods=["GET", "POST"])
    def index():
        """Render the market picker and, after submission, the forecast.

        Returns:
            The rendered HTML page.
        """
        result = None
        error = None
        selected = request.form.get("market") if request.method == "POST" else None
        if request.method == "POST":
            if app.config["MODEL"] is None:
                error = "No model has been registered yet."
            elif not selected:
                error = "Please choose a market."
            else:
                try:
                    result = predict_for(selected)
                except KeyError:
                    error = f"Unknown market: {selected}"
        return render_template(
            "index.html",
            markets=markets(),
            selected=selected,
            result=result,
            error=error,
            card=app.config["CARD"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
