"""Helpers for loading the registered MLflow model and generating predictions."""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_features(ride):
    """Transform the API payload into the feature columns expected by the model."""
    # The random forest in run_model.ipynb trains on the cleaned columns as they are,
    # so every field of the request is one feature column.
    return ride.model_dump()


@lru_cache
def load_model(model_name, alias):
    """Resolve a model by name and alias from the local MLflow Model Registry."""
    import mlflow

    # Aliases keep the API stable while the underlying registered version changes.
    # The cache loads the model once per process instead of on every request;
    # restart the API after moving the alias to a new version.
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def configure_mlflow():
    """Load local settings and point MLflow at the configured tracking server."""
    import mlflow

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    model_name = os.getenv("REGISTERED_MODEL_NAME", "rf-taxi-duration")
    model_alias = os.getenv("MODEL_ALIAS", "production")

    # Point MLflow at the local tracking server used throughout this repo.
    mlflow.set_tracking_uri(tracking_uri)
    return model_name, model_alias


def predict(data):
    """Read local configuration, load the production model, and score one ride."""
    model_name, model_alias = configure_mlflow()

    # MLflow pyfunc models usually expect tabular input, even for a single request.
    model_input = pd.DataFrame([prepare_features(data)])
    model = load_model(model_name, model_alias)
    prediction = model.predict(model_input)
    return float(prediction[0])


def predict_batch(rides):
    """Score a list of rides with one model load and one tabular prediction call."""
    model_name, model_alias = configure_mlflow()

    # Batch prediction keeps request order by building one feature row per ride.
    model_input = pd.DataFrame([prepare_features(ride) for ride in rides])
    model = load_model(model_name, model_alias)
    predictions = model.predict(model_input)
    return [float(prediction) for prediction in predictions]
