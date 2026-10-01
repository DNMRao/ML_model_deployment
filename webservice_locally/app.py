"""FastAPI entrypoint for serving local taxi-duration predictions."""

# Run this module with Uvicorn after the notebook has registered a model in the
# local MLflow Model Registry. The route functions stay intentionally small so
# you can see where validation ends and prediction logic begins.

from fastapi import FastAPI

from .data_model import TaxiRide, TaxiRidePrediction
from .predict import predict, predict_batch

app = FastAPI(
    title="Local Taxi Duration Prediction API",
    description=(
        "A local-first FastAPI service that loads a registered MLflow model "
        "and returns taxi ride duration predictions."
    ),
    version="1.0.0",
)


@app.get("/", tags=["health"])
def index():
    """Return a simple message so we can verify the API is running."""
    return {"message": "NYC Taxi Ride Duration Prediction"}


@app.post("/predict", response_model=TaxiRidePrediction, tags=["predictions"])
def predict_duration(data: TaxiRide):
    """Load the current production model from MLflow and return one prediction."""
    prediction = predict(data)
    return TaxiRidePrediction(**data.model_dump(), predicted_duration=prediction)


@app.post(
    "/predict_batch", response_model=list[TaxiRidePrediction], tags=["predictions"]
)
def predict_duration_batch(data: list[TaxiRide]):
    """Return one prediction for each ride in the request body."""
    if not data:
        # Empty batches are valid: there is nothing to score and no model load needed.
        return []

    predictions = predict_batch(data)
    return [
        TaxiRidePrediction(**ride.model_dump(), predicted_duration=prediction)
        for ride, prediction in zip(data, predictions)
    ]
