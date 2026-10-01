"""Request and response schemas for the taxi-duration prediction API."""

from pydantic import BaseModel, Field


class TaxiRide(BaseModel):
    """One taxi ride, with the same feature columns the model was trained on."""

    VendorID: int = Field(examples=[1])
    trip_distance: float = Field(ge=0, examples=[1.6])
    PULocationID: int = Field(examples=[229])
    DOLocationID: int = Field(examples=[237])
    fare_amount: float = Field(ge=0, examples=[10.0])
    extra: float = Field(examples=[3.5])
    mta_tax: float = Field(examples=[0.5])
    tip_amount: float = Field(examples=[3.0])
    tolls_amount: float = Field(examples=[0.0])
    improvement_surcharge: float = Field(examples=[1.0])
    congestion_surcharge: float = Field(examples=[2.5])
    Airport_fee: float = Field(examples=[0.0])
    cbd_congestion_fee: float = Field(examples=[0.0])


class TaxiRidePrediction(TaxiRide):
    """The ride that was sent in, plus the predicted duration in minutes."""

    predicted_duration: float
