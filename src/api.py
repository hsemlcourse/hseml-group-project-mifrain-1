from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prediction import CarPricePredictor


app = FastAPI(
    title="Business Class Car Price API",
    description="API for predicting business-class-and-above car prices in EGP.",
    version="1.0.0",
)


class CarPriceRequest(BaseModel):
    dataset_part: Literal["used", "new"] = Field(default="used", description="Market segment from the source dataset.")
    brand: str = Field(default="BMW", min_length=1)
    model: str = Field(default="X5", min_length=1)
    trim_or_variant: str = Field(default="BMW X5 2020 A/T Turbo SUV")
    model_year: int = Field(default=2020, ge=1900, le=2035)
    snapshot_date: date = Field(default_factory=date.today)

    def to_record(self) -> dict[str, object]:
        return {
            "dataset_part": self.dataset_part,
            "brand": self.brand.strip(),
            "model": self.model.strip(),
            "trim_or_variant": self.trim_or_variant.strip(),
            "model_year": self.model_year,
            "snapshot_date": self.snapshot_date.isoformat(),
        }


class BatchCarPriceRequest(BaseModel):
    items: list[CarPriceRequest] = Field(min_length=1, max_length=100)


class CarPriceResponse(BaseModel):
    predicted_price_egp: float
    model_name: str
    primary_metric: str


@lru_cache(maxsize=1)
def get_predictor() -> CarPricePredictor:
    return CarPricePredictor()


@app.get("/health")
def health() -> dict[str, object]:
    try:
        metadata = get_predictor().metadata()
    except Exception as exc:  # pragma: no cover - used for deployment diagnostics
        return {"status": "error", "model_loaded": False, "detail": str(exc)}
    return {"status": "ok", "model_loaded": True, "model_name": metadata["model_name"]}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    try:
        return get_predictor().metadata()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=CarPriceResponse)
def predict(payload: CarPriceRequest) -> CarPriceResponse:
    try:
        prediction = get_predictor().predict_one(payload.to_record())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CarPriceResponse(**prediction.__dict__)


@app.post("/predict/batch", response_model=list[CarPriceResponse])
def predict_batch(payload: BatchCarPriceRequest) -> list[CarPriceResponse]:
    try:
        predictions = get_predictor().predict_many([item.to_record() for item in payload.items])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [CarPriceResponse(**prediction.__dict__) for prediction in predictions]
