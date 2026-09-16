"""Stage 4 - FastAPI request/response schemas (Pydantic v2).

Field aliases match the dataset column names EXACTLY ("fixed acidity", ...),
so the request JSON mirrors the training-time schema. Robust bounds are
physical, not data-specific: the API must not reject legitimate unseen wines,
but must reject garbage (negative concentrations, pH outside [0, 14],
free SO2 > total SO2, unknown wine types).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WineFeatures(BaseModel):
    """One wine sample: 11 numeric features + wine_type."""

    model_config = ConfigDict(populate_by_name=True)

    fixed_acidity: float = Field(..., alias="fixed acidity", ge=0, le=30,
                                 description="g/L (tartaric acid)")
    volatile_acidity: float = Field(..., alias="volatile acidity", ge=0, le=5)
    citric_acid: float = Field(..., alias="citric acid", ge=0, le=5)
    residual_sugar: float = Field(..., alias="residual sugar", ge=0, le=100)
    chlorides: float = Field(..., ge=0, le=10)
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide", ge=0, le=300)
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide", ge=0, le=500)
    density: float = Field(..., ge=0.9, le=1.1)
    ph: float = Field(..., alias="pH", ge=0, le=14)
    sulphates: float = Field(..., ge=0, le=5)
    alcohol: float = Field(..., ge=0, le=100, description="% vol")
    wine_type: Literal["red", "white"]

    @model_validator(mode="after")
    def free_le_total(self) -> "WineFeatures":
        if self.free_sulfur_dioxide > self.total_sulfur_dioxide:
            raise ValueError("free sulfur dioxide cannot exceed total sulfur dioxide")
        return self


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = Good (quality > 5), 0 = Poor", ge=0, le=1)
    label: Literal["Good", "Poor"]
    probability_good: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_path: str
    model_version: str
    model_type: str | None = None
