from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


# =========================
# 🔹 REQUEST
# =========================
class ForecastV2Request(BaseModel):
    saved_filename: str
    date_column: str
    target_column: str
    horizon: int = 7
    schema_mapping: Optional[Dict[str, str]] = None
    segment_column: Optional[str] = None   # 🔥 NEW


# =========================
# 🔹 BASE POINTS
# =========================
class ForecastPoint(BaseModel):
    date: str
    value: float


class FutureForecastPoint(BaseModel):
    date: str
    predicted: float


class ValidationPoint(BaseModel):
    date: str
    actual: float
    predicted: float


# =========================
# 🔹 SEGMENT STRUCTURE
# =========================
class SegmentForecast(BaseModel):
    segment: str
    model_used: Optional[str]
    future_forecast: List[Dict[str, Any]]
    metrics: Optional[Dict[str, float]]
    reliability: Optional[str]


# =========================
# 🔹 RESPONSE
# =========================
class ForecastV2Response(BaseModel):
    forecast_version: str

    # 🔹 Common fields
    target_column: Optional[str] = None
    date_column: Optional[str] = None
    forecast_horizon: Optional[int] = None

    # 🔹 Model info
    model_used: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None

    # 🔹 Time-series outputs
    historical_points: Optional[List[Dict[str, Any]]] = None
    validation_points: Optional[List[Dict[str, Any]]] = None
    future_forecast: Optional[List[Dict[str, Any]]] = None  # 🔥 NEW

    # 🔹 Feature info
    feature_summary: Optional[Dict[str, Any]] = None

    # 🔹 Warnings & reliability
    warnings: Optional[List[str]] = None
    reliability_label: Optional[str] = None

    # 🔹 Summary
    summary: Optional[str] = None

    # =========================
    # 🔥 SEGMENTED OUTPUT
    # =========================
    segment_column: Optional[str] = None
    segments_count: Optional[int] = None
    segments: Optional[List[SegmentForecast]] = None