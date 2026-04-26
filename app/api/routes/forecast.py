from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.data.dataset_store import DatasetStore
from app.engines.forecasting.forecast_orchestrator import ForecastOrchestrator
from app.engines.decisions.decision_engine import generate_decisions

router = APIRouter()

forecast_engine = ForecastOrchestrator()


class ForecastRequest(BaseModel):
    saved_filename: str
    schema_suggestions: Dict[str, Any]
    target_role: str = "revenue"
    forecast_horizon: int = 7


def resolve_date_column(schema_suggestions: Dict[str, Any]) -> Optional[str]:
    date_value = schema_suggestions.get("date")

    if isinstance(date_value, str) and date_value.strip():
        return date_value.strip()

    if isinstance(date_value, dict):
        column_name = date_value.get("column") or date_value.get("name")
        if isinstance(column_name, str) and column_name.strip():
            return column_name.strip()

    return None


def resolve_target_column(
    schema_suggestions: Dict[str, Any],
    target_role: str,
) -> Optional[str]:
    role_value = schema_suggestions.get(target_role)

    if isinstance(role_value, str) and role_value.strip():
        return role_value.strip()

    if isinstance(role_value, dict):
        column_name = role_value.get("column") or role_value.get("name")
        if isinstance(column_name, str) and column_name.strip():
            return column_name.strip()

    return None


def resolve_segment_column(schema_suggestions: Dict[str, Any]) -> Optional[str]:
    for key in ["product", "category", "region"]:
        value = schema_suggestions.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        if isinstance(value, dict):
            column_name = value.get("column") or value.get("name")
            if isinstance(column_name, str) and column_name.strip():
                return column_name.strip()

    return None


def _ensure_forecast_defaults(forecast_result: Dict[str, Any]) -> Dict[str, Any]:
    forecast_result = dict(forecast_result or {})

    forecast_result.setdefault("forecast_mode", "refuse_forecast")
    forecast_result.setdefault("model_used", None)
    forecast_result.setdefault("model_selection_reason", "")
    forecast_result.setdefault("forecast_values", [])
    forecast_result.setdefault("validation_metrics", {})
    forecast_result.setdefault("reliability_label", "Not Reliable")
    forecast_result.setdefault("reliability_score", 0.0)
    forecast_result.setdefault("warnings", [])
    forecast_result.setdefault("decision_usability_flag", "not_recommended")
    forecast_result.setdefault("when_to_trust", [])
    forecast_result.setdefault("when_not_to_trust", [])
    forecast_result.setdefault("data_quality_summary", {})
    forecast_result.setdefault("recommended_next_step", "")

    return forecast_result


def _ensure_decision_defaults(decisions_result: Dict[str, Any]) -> Dict[str, Any]:
    decisions_result = dict(decisions_result or {})
    decisions_result.setdefault("top_decisions", [])
    decisions_result.setdefault("summary", "No decision summary available.")
    decisions_result.setdefault("error", None)

    normalized = []
    for item in decisions_result["top_decisions"]:
        item = dict(item)
        item.setdefault("decision_type", "general")
        item.setdefault("title", "Untitled decision")
        item.setdefault("priority", "low")
        item.setdefault("risk_level", "normal")
        item.setdefault("recommendation", "")
        item.setdefault("rationale", "")
        item.setdefault("explanation", "")
        item.setdefault("evidence", {})
        normalized.append(item)

    decisions_result["top_decisions"] = normalized
    return decisions_result


def _build_dashboard_summary(
    forecast_result: Dict[str, Any],
    decisions_result: Dict[str, Any],
) -> Dict[str, Any]:
    top_decisions = decisions_result.get("top_decisions", [])
    high_priority_count = sum(
        1 for d in top_decisions
        if str(d.get("priority", "")).lower() == "high"
    )

    risk_counts: Dict[str, int] = {}
    for d in top_decisions:
        risk = str(d.get("risk_level", "normal"))
        risk_counts[risk] = risk_counts.get(risk, 0) + 1

    top_risk = max(risk_counts, key=risk_counts.get) if risk_counts else "normal"

    return {
        "forecast_mode": forecast_result.get("forecast_mode"),
        "model_used": forecast_result.get("model_used"),
        "reliability_label": forecast_result.get("reliability_label"),
        "reliability_score": forecast_result.get("reliability_score"),
        "decision_usability_flag": forecast_result.get("decision_usability_flag"),
        "decision_count": len(top_decisions),
        "high_priority_count": high_priority_count,
        "top_risk": top_risk,
    }


@router.post("/forecast")
def forecast_route(payload: ForecastRequest):
    try:
        dataset_store = DatasetStore()
        df = dataset_store.load_dataset(payload.saved_filename)
        df.columns = [str(col).strip() for col in df.columns]

        date_column = resolve_date_column(payload.schema_suggestions)
        if not date_column:
            raise ValueError(
                "Could not resolve date column from schema_suggestions. Expected schema_suggestions['date']."
            )

        target_column = resolve_target_column(
            schema_suggestions=payload.schema_suggestions,
            target_role=payload.target_role,
        )
        if not target_column:
            raise ValueError(
                f"Could not resolve target column for target_role='{payload.target_role}'."
            )

        segment_column = resolve_segment_column(payload.schema_suggestions)

        if date_column not in df.columns:
            raise ValueError(f"Resolved date column '{date_column}' not found in dataset.")

        if target_column not in df.columns:
            raise ValueError(f"Resolved target column '{target_column}' not found in dataset.")

        if segment_column and segment_column not in df.columns:
            segment_column = None

        raw_forecast_result = forecast_engine.run(
            df=df,
            date_column=date_column,
            target_column=target_column,
            forecast_horizon=payload.forecast_horizon,
        )

        raw_decisions_result = generate_decisions(
            df=df,
            schema_suggestions=payload.schema_suggestions,
            forecast_result=raw_forecast_result,
        )

        forecast_result = _ensure_forecast_defaults(raw_forecast_result)
        decisions_result = _ensure_decision_defaults(raw_decisions_result)
        dashboard_summary = _build_dashboard_summary(forecast_result, decisions_result)

        return {
            "message": "Forecast & decisions generated successfully",
            "forecast": forecast_result,
            "decisions": decisions_result,
            "dashboard_summary": dashboard_summary,
            "resolved_columns": {
                "date_column": date_column,
                "target_column": target_column,
                "segment_column": segment_column,
            },
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {str(e)}"
        )