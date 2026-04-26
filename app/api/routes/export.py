from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict, List
from pathlib import Path
import pandas as pd
import json
from uuid import uuid4

# ✅ NEW IMPORTS (PDF)
from app.services.reports.forecast_report_builder import build_forecast_pdf
from app.services.reports.decision_report_builder import build_decision_pdf


router = APIRouter(prefix="/export", tags=["Export"])

EXPORT_DIR = Path(__file__).resolve().parents[2] / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class DecisionsExportRequest(BaseModel):
    decisions: List[Dict[str, Any]]


class ForecastExportRequest(BaseModel):
    forecast: Dict[str, Any]


# =========================
# CSV EXPORTS (EXISTING)
# =========================

@router.post("/decisions-csv")
async def export_decisions_csv(payload: DecisionsExportRequest):
    try:
        if not payload.decisions:
            raise HTTPException(status_code=400, detail="No decisions found to export")

        rows = []
        for idx, decision in enumerate(payload.decisions, start=1):
            rows.append({
                "sr_no": idx,
                "priority": decision.get("priority"),
                "action": decision.get("action"),
                "rationale": decision.get("rationale"),
                "decision_type": decision.get("decision_type"),
                "rank_order": decision.get("rank_order"),
                "supporting_evidence": json.dumps(
                    decision.get("supporting_evidence", {}),
                    ensure_ascii=False
                )
            })

        df = pd.DataFrame(rows)

        filename = f"decision_report_{uuid4().hex[:8]}.csv"
        file_path = EXPORT_DIR / filename
        df.to_csv(file_path, index=False)

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="text/csv"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Decision CSV export failed: {str(e)}"
        )


@router.post("/forecast-csv")
async def export_forecast_csv(payload: ForecastExportRequest):
    try:
        forecast = payload.forecast or {}
        historical_series = forecast.get("historical_series", [])
        forecast_series = forecast.get("forecast_series", [])

        if not historical_series and not forecast_series:
            raise HTTPException(status_code=400, detail="No forecast data found to export")

        rows = []

        for row in historical_series:
            rows.append({
                "series_type": "historical",
                "date": row.get("date"),
                "value": row.get("value")
            })

        for row in forecast_series:
            rows.append({
                "series_type": "forecast",
                "date": row.get("date"),
                "value": row.get("value")
            })

        df = pd.DataFrame(rows)

        filename = f"forecast_report_{uuid4().hex[:8]}.csv"
        file_path = EXPORT_DIR / filename
        df.to_csv(file_path, index=False)

        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="text/csv"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast CSV export failed: {str(e)}"
        )


# =========================
# PDF EXPORTS (NEW)
# =========================

@router.post("/decisions-pdf")
async def export_decisions_pdf(payload: DecisionsExportRequest):
    try:
        if not payload.decisions:
            raise HTTPException(status_code=400, detail="No decisions found to export")

        pdf_path = build_decision_pdf(payload.decisions)

        return FileResponse(
            path=str(pdf_path),
            filename=Path(pdf_path).name,
            media_type="application/pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Decision PDF export failed: {str(e)}"
        )


@router.post("/forecast-pdf")
async def export_forecast_pdf(payload: ForecastExportRequest):
    try:
        forecast = payload.forecast or {}

        print("RAW FORECAST PDF PAYLOAD:", forecast)

        normalized_forecast = {
            "forecast_target_role": (
                forecast.get("forecast_target_role")
                or forecast.get("target_role")
                or forecast.get("target_column")
                or forecast.get("metric")
                or forecast.get("forecast_target")
            ),
            "forecast_horizon": (
                forecast.get("forecast_horizon")
                or forecast.get("horizon")
                or forecast.get("periods")
            ),
            "model_used": (
                forecast.get("model_used")
                or forecast.get("selected_model")
                or forecast.get("model_name")
            ),
            "reliability_label": (
                forecast.get("reliability_label")
                or (
                    forecast.get("reliability", {}).get("label")
                    if isinstance(forecast.get("reliability"), dict)
                    else None
                )
            ),
            "warnings": forecast.get("warnings", []),
            "historical_series": (
                forecast.get("historical_series")
                or forecast.get("history")
                or forecast.get("historical")
                or forecast.get("actual_series")
                or []
            ),
            "forecast_series": (
                forecast.get("forecast_series")
                or forecast.get("predictions")
                or forecast.get("future_series")
                or forecast.get("forecast")
                or []
            ),
            "trend": forecast.get("trend"),
            "growth_percent": forecast.get("growth_percent"),
        }

        if not normalized_forecast["historical_series"] and not normalized_forecast["forecast_series"]:
            raise HTTPException(status_code=400, detail="No forecast data found to export")

        print("NORMALIZED FORECAST PDF PAYLOAD:", normalized_forecast)

        pdf_path = build_forecast_pdf(normalized_forecast)

        return FileResponse(
            path=str(pdf_path),
            filename=Path(pdf_path).name,
            media_type="application/pdf"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast PDF export failed: {str(e)}"
        )