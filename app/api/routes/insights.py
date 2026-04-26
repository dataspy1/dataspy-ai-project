from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from app.engines.insights.insight_engine import generate_insights
import pandas as pd
import os
import json

router = APIRouter()


class InsightsRequest(BaseModel):
    saved_filename: str
    schema_suggestions: Dict[str, Any]
    capabilities: Dict[str, Any]


def load_dataframe(file_path: str) -> pd.DataFrame:
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".csv":
        return pd.read_csv(file_path)
    elif extension in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif extension == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format")


@router.post("/insights")
def get_insights(payload: InsightsRequest):
    file_path = os.path.join("backend/app/uploads", payload.saved_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    try:
        df = load_dataframe(file_path)
        df.columns = [str(col).strip() for col in df.columns]

        insights = generate_insights(
            df=df,
            schema_suggestions=payload.schema_suggestions,
            capabilities=payload.capabilities
        )

        return {
            "message": "Insights generated successfully",
            "insights": insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")