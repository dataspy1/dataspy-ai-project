from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.engines.mapping.schema_mapper import SchemaMapper
import pandas as pd
import os
import json

router = APIRouter()


class SchemaDetectRequest(BaseModel):
    columns: Optional[List[str]] = None
    saved_filename: Optional[str] = None


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


@router.post("/schema/detect")
def detect_schema_route(payload: SchemaDetectRequest):
    df = None
    columns = payload.columns

    if payload.saved_filename:
        file_path = os.path.join("backend/app/uploads", payload.saved_filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Uploaded file not found")

        df = load_dataframe(file_path)
        df.columns = [str(col).strip() for col in df.columns]
        columns = list(df.columns)

    if not columns:
        raise HTTPException(
            status_code=400,
            detail="Either columns or saved_filename must be provided"
        )

    # If only columns are passed and no dataframe is available,
    # create a dummy dataframe so SchemaMapper can still run.
    if df is None:
        df = pd.DataFrame(columns=columns)

    try:
        suggestions = SchemaMapper.suggest_roles(df)
        role_map = SchemaMapper.role_to_column_map(suggestions)

        return {
            "columns": columns,
            "suggestions": suggestions,
            "role_map": role_map
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema detection failed: {str(e)}")