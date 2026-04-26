from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.engines.understanding.profiler import profile_dataframe
import pandas as pd
import os
import json

router = APIRouter()


class ProfileRequest(BaseModel):
    saved_filename: str


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


@router.post("/profile")
def profile_dataset(payload: ProfileRequest):
    file_path = os.path.join("backend/app/uploads", payload.saved_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    try:
        df = load_dataframe(file_path)
        df.columns = [str(col).strip() for col in df.columns]

        profile = profile_dataframe(df)

        return {
            "message": "Dataset profiling completed",
            "profile": profile
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")