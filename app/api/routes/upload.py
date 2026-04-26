from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import os
import json
from uuid import uuid4

router = APIRouter()

UPLOAD_DIR = "backend/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_dataframe(file_path: str, extension: str) -> pd.DataFrame:
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


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name missing")

    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in [".csv", ".xlsx", ".xls", ".json"]:
        raise HTTPException(
            status_code=400,
            detail="Only CSV, Excel, and JSON files are supported"
        )

    saved_filename = f"{uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        df = load_dataframe(file_path, extension)
        df.columns = [str(col).strip() for col in df.columns]

        return {
            "message": "File uploaded successfully",
            "original_filename": file.filename,
            "saved_filename": saved_filename,
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "preview": df.head(5).fillna("").to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

