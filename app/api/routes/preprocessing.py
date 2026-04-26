from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.schemas.preprocessing import (
    MissingProfileResponse,
    MissingColumnProfile,
    MissingHandlingRequest,
    MissingHandlingResponse,
    AppliedTransformation,
)
from app.services.data.dataset_store import DatasetStore
from app.engines.preprocessing.missing_handler import MissingHandler

router = APIRouter(prefix="/api/preprocessing", tags=["Preprocessing"])


@router.get("/missing-profile/{filename}", response_model=MissingProfileResponse)
def get_missing_profile(filename: str):
    try:
        df = DatasetStore.load_dataframe(filename)
        profiles = MissingHandler.profile_missing(df)

        return MissingProfileResponse(
            filename=filename,
            total_rows=len(df),
            total_columns=len(df.columns),
            columns=[MissingColumnProfile(**item) for item in profiles],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing profile failed: {str(e)}")


@router.post("/apply-missing-strategies", response_model=MissingHandlingResponse)
def apply_missing_strategies(payload: MissingHandlingRequest):
    try:
        df = DatasetStore.load_dataframe(payload.filename)
        cleaned_df, logs = MissingHandler.apply_strategies(df, payload.strategies)

        original_path = Path(payload.filename)
        cleaned_name = payload.cleaned_filename

        if not cleaned_name:
            cleaned_name = f"{original_path.stem}_cleaned{original_path.suffix}"

        saved_filename = DatasetStore.save_dataframe(cleaned_df, cleaned_name, processed=True)

        return MissingHandlingResponse(
            original_filename=payload.filename,
            cleaned_filename=saved_filename,
            transformations=[AppliedTransformation(**log) for log in logs],
            message="Missing-data strategies applied and cleaned dataset saved successfully.",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing strategy application failed: {str(e)}")