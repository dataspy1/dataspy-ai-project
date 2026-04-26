import json
from fastapi import APIRouter, HTTPException

from app.schemas.mapping import (
    AutoMapResponse,
    ManualMappingRequest,
    ManualMappingResponse,
    ColumnRoleSuggestion,
)
from app.services.data.dataset_store import DatasetStore
from app.engines.mapping.schema_mapper import SchemaMapper

router = APIRouter(prefix="/api/mapping", tags=["Schema Mapping"])


@router.get("/auto/{filename}", response_model=AutoMapResponse)
def auto_map_columns(filename: str):
    try:
        df = DatasetStore.load_dataframe(filename)
        raw_suggestions = SchemaMapper.suggest_roles(df)
        role_map = SchemaMapper.role_to_column_map(raw_suggestions)

        suggestions = [ColumnRoleSuggestion(**item) for item in raw_suggestions]

        return AutoMapResponse(
            filename=filename,
            suggestions=suggestions,
            role_to_column=role_map,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto mapping failed: {str(e)}")


@router.post("/manual", response_model=ManualMappingResponse)
def apply_manual_mapping(payload: ManualMappingRequest):
    try:
        df = DatasetStore.load_dataframe(payload.filename)

        existing_columns = set(df.columns.tolist())
        invalid_columns = [v for v in payload.mappings.values() if v not in existing_columns]
        if invalid_columns:
            raise HTTPException(
                status_code=400,
                detail=f"These mapped columns do not exist in dataset: {invalid_columns}",
            )

        if payload.save_as_template and payload.template_name:
            path = DatasetStore.get_template_path(payload.template_name)
            path.write_text(json.dumps(payload.mappings, indent=2), encoding="utf-8")

        mapped_columns = set(payload.mappings.values())
        unmapped_columns = [col for col in df.columns if col not in mapped_columns]

        return ManualMappingResponse(
            filename=payload.filename,
            applied_mappings=payload.mappings,
            unmapped_columns=unmapped_columns,
            message="Manual mapping saved successfully.",
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual mapping failed: {str(e)}")