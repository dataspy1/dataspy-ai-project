from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from app.engines.insights.narrative_engine import generate_narrative

router = APIRouter()


class NarrativeRequest(BaseModel):
    schema_suggestions: Dict[str, Any]
    capabilities: Dict[str, Any]
    profile: Dict[str, Any]
    insights: Dict[str, Any]


@router.post("/narrative")
def generate_narrative_route(payload: NarrativeRequest):
    narratives = generate_narrative(
        schema_suggestions=payload.schema_suggestions,
        capabilities=payload.capabilities,
        profile=payload.profile,
        insights=payload.insights
    )

    return {
        "message": "Narrative summary generated successfully",
        "narratives": narratives
    }