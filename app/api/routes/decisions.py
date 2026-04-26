from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from app.engines.decisions.decision_engine import generate_decisions

router = APIRouter()


class DecisionRequest(BaseModel):
    capabilities: Dict[str, Any]
    insights: Dict[str, Any]
    forecast: Dict[str, Any]
    schema_suggestions: Dict[str, Any]


@router.post("/decisions")
def generate_decisions_route(payload: DecisionRequest):
    decision_output = generate_decisions(
        capabilities=payload.capabilities,
        insights=payload.insights,
        forecast=payload.forecast,
        schema_suggestions=payload.schema_suggestions
    )

    return {
        "message": "Decisions generated successfully",
        "result": decision_output
    }