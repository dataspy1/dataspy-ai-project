from typing import Dict, Any
from fastapi import APIRouter
from pydantic import BaseModel

from app.engines.understanding.capability_detector import detect_capabilities

router = APIRouter()


class CapabilityRequest(BaseModel):
    suggestions: Dict[str, Any]


@router.post("/detect-capabilities")
def detect_capabilities_api(request: CapabilityRequest):
    print("REQUEST MODEL:", request.model_dump())
    print("REQUEST SUGGESTIONS:", request.suggestions)

    capabilities = detect_capabilities(request.suggestions)

    return {
        "message": "Capability detection completed",
        "capabilities": capabilities
    }