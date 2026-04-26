from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExplainAnalysisRequest(BaseModel):
    analysis_context: Dict[str, Any] = Field(..., description="Structured output from /api/analyze")
    audience: Optional[str] = Field(default="management")
    tone: Optional[str] = Field(default="executive")


class ExplainAnalysisResponse(BaseModel):
    executive_summary: str
    business_explanation: str
    recommended_next_steps: List[str]
    risk_summary: List[str]
    model_used: str