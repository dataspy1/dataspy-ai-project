from pydantic import BaseModel, Field
from typing import Dict, Any, List


class ChatQueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User question related to dataset analysis"
    )
    
    analysis_context: Dict[str, Any] = Field(
        ...,
        description="Output from /api/analyze endpoint"
    )


class ChatQueryResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Final answer returned to the user"
    )
    
    context_used: List[str] = Field(
        default_factory=list,
        description="List of context elements used to generate the answer"
    )
    
    answer_source: str = Field(
        ...,
        description="Source of answer: Direct From Data or Data Through LLM"
    )