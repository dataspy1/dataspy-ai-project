from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


AllowedRole = Literal[
    "date",
    "product",
    "region",
    "revenue",
    "quantity",
    "order_id",
    "delivery_date",
    "shipment_date",
    "status",
    "stock",
    "reorder_level",
    "lead_time",
    "unknown",
]


class ColumnRoleSuggestion(BaseModel):
    column_name: str
    detected_role: AllowedRole
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = []


class AutoMapResponse(BaseModel):
    filename: str
    suggestions: List[ColumnRoleSuggestion]
    role_to_column: Dict[str, Optional[str]]


class ManualMappingRequest(BaseModel):
    filename: str
    mappings: Dict[str, str]
    save_as_template: bool = False
    template_name: Optional[str] = None


class ManualMappingResponse(BaseModel):
    filename: str
    applied_mappings: Dict[str, str]
    unmapped_columns: List[str]
    message: str