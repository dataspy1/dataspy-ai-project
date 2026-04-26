from typing import Dict, List, Literal, Optional
from pydantic import BaseModel


MissingStrategy = Literal[
    "mean",
    "median",
    "mode",
    "forward_fill",
    "backward_fill",
    "interpolate",
    "drop_rows",
    "drop_column",
    "unknown_label",
    "zero_fill",
]


class MissingColumnProfile(BaseModel):
    column_name: str
    dtype: str
    missing_count: int
    missing_percent: float
    suggested_strategies: List[str]


class MissingProfileResponse(BaseModel):
    filename: str
    total_rows: int
    total_columns: int
    columns: List[MissingColumnProfile]


class MissingHandlingRequest(BaseModel):
    filename: str
    strategies: Dict[str, MissingStrategy]
    create_new_clean_file: bool = True
    cleaned_filename: Optional[str] = None


class AppliedTransformation(BaseModel):
    column_name: str
    strategy: str
    affected_rows: int
    note: str


class MissingHandlingResponse(BaseModel):
    original_filename: str
    cleaned_filename: str
    transformations: List[AppliedTransformation]
    message: str