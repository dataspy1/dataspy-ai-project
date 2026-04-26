from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ReadinessIssue:
    code: str
    message: str
    severity: str  # info / warning / critical

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualitySummary:
    total_rows: int
    usable_rows: int
    date_column: Optional[str]
    target_column: Optional[str]
    unique_time_periods: int
    missing_target_ratio: float
    duplicate_timestamp_count: int
    inferred_frequency: Optional[str]
    date_range_start: Optional[str]
    date_range_end: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReadinessResult:
    is_forecastable: bool
    readiness_score: float
    readiness_label: str
    recommended_mode: str
    issues: List[ReadinessIssue] = field(default_factory=list)
    data_quality_summary: Optional[DataQualitySummary] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_forecastable": self.is_forecastable,
            "readiness_score": self.readiness_score,
            "readiness_label": self.readiness_label,
            "recommended_mode": self.recommended_mode,
            "issues": [issue.to_dict() for issue in self.issues],
            "data_quality_summary": (
                self.data_quality_summary.to_dict()
                if self.data_quality_summary else None
            ),
        }


@dataclass
class ValidationMetrics:
    mae: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    smape: Optional[float]
    wape: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FoldResult:
    fold_number: int
    train_size: int
    test_size: int
    metrics: ValidationMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold_number": self.fold_number,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class ValidationResult:
    model_name: str
    fold_results: List[FoldResult]
    aggregate_metrics: ValidationMetrics
    is_stable: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "fold_results": [fold.to_dict() for fold in self.fold_results],
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
            "is_stable": self.is_stable,
            "notes": self.notes,
        }


@dataclass
class CandidateModelDiagnostic:
    model_name: str
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    smape: Optional[float] = None
    wape: Optional[float] = None
    is_stable: bool = False
    selected: bool = False
    is_baseline: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastSummary:
    selected_model: Optional[str]
    selected_model_type: Optional[str]
    confidence_band_note: Optional[str]
    pattern_detected: Optional[str]
    validation_summary: Optional[str]
    seasonality_detected: Optional[bool] = None
    seasonality_strength: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastResponse:
    forecast_mode: str
    model_used: Optional[str]
    model_selection_reason: str
    forecast_values: List[Dict[str, Any]]
    validation_metrics: Dict[str, Any]
    reliability_label: str
    reliability_score: float
    warnings: List[str]
    decision_usability_flag: str
    when_to_trust: List[str]
    when_not_to_trust: List[str]
    data_quality_summary: Dict[str, Any]
    recommended_next_step: str

    history_metadata: Dict[str, Any] = field(default_factory=dict)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)

    forecast_summary: Dict[str, Any] = field(default_factory=dict)
    candidate_diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    debug_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)