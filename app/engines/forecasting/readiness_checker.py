from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from app.engines.forecasting.schemas import (
    DataQualitySummary,
    ReadinessIssue,
    ReadinessResult,
)


class ForecastReadinessChecker:
    """
    Evaluates whether a dataset is suitable for forecasting.

    This checker focuses on:
    - presence of required columns
    - row sufficiency
    - missing target values
    - duplicate timestamps
    - date continuity / frequency inference
    - basic target stability
    """

    def __init__(
        self,
        min_rows: int = 30,
        min_unique_periods: int = 20,
        max_missing_target_ratio: float = 0.2,
        max_duplicate_ratio: float = 0.1,
    ) -> None:
        self.min_rows = min_rows
        self.min_unique_periods = min_unique_periods
        self.max_missing_target_ratio = max_missing_target_ratio
        self.max_duplicate_ratio = max_duplicate_ratio

    def evaluate(
        self,
        df: pd.DataFrame,
        date_column: Optional[str],
        target_column: Optional[str],
    ) -> ReadinessResult:
        issues: List[ReadinessIssue] = []

        if df is None or df.empty:
            issues.append(
                ReadinessIssue(
                    code="empty_dataframe",
                    message="Dataset is empty. Forecasting cannot be performed.",
                    severity="critical",
                )
            )
            return self._build_failure_result(issues)

        if not date_column or date_column not in df.columns:
            issues.append(
                ReadinessIssue(
                    code="missing_date_column",
                    message="Valid date column is required for forecasting.",
                    severity="critical",
                )
            )
            return self._build_failure_result(issues)

        if not target_column or target_column not in df.columns:
            issues.append(
                ReadinessIssue(
                    code="missing_target_column",
                    message="Valid target column is required for forecasting.",
                    severity="critical",
                )
            )
            return self._build_failure_result(issues)

        working_df = df[[date_column, target_column]].copy()

        working_df[date_column] = pd.to_datetime(working_df[date_column], errors="coerce")
        working_df[target_column] = pd.to_numeric(working_df[target_column], errors="coerce")

        total_rows = len(working_df)
        usable_df = working_df.dropna(subset=[date_column, target_column]).copy()
        usable_rows = len(usable_df)

        if usable_rows == 0:
            issues.append(
                ReadinessIssue(
                    code="no_usable_rows",
                    message="No usable date-target rows remain after cleaning.",
                    severity="critical",
                )
            )
            return self._build_failure_result(issues)

        usable_df = usable_df.sort_values(by=date_column)

        unique_time_periods = int(usable_df[date_column].nunique())
        missing_target_ratio = float(working_df[target_column].isna().mean())

        duplicate_timestamp_count = int(usable_df.duplicated(subset=[date_column]).sum())
        duplicate_ratio = duplicate_timestamp_count / max(1, usable_rows)

        inferred_frequency = self._infer_frequency(usable_df[date_column])
        target_stability_score, stability_issue = self._assess_target_stability(usable_df[target_column])

        if total_rows < self.min_rows:
            issues.append(
                ReadinessIssue(
                    code="insufficient_rows",
                    message=f"Dataset has only {total_rows} rows; minimum recommended is {self.min_rows}.",
                    severity="critical",
                )
            )

        if unique_time_periods < self.min_unique_periods:
            issues.append(
                ReadinessIssue(
                    code="insufficient_time_periods",
                    message=(
                        f"Dataset has only {unique_time_periods} unique time periods; "
                        f"minimum recommended is {self.min_unique_periods}."
                    ),
                    severity="critical",
                )
            )

        if missing_target_ratio > self.max_missing_target_ratio:
            issues.append(
                ReadinessIssue(
                    code="high_missing_target_ratio",
                    message=(
                        f"Target column has missing ratio {missing_target_ratio:.2%}, "
                        f"which exceeds allowed threshold {self.max_missing_target_ratio:.2%}."
                    ),
                    severity="critical",
                )
            )

        if duplicate_ratio > self.max_duplicate_ratio:
            issues.append(
                ReadinessIssue(
                    code="high_duplicate_timestamp_ratio",
                    message=(
                        f"Duplicate timestamp ratio is {duplicate_ratio:.2%}, "
                        f"which exceeds allowed threshold {self.max_duplicate_ratio:.2%}."
                    ),
                    severity="warning",
                )
            )

        if inferred_frequency is None:
            issues.append(
                ReadinessIssue(
                    code="frequency_not_inferred",
                    message="Could not confidently infer time-series frequency.",
                    severity="warning",
                )
            )

        if stability_issue:
            issues.append(stability_issue)

        data_quality_summary = DataQualitySummary(
            total_rows=total_rows,
            usable_rows=usable_rows,
            date_column=date_column,
            target_column=target_column,
            unique_time_periods=unique_time_periods,
            missing_target_ratio=round(missing_target_ratio, 4),
            duplicate_timestamp_count=duplicate_timestamp_count,
            inferred_frequency=inferred_frequency,
            date_range_start=str(usable_df[date_column].min().date()) if not usable_df.empty else None,
            date_range_end=str(usable_df[date_column].max().date()) if not usable_df.empty else None,
        )

        readiness_score = self._calculate_readiness_score(
            total_rows=total_rows,
            unique_time_periods=unique_time_periods,
            missing_target_ratio=missing_target_ratio,
            duplicate_ratio=duplicate_ratio,
            target_stability_score=target_stability_score,
            issues=issues,
        )

        is_forecastable = not any(issue.severity == "critical" for issue in issues) and readiness_score >= 0.45

        if readiness_score >= 0.8:
            readiness_label = "High"
        elif readiness_score >= 0.6:
            readiness_label = "Medium"
        elif readiness_score >= 0.45:
            readiness_label = "Low"
        else:
            readiness_label = "Not Reliable"

        recommended_mode = "predictive_forecast" if is_forecastable else "refuse_forecast"

        return ReadinessResult(
            is_forecastable=is_forecastable,
            readiness_score=round(readiness_score, 4),
            readiness_label=readiness_label,
            recommended_mode=recommended_mode,
            issues=issues,
            data_quality_summary=data_quality_summary,
        )

    def _build_failure_result(self, issues: List[ReadinessIssue]) -> ReadinessResult:
        return ReadinessResult(
            is_forecastable=False,
            readiness_score=0.0,
            readiness_label="Not Reliable",
            recommended_mode="refuse_forecast",
            issues=issues,
            data_quality_summary=None,
        )

    def _infer_frequency(self, date_series: pd.Series) -> Optional[str]:
        if len(date_series) < 3:
            return None

        try:
            sorted_dates = pd.Series(pd.to_datetime(date_series).dropna().sort_values().unique())
            if len(sorted_dates) < 3:
                return None

            deltas = sorted_dates.diff().dropna()
            if deltas.empty:
                return None

            most_common_delta = deltas.mode()
            if most_common_delta.empty:
                return None

            days = most_common_delta.iloc[0] / pd.Timedelta(days=1)

            if days == 1:
                return "daily"
            if 6 <= days <= 8:
                return "weekly"
            if 27 <= days <= 31:
                return "monthly"

            return f"irregular_{days:.1f}_day_gap"
        except Exception:
            return None

    def _assess_target_stability(self, target_series: pd.Series) -> Tuple[float, Optional[ReadinessIssue]]:
        clean_series = pd.to_numeric(target_series, errors="coerce").dropna()

        if len(clean_series) < 5:
            return 0.3, ReadinessIssue(
                code="insufficient_target_observations",
                message="Target series has too few valid observations for stability analysis.",
                severity="warning",
            )

        mean_value = clean_series.mean()
        std_value = clean_series.std(ddof=0)

        if mean_value == 0:
            return 0.2, ReadinessIssue(
                code="zero_mean_target",
                message="Target series mean is zero, which makes error interpretation unreliable.",
                severity="warning",
            )

        coefficient_of_variation = abs(std_value / mean_value)

        if coefficient_of_variation < 0.25:
            return 1.0, None
        if coefficient_of_variation < 0.5:
            return 0.8, None
        if coefficient_of_variation < 1.0:
            return 0.6, ReadinessIssue(
                code="moderate_target_volatility",
                message="Target series shows moderate volatility.",
                severity="warning",
            )

        return 0.3, ReadinessIssue(
            code="high_target_volatility",
            message="Target series is highly volatile; forecast trust may be low.",
            severity="warning",
        )

    def _calculate_readiness_score(
        self,
        total_rows: int,
        unique_time_periods: int,
        missing_target_ratio: float,
        duplicate_ratio: float,
        target_stability_score: float,
        issues: List[ReadinessIssue],
    ) -> float:
        row_score = min(1.0, total_rows / max(self.min_rows * 2, 1))
        period_score = min(1.0, unique_time_periods / max(self.min_unique_periods * 2, 1))
        missing_score = max(0.0, 1.0 - (missing_target_ratio / max(self.max_missing_target_ratio, 1e-9)))
        duplicate_score = max(0.0, 1.0 - (duplicate_ratio / max(self.max_duplicate_ratio, 1e-9)))

        base_score = (
            0.25 * row_score +
            0.25 * period_score +
            0.20 * missing_score +
            0.15 * duplicate_score +
            0.15 * target_stability_score
        )

        critical_count = sum(1 for issue in issues if issue.severity == "critical")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")

        penalty = min(0.5, critical_count * 0.15 + warning_count * 0.04)

        return max(0.0, min(1.0, base_score - penalty))