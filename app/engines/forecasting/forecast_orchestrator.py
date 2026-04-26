from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.engines.forecasting.readiness_checker import ForecastReadinessChecker
from app.engines.forecasting.schemas import ForecastResponse
from app.engines.forecasting.validator import TimeSeriesValidator
from app.engines.forecasting.ml_models import XGBoostForecastModel


class ForecastOrchestrator:
    """
    Forecast orchestrator with safer business-missing handling.

    Purpose:
    - stop flat repeated-value forecasts
    - prioritize weekly seasonal behavior when present
    - expose diagnostics for debugging model selection
    - avoid corrupting business-valid missing dates like pending delivery dates
    """

    def __init__(self) -> None:
        self.readiness_checker = ForecastReadinessChecker()
        self.validator = TimeSeriesValidator()
        self.xgb_model: Optional[XGBoostForecastModel] = None

        try:
            self.xgb_model = XGBoostForecastModel()
        except Exception:
            self.xgb_model = None

    def run(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        forecast_horizon: int = 7,
    ) -> Dict[str, Any]:
        working_df = df.copy()

        # Global cleanup first
        working_df = self._clean_general_columns(working_df)

        # Business-missing safety
        working_df, preprocessing_warnings, preprocessing_meta = self._apply_business_missing_safety(
            df=working_df,
            date_column=date_column,
            target_column=target_column,
        )

        readiness = self.readiness_checker.evaluate(
            df=working_df,
            date_column=date_column,
            target_column=target_column,
        )

        if not readiness.is_forecastable:
            return self._build_response(
                forecast_mode="refuse_forecast",
                model_used=None,
                model_selection_reason="Forecast generation was refused because dataset failed readiness checks.",
                forecast_values=[],
                validation_metrics={},
                baseline_comparison={},
                reliability_label=readiness.readiness_label,
                reliability_score=readiness.readiness_score,
                warnings=preprocessing_warnings + [issue.message for issue in readiness.issues],
                decision_usability_flag="not_recommended",
                when_to_trust=[],
                when_not_to_trust=[
                    "Data quality is insufficient for reliable forecasting.",
                    "Historical continuity or target quality is too weak.",
                ],
                data_quality_summary=(
                    readiness.data_quality_summary.to_dict()
                    if readiness.data_quality_summary else {}
                ),
                recommended_next_step=preprocessing_meta.get(
                    "recommended_next_step",
                    "Improve dataset quality before forecasting.",
                ),
                history_metadata={
                    "history_available_points": 0,
                    "history_used_points": 0,
                    "history_selection_reason": "Readiness checks failed before history selection.",
                    "preprocessing_metadata": preprocessing_meta,
                },
                candidate_diagnostics=[],
            )

        ts_df = self._prepare_time_series(working_df, date_column, target_column)

        if ts_df.empty or len(ts_df) < 10:
            return self._build_response(
                forecast_mode="refuse_forecast",
                model_used=None,
                model_selection_reason="Not enough usable aggregated time periods after time-series preparation.",
                forecast_values=[],
                validation_metrics={},
                baseline_comparison={},
                reliability_label="Not Reliable",
                reliability_score=0.0,
                warnings=preprocessing_warnings + ["Aggregated series is too short for forecasting."],
                decision_usability_flag="not_recommended",
                when_to_trust=[],
                when_not_to_trust=["Series is too short after cleaning and aggregation."],
                data_quality_summary=(
                    readiness.data_quality_summary.to_dict()
                    if readiness.data_quality_summary else {}
                ),
                recommended_next_step=preprocessing_meta.get(
                    "recommended_next_step",
                    "Provide more historical time-based records.",
                ),
                history_metadata={
                    "history_available_points": int(len(ts_df)),
                    "history_used_points": int(len(ts_df)),
                    "history_selection_reason": "Series too short after aggregation.",
                    "preprocessing_metadata": preprocessing_meta,
                },
                candidate_diagnostics=[],
            )

        full_series = ts_df[target_column]
        selected_series, history_metadata = self._select_training_window(full_series)
        history_metadata["preprocessing_metadata"] = preprocessing_meta

        if selected_series.empty or len(selected_series) < 10:
            return self._build_response(
                forecast_mode="refuse_forecast",
                model_used=None,
                model_selection_reason="Selected training window is too short for forecasting.",
                forecast_values=[],
                validation_metrics={},
                baseline_comparison={},
                reliability_label="Not Reliable",
                reliability_score=0.0,
                warnings=preprocessing_warnings + ["Selected historical window is too short after optimization."],
                decision_usability_flag="not_recommended",
                when_to_trust=[],
                when_not_to_trust=["Selected history is insufficient for forecast generation."],
                data_quality_summary=(
                    readiness.data_quality_summary.to_dict()
                    if readiness.data_quality_summary else {}
                ),
                recommended_next_step=preprocessing_meta.get(
                    "recommended_next_step",
                    "Provide more historical records or improve time-series continuity.",
                ),
                history_metadata=history_metadata,
                candidate_diagnostics=[],
            )

        candidate_models = self._get_candidate_models()
        validation_results = []

        for model_name, predictor_fn in candidate_models.items():
            try:
                result = self.validator.validate(
                    series=selected_series,
                    model_name=model_name,
                    predictor_fn=predictor_fn,
                )
                validation_results.append(result)
            except Exception as e:
                validation_results.append(
                    self._build_failed_validation_result(model_name, str(e))
                )

        seasonality_strength = self._detect_weekly_seasonality(selected_series)
        best_result = self._select_best_model(
            validation_results,
            seasonality_strength=seasonality_strength,
        )
        baseline_result = self._get_baseline_result(validation_results)

        if best_result is None or best_result.aggregate_metrics.mae is None:
            return self._build_response(
                forecast_mode="risk_signal",
                model_used=None,
                model_selection_reason="Forecast not reliable enough for prediction, but the system detected weak planning signal.",
                forecast_values=[],
                validation_metrics={},
                baseline_comparison={},
                reliability_label="Low",
                reliability_score=0.35,
                warnings=preprocessing_warnings + ["Unable to establish reliable forecast validation."],
                decision_usability_flag="usable_with_caution",
                when_to_trust=["Only for rough directional review."],
                when_not_to_trust=["Do not use for critical financial, procurement, or inventory commitments."],
                data_quality_summary=(
                    readiness.data_quality_summary.to_dict()
                    if readiness.data_quality_summary else {}
                ),
                recommended_next_step=preprocessing_meta.get(
                    "recommended_next_step",
                    "Add more history or improve continuity before using predictive forecasts.",
                ),
                history_metadata=history_metadata,
                candidate_diagnostics=self._build_candidate_diagnostics(validation_results),
            )

        selected_predictor = candidate_models[best_result.model_name]
        future_predictions = selected_predictor(selected_series, forecast_horizon)
        future_dates = self._generate_future_dates(selected_series.index, forecast_horizon)

        forecast_values = [
            {
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "predicted_value": round(float(value), 4),
            }
            for date, value in zip(future_dates, future_predictions)
        ]

        baseline_comparison = self._build_baseline_comparison(best_result, baseline_result)

        reliability_score, reliability_label, decision_usability_flag, warnings = self._build_reliability_summary(
            readiness_score=readiness.readiness_score,
            validation_result=best_result,
            baseline_comparison=baseline_comparison,
            readiness_warnings=preprocessing_warnings + [issue.message for issue in readiness.issues],
        )

        if seasonality_strength >= 0.15:
            warnings.append(
                f"Weekly seasonality detected (lag-7 autocorrelation={round(float(seasonality_strength), 3)}), so seasonal models were prioritized."
            )

        forecast_mode = "predictive_forecast"
        if reliability_label in {"Low", "Not Reliable"}:
            forecast_mode = "trend_projection"

        return self._build_response(
            forecast_mode=forecast_mode,
            model_used=best_result.model_name,
            model_selection_reason=self._build_model_selection_reason(
                best_result=best_result,
                baseline_comparison=baseline_comparison,
                seasonality_strength=seasonality_strength,
            ),
            forecast_values=forecast_values,
            validation_metrics=best_result.aggregate_metrics.to_dict(),
            baseline_comparison=baseline_comparison,
            reliability_label=reliability_label,
            reliability_score=reliability_score,
            warnings=warnings,
            decision_usability_flag=decision_usability_flag,
            when_to_trust=[
                "Use for short-term operational planning.",
                "Use when recent business pattern remains similar to recent history.",
            ],
            when_not_to_trust=[
                "Avoid relying on this alone for long-term commitments.",
                "Do not use without caution if business conditions changed recently.",
                "Do not treat this as guaranteed future truth.",
            ],
            data_quality_summary=(
                readiness.data_quality_summary.to_dict()
                if readiness.data_quality_summary else {}
            ),
            recommended_next_step=preprocessing_meta.get(
                "recommended_next_step",
                "Forecast engine avoids filling business-valid missing dates and uses only forecastable history.",
            ),
            history_metadata=history_metadata,
            candidate_diagnostics=self._build_candidate_diagnostics(validation_results),
        )

    def _clean_general_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False, na=False)]
        df.replace(["", " ", "NA", "N/A", "null", "None"], pd.NA, inplace=True)
        df.columns = [str(col).strip() for col in df.columns]

        return df

    def _apply_business_missing_safety(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Never forward fill / backward fill / interpolate date columns here.

        Special case:
        If the date column appears to be a delivery-style date and the dataset contains
        `is_delivered`, then pending rows are excluded from forecast training because their
        empty delivery dates are business-valid missing values.
        """
        working_df = df.copy()
        warnings: List[str] = []
        metadata: Dict[str, Any] = {
            "business_missing_strategy": "keep_null",
            "rows_before_filtering": int(len(working_df)),
            "rows_removed_for_training": 0,
            "used_is_delivered_filter": False,
            "date_column_type": "generic_date",
            "recommended_next_step": "Proceed with forecasting on valid historical records only.",
        }

        normalized_date_name = (
            str(date_column).strip().lower().replace(" ", "_").replace("-", "_")
        )

        delivery_like_tokens = {"delivery", "invoice", "delivered", "shipment"}
        is_delivery_like_date = any(token in normalized_date_name for token in delivery_like_tokens)

        if is_delivery_like_date:
            metadata["date_column_type"] = "delivery_style_date"

        if date_column not in working_df.columns:
            return working_df, warnings, metadata

        if target_column not in working_df.columns:
            return working_df, warnings, metadata

        # Safe date parsing only. No fill.
        working_df[date_column] = pd.to_datetime(
            working_df[date_column], errors="coerce", dayfirst=True
        )

        # If this looks like a delivery-style date and helper flag exists,
        # use only delivered rows for training.
        if is_delivery_like_date and "is_delivered" in working_df.columns:
            before_rows = len(working_df)
            delivered_mask = pd.to_numeric(working_df["is_delivered"], errors="coerce").fillna(0).astype(int) == 1
            working_df = working_df.loc[delivered_mask].copy()
            removed_rows = before_rows - len(working_df)

            metadata["rows_removed_for_training"] = int(removed_rows)
            metadata["used_is_delivered_filter"] = True
            metadata["recommended_next_step"] = (
                "Pending rows were excluded because delivery-style dates should remain empty until the business event happens."
            )

            if removed_rows > 0:
                warnings.append(
                    f"{removed_rows} pending or undelivered rows were excluded from forecast training because '{date_column}' is a delivery-style date."
                )

        # Never fill missing dates. Let downstream drop only unusable rows.
        missing_date_count = int(working_df[date_column].isna().sum())
        if missing_date_count > 0:
            warnings.append(
                f"{missing_date_count} rows still have empty or invalid '{date_column}' values. These rows will not be used in time-series aggregation."
            )

        metadata["rows_after_filtering"] = int(len(working_df))

        return working_df, warnings, metadata

    def _build_failed_validation_result(self, model_name: str, error_message: str):
        class _DummyMetrics:
            mae = None
            rmse = None
            mape = None
            smape = None
            wape = None

            def to_dict(self):
                return {
                    "mae": None,
                    "rmse": None,
                    "mape": None,
                    "smape": None,
                    "wape": None,
                }

        class _DummyResult:
            def __init__(self, name: str, error: str):
                self.model_name = name
                self.aggregate_metrics = _DummyMetrics()
                self.is_stable = False
                self.notes = [f"validation_failed={error}"]

        return _DummyResult(model_name, error_message)

    def _build_response(
        self,
        forecast_mode: str,
        model_used: Optional[str],
        model_selection_reason: str,
        forecast_values: List[Dict[str, Any]],
        validation_metrics: Dict[str, Any],
        baseline_comparison: Dict[str, Any],
        reliability_label: str,
        reliability_score: float,
        warnings: List[str],
        decision_usability_flag: str,
        when_to_trust: List[str],
        when_not_to_trust: List[str],
        data_quality_summary: Dict[str, Any],
        recommended_next_step: str,
        history_metadata: Dict[str, Any],
        candidate_diagnostics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = ForecastResponse(
            forecast_mode=forecast_mode,
            model_used=model_used,
            model_selection_reason=model_selection_reason,
            forecast_values=forecast_values,
            validation_metrics=validation_metrics,
            reliability_label=reliability_label,
            reliability_score=reliability_score,
            warnings=warnings,
            decision_usability_flag=decision_usability_flag,
            when_to_trust=when_to_trust,
            when_not_to_trust=when_not_to_trust,
            data_quality_summary=data_quality_summary,
            recommended_next_step=recommended_next_step,
            history_metadata=history_metadata,
            baseline_comparison=baseline_comparison,
        ).to_dict()

        payload["candidate_diagnostics"] = candidate_diagnostics
        return payload

    def _prepare_time_series(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
    ) -> pd.DataFrame:
        working_df = df[[date_column, target_column]].copy()
        working_df[date_column] = pd.to_datetime(working_df[date_column], errors="coerce")
        working_df[target_column] = pd.to_numeric(working_df[target_column], errors="coerce")
        working_df = working_df.dropna(subset=[date_column, target_column]).copy()

        if working_df.empty:
            return pd.DataFrame(columns=[target_column])

        return (
            working_df.groupby(date_column, as_index=True)[target_column]
            .sum()
            .sort_index()
            .to_frame()
        )

    def _select_training_window(
        self,
        series: pd.Series,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        total_len = len(series)

        if total_len < 60:
            return series, {
                "history_available_points": int(total_len),
                "history_used_points": int(total_len),
                "history_used_span": "full_history",
                "history_selection_reason": "Dataset too small for window optimization, so full history was used.",
            }

        candidate_windows = sorted({
            max(28, int(total_len * 0.5)),
            max(28, int(total_len * 0.7)),
            max(28, int(total_len * 0.9)),
            total_len,
        })

        best_series = series
        best_score = float("inf")
        best_window = total_len
        best_model_name = "naive_last_value"

        candidate_models = self._get_candidate_models()

        for window in candidate_windows:
            sub_series = series.iloc[-window:]
            seasonality_strength = self._detect_weekly_seasonality(sub_series)

            for model_name, predictor_fn in candidate_models.items():
                try:
                    result = self.validator.validate(
                        series=sub_series,
                        model_name=f"window_test_{model_name}",
                        predictor_fn=predictor_fn,
                    )
                    metric = result.aggregate_metrics.mae
                    if metric is None:
                        continue

                    adjusted_metric = float(metric)

                    if seasonality_strength >= 0.15 and "seasonal" in model_name:
                        adjusted_metric *= 0.93

                    if adjusted_metric < best_score:
                        best_score = adjusted_metric
                        best_series = sub_series
                        best_window = window
                        best_model_name = model_name
                except Exception:
                    continue

        used_span = "full_history" if best_window == total_len else f"last_{best_window}_points"

        return best_series, {
            "history_available_points": int(total_len),
            "history_used_points": int(best_window),
            "history_used_span": used_span,
            "history_selection_reason": (
                f"Selected the historical window with the best seasonality-aware validation performance using candidate model '{best_model_name}'."
            ),
        }

    def _get_candidate_models(self) -> Dict[str, Any]:
        candidates = {
            "naive_last_value": self._predict_naive_last_value,
            "seasonal_naive_7": self._predict_seasonal_naive_7,
            "seasonal_weekday_average": self._predict_seasonal_weekday_average,
            "drift_trend_model": self._predict_drift_trend,
        }

        if self.xgb_model is not None:
            candidates["xgboost_lag_model"] = self.xgb_model.predict

        return candidates

    def _predict_naive_last_value(self, train_series: pd.Series, horizon: int) -> np.ndarray:
        last_value = float(train_series.iloc[-1])
        return np.array([last_value] * horizon, dtype=float)

    def _predict_drift_trend(self, train_series: pd.Series, horizon: int) -> np.ndarray:
        if len(train_series) < 2:
            return self._predict_naive_last_value(train_series, horizon)

        y = train_series.values.astype(float)
        recent_window = min(len(y), 12)
        recent_y = y[-recent_window:]

        if len(recent_y) < 2:
            return self._predict_naive_last_value(train_series, horizon)

        x = np.arange(recent_window, dtype=float)
        try:
            slope, intercept = np.polyfit(x, recent_y, 1)
        except Exception:
            return self._predict_naive_last_value(train_series, horizon)

        recent_std = float(np.std(recent_y)) if len(recent_y) > 1 else 0.0
        max_abs_slope = max(1.0, recent_std * 0.5)
        slope = float(np.clip(slope, -max_abs_slope, max_abs_slope))

        last_x = x[-1]
        future = [intercept + slope * (last_x + i + 1) for i in range(horizon)]
        return np.maximum(np.array(future, dtype=float), 0.0)

    def _predict_seasonal_naive_7(self, train_series: pd.Series, horizon: int) -> np.ndarray:
        series = train_series.astype(float).sort_index()

        if len(series) < 7:
            return self._predict_naive_last_value(series, horizon)

        values = series.values
        preds = [float(values[-7 + (i % 7)]) for i in range(horizon)]
        return np.array(preds, dtype=float)

    def _predict_seasonal_weekday_average(self, train_series: pd.Series, horizon: int) -> np.ndarray:
        series = train_series.astype(float).sort_index()

        if not isinstance(series.index, pd.DatetimeIndex):
            return self._predict_seasonal_naive_7(series, horizon)

        if len(series) < 14:
            return self._predict_seasonal_naive_7(series, horizon)

        recent_cycles = min(len(series), 28)
        recent_series = series.iloc[-recent_cycles:]
        weekday_means = recent_series.groupby(recent_series.index.dayofweek).mean()
        overall_mean = float(recent_series.mean())

        future_dates = self._generate_future_dates(series.index, horizon)
        preds: List[float] = []

        for next_ts in future_dates:
            weekday = int(next_ts.dayofweek)
            pred = float(weekday_means.get(weekday, overall_mean))
            preds.append(max(0.0, pred))

        return np.array(preds, dtype=float)

    def _detect_weekly_seasonality(self, series: pd.Series) -> float:
        try:
            if len(series) < 21:
                return 0.0
            value = float(series.autocorr(lag=7))
            if np.isnan(value):
                return 0.0
            return value
        except Exception:
            return 0.0

    def _select_best_model(
        self,
        validation_results: List[Any],
        seasonality_strength: float = 0.0,
    ) -> Optional[Any]:
        valid_results = [
            result for result in validation_results
            if result.aggregate_metrics.mae is not None
        ]

        if not valid_results:
            return None

        valid_results.sort(key=lambda x: x.aggregate_metrics.mae)
        best_result = valid_results[0]
        best_mae = float(best_result.aggregate_metrics.mae)
        results_by_name = {r.model_name: r for r in valid_results}

        if seasonality_strength >= 0.15:
            seasonal_priority = [
                "xgboost_lag_model",
                "seasonal_weekday_average",
                "seasonal_naive_7",
                "drift_trend_model",
                "naive_last_value",
            ]

            for model_name in seasonal_priority:
                candidate = results_by_name.get(model_name)
                if candidate is None:
                    continue
                candidate_mae = float(candidate.aggregate_metrics.mae)
                if candidate_mae <= best_mae * 1.20:
                    return candidate

        tolerance = best_mae * 1.15
        shortlisted = [
            result for result in valid_results
            if float(result.aggregate_metrics.mae) <= tolerance
        ]

        preferred_models = [
            "xgboost_lag_model",
            "seasonal_weekday_average",
            "seasonal_naive_7",
            "drift_trend_model",
            "naive_last_value",
        ]

        for preferred_model in preferred_models:
            for result in shortlisted:
                if result.model_name == preferred_model:
                    return result

        return best_result

    def _get_baseline_result(self, validation_results: List[Any]) -> Optional[Any]:
        baseline_candidates = [
            result for result in validation_results
            if result.model_name == "naive_last_value"
            and result.aggregate_metrics.mae is not None
        ]
        if not baseline_candidates:
            return None
        baseline_candidates.sort(key=lambda x: x.aggregate_metrics.mae)
        return baseline_candidates[0]

    def _build_baseline_comparison(
        self,
        best_result: Any,
        baseline_result: Optional[Any],
    ) -> Dict[str, Any]:
        if baseline_result is None or baseline_result.aggregate_metrics.mae is None:
            return {}

        baseline_mae = baseline_result.aggregate_metrics.mae
        best_mae = best_result.aggregate_metrics.mae

        improvement_percent = 0.0 if baseline_mae == 0 else ((baseline_mae - best_mae) / baseline_mae) * 100

        return {
            "baseline_model": baseline_result.model_name,
            "baseline_mae": round(float(baseline_mae), 4),
            "selected_model_mae": round(float(best_mae), 4),
            "improvement_percent": round(float(improvement_percent), 4),
            "selected_model_beats_baseline": (
                best_result.model_name != baseline_result.model_name and improvement_percent > 0
            ),
        }

    def _build_candidate_diagnostics(self, validation_results: List[Any]) -> List[Dict[str, Any]]:
        diagnostics = []
        for result in validation_results:
            mae = getattr(result.aggregate_metrics, "mae", None)
            diagnostics.append(
                {
                    "model_name": result.model_name,
                    "mae": round(float(mae), 4) if mae is not None else None,
                    "is_stable": bool(getattr(result, "is_stable", False)),
                    "notes": list(getattr(result, "notes", [])),
                }
            )

        diagnostics.sort(key=lambda x: (x["mae"] is None, x["mae"] if x["mae"] is not None else float("inf")))
        return diagnostics

    def _build_model_selection_reason(
        self,
        best_result: Any,
        baseline_comparison: Dict[str, Any],
        seasonality_strength: float = 0.0,
    ) -> str:
        if seasonality_strength >= 0.15 and best_result.model_name in {
            "seasonal_weekday_average",
            "seasonal_naive_7",
            "xgboost_lag_model",
        }:
            return (
                f"Selected '{best_result.model_name}' because weekly seasonality was detected "
                f"(lag-7 autocorrelation={round(float(seasonality_strength), 3)}), and seasonal models "
                f"outperformed or came close enough to the top MAE while preserving date-wise variation."
            )

        if baseline_comparison.get("selected_model_beats_baseline"):
            improvement = baseline_comparison.get("improvement_percent", 0.0)
            baseline_model = baseline_comparison.get("baseline_model")
            return (
                f"Selected '{best_result.model_name}' because it achieved strong validation performance "
                f"and improved over baseline model '{baseline_model}' by {round(float(improvement), 2)}% on MAE."
            )

        return f"Selected '{best_result.model_name}' because it achieved the best usable validation performance among available candidates."

    def _generate_future_dates(self, historical_index: pd.Index, horizon: int) -> List[pd.Timestamp]:
        if len(historical_index) < 2:
            last_date = pd.to_datetime(historical_index[-1])
            return [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]

        inferred_freq = pd.infer_freq(historical_index)
        last_date = pd.to_datetime(historical_index[-1])

        if inferred_freq:
            return list(pd.date_range(start=last_date, periods=horizon + 1, freq=inferred_freq)[1:])

        return [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]

    def _build_reliability_summary(
        self,
        readiness_score: float,
        validation_result: Any,
        baseline_comparison: Dict[str, Any],
        readiness_warnings: List[str],
    ) -> tuple[float, str, str, List[str]]:
        warnings = list(readiness_warnings)

        validation_mae = validation_result.aggregate_metrics.mae or 0.0
        notes_text = " ".join(validation_result.notes).lower()

        if validation_mae > 1000000:
            validation_component = 0.3
            warnings.append("Validation error is high.")
        elif validation_mae > 500000:
            validation_component = 0.5
            warnings.append("Validation error is moderately high.")
        elif validation_mae > 100000:
            validation_component = 0.7
        else:
            validation_component = 0.9

        stability_component = 0.9 if validation_result.is_stable else 0.5
        if not validation_result.is_stable:
            warnings.append("Forecast validation is unstable across folds.")

        improvement_percent = float(baseline_comparison.get("improvement_percent", 0.0))
        if baseline_comparison.get("selected_model_beats_baseline"):
            baseline_component = 0.9 if improvement_percent >= 10 else 0.75
        else:
            baseline_component = 0.55
            warnings.append("Selected model does not meaningfully outperform baseline.")

        consistency_component = 0.8 if "consistency_score=" in notes_text else 0.6

        reliability_score = round(
            (0.35 * readiness_score)
            + (0.25 * validation_component)
            + (0.15 * stability_component)
            + (0.15 * baseline_component)
            + (0.10 * consistency_component),
            4,
        )

        if reliability_score >= 0.8:
            return reliability_score, "High", "usable", warnings
        if reliability_score >= 0.6:
            return reliability_score, "Medium", "usable_with_caution", warnings
        if reliability_score >= 0.45:
            return reliability_score, "Low", "usable_with_caution", warnings

        return reliability_score, "Not Reliable", "not_recommended", warnings