from typing import Any, Dict, List


class AnalysisContextBuilder:
    @staticmethod
    def build_explainer_context(analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        schema_suggestions = analysis_context.get("schema_suggestions", {})
        capabilities = analysis_context.get("capabilities", {})
        profile = analysis_context.get("profile", {})
        insights = analysis_context.get("insights", {})
        forecast = analysis_context.get("forecast", {})
        decisions = analysis_context.get("decisions", {})
        narratives = analysis_context.get("narratives", {})
        dashboard_summary = analysis_context.get("dashboard_summary", {})

        dataset_profile = profile.get("dataset_profile", {})
        safe_insights = AnalysisContextBuilder._sanitize_insights(insights)

        enabled_capabilities = [
            cap_name
            for cap_name, cap_data in capabilities.items()
            if isinstance(cap_data, dict) and cap_data.get("enabled")
        ]

        executive_overview = {
            "dataset_name": dashboard_summary.get("dataset_name"),
            "rows": dashboard_summary.get("rows", dataset_profile.get("total_rows")),
            "columns": dashboard_summary.get("columns", dataset_profile.get("total_columns")),
            "duplicate_rows": dashboard_summary.get("duplicate_rows", dataset_profile.get("duplicate_rows")),
            "enabled_capabilities": enabled_capabilities,
            "business_signals": dashboard_summary.get("business_signals", []),
            "executive_summary": dashboard_summary.get("executive_summary"),
            "headline_summary": dashboard_summary.get("headline_summary"),
            "recommended_next_step": dashboard_summary.get("recommended_next_step"),
            "forecast_type": dashboard_summary.get("forecast_type"),
            "model_used": dashboard_summary.get("model_used"),
            "reliability_label": dashboard_summary.get("reliability_label"),
            "decision_count": dashboard_summary.get("decision_count"),
            "high_priority_count": dashboard_summary.get("high_priority_count"),
            "segment_count": dashboard_summary.get("segment_count"),
            "top_risk": dashboard_summary.get("top_risk"),
        }

        sales_context = {
            "top_products_by_revenue": AnalysisContextBuilder._limit_list(
                safe_insights.get("sales", {}).get("top_products_by_revenue", []), 5
            ),
            "top_regions_by_revenue": AnalysisContextBuilder._limit_list(
                safe_insights.get("sales", {}).get("top_regions_by_revenue", []), 5
            ),
            "revenue_trend": AnalysisContextBuilder._limit_list(
                safe_insights.get("sales", {}).get("revenue_trend", []), 20
            ),
            "quantity_trend": AnalysisContextBuilder._limit_list(
                safe_insights.get("sales", {}).get("quantity_trend", []), 20
            ),
        }

        inventory_context = safe_insights.get("inventory", {})
        logistics_context = safe_insights.get("logistics", {})

        forecast_context = {
            "forecast_version": forecast.get("forecast_version"),
            "target_role": forecast.get("forecast_target_role", forecast.get("target_role")),
            "target_column": forecast.get("forecast_target_column", forecast.get("target_column")),
            "date_column": forecast.get("forecast_date_column", forecast.get("date_column")),
            "model_type": forecast.get("model_type"),
            "model_used": forecast.get("model_used", forecast.get("model_type")),
            "forecast_horizon": forecast.get("forecast_horizon"),
            "reliability_label": forecast.get("reliability_label"),
            "metrics": forecast.get("metrics", {}),
            "warnings": forecast.get("warnings", []),
            "summary": forecast.get("summary"),
            "error": forecast.get("error"),
            "latest_actual_value": forecast.get("latest_actual_value"),
            "average_forecast_value": forecast.get(
                "average_forecast_value",
                forecast.get("average_forecast")
            ),

            # old structure support
            "historical_points": AnalysisContextBuilder._limit_list(
                forecast.get("historical_points", []), 20
            ),
            "forecast_points": AnalysisContextBuilder._limit_list(
                forecast.get("forecast_points", []), 20
            ),
            "historical_series_preview": AnalysisContextBuilder._limit_list(
                forecast.get("historical_series", []), 12
            ),
            "forecast_series_preview": AnalysisContextBuilder._limit_list(
                forecast.get("forecast_series", []), 12
            ),

            # new structure support
            "validation_points": AnalysisContextBuilder._limit_list(
                forecast.get("validation_points", []), 12
            ),
            "future_forecast": AnalysisContextBuilder._limit_list(
                forecast.get("future_forecast", []), 12
            ),

            # segmented forecast support
            "segment_column": forecast.get("segment_column"),
            "segments_count": forecast.get("segments_count", 0),
            "segments_preview": AnalysisContextBuilder._compact_segments(
                forecast.get("segments", []), 5
            ),

            # feature engineering support
            "feature_summary": forecast.get("feature_summary", {}),
        }

        compact_decisions = {
            "summary": decisions.get("summary") or decisions.get("decision_summary") or "",
            "top_decisions": AnalysisContextBuilder._compact_decisions(
                decisions.get("top_decisions", []), 8
            ),
            "error": decisions.get("error"),
        }

        data_quality_context = {
            "total_rows": dataset_profile.get("total_rows"),
            "total_columns": dataset_profile.get("total_columns"),
            "duplicate_rows": dataset_profile.get("duplicate_rows"),
            "null_counts": dataset_profile.get("null_counts", {}),
            "null_percentages": dataset_profile.get("null_percentages", {}),
            "numeric_columns": dataset_profile.get("numeric_columns", []),
            "categorical_columns": dataset_profile.get("categorical_columns", []),
            "datetime_candidate_columns": dataset_profile.get("datetime_candidate_columns", []),
        }

        return {
            "executive_overview": executive_overview,
            "schema_suggestions": schema_suggestions,
            "capabilities": capabilities,
            "sales_context": sales_context,
            "inventory_context": inventory_context,
            "logistics_context": logistics_context,
            "forecast_context": forecast_context,
            "decision_context": compact_decisions,
            "data_quality_context": data_quality_context,
            "narratives": narratives,
        }

    @staticmethod
    def _sanitize_insights(insights: Dict[str, Any]) -> Dict[str, Any]:
        safe_insights = {}

        for key, value in insights.items():
            lower_key = str(key).lower()

            if isinstance(value, dict):
                safe_insights[key] = {
                    sub_key: AnalysisContextBuilder._sanitize_nested_value(sub_key, sub_value)
                    for sub_key, sub_value in value.items()
                }
                continue

            safe_insights[key] = AnalysisContextBuilder._sanitize_nested_value(lower_key, value)

        return safe_insights

    @staticmethod
    def _sanitize_nested_value(key: str, value: Any) -> Any:
        lower_key = str(key).lower()

        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            return {
                k: AnalysisContextBuilder._sanitize_nested_value(k, v)
                for k, v in value.items()
            }

        if "trend" in lower_key and isinstance(value, str):
            return {
                "value": value,
                "trusted": False,
                "note": "Trend text should be used cautiously unless explicitly validated by deterministic logic."
            }

        return value

    @staticmethod
    def _limit_list(value: Any, limit: int) -> List[Any]:
        if isinstance(value, list):
            return value[:limit]
        return []

    @staticmethod
    def _compact_segments(segments: Any, limit: int) -> List[Dict[str, Any]]:
        if not isinstance(segments, list):
            return []

        compact = []
        for seg in segments[:limit]:
            if not isinstance(seg, dict):
                continue

            compact.append({
                "segment": seg.get("segment"),
                "model_used": seg.get("model_used"),
                "reliability": seg.get("reliability"),
                "metrics": seg.get("metrics", {}),
                "future_forecast": AnalysisContextBuilder._limit_list(
                    seg.get("future_forecast", []), 7
                ),
            })

        return compact

    @staticmethod
    def _compact_decisions(decisions: Any, limit: int) -> List[Dict[str, Any]]:
        if not isinstance(decisions, list):
            return []

        compact = []
        for item in decisions[:limit]:
            if not isinstance(item, dict):
                continue

            compact.append({
                "decision_type": item.get("decision_type"),
                "title": item.get("title"),
                "priority": item.get("priority"),
                "risk_level": item.get("risk_level"),
                "recommendation": item.get("recommendation"),
                "rationale": item.get("rationale"),
                "explanation": item.get("explanation"),
                "evidence": item.get("evidence", {}),
            })

        return compact