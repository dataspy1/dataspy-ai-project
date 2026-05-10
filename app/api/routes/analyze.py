from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import math
import pandas as pd

import re # added for proper dataset name

from app.engines.mapping.schema_mapper import detect_schema
from app.engines.understanding.capability_detector import detect_capabilities
from app.engines.understanding.profiler import profile_dataframe
from app.engines.insights.insight_engine import generate_insights
from app.engines.insights.narrative_engine import generate_narrative
from app.engines.forecasting.forecast_orchestrator import ForecastOrchestrator
from app.engines.decisions.decision_engine import generate_decisions
from app.services.data.dataset_store import DatasetStore
from app.utils.client_dataset_standardizer import standardize_client_dataset

router = APIRouter()
forecast_engine = ForecastOrchestrator()


class AnalyzeRequest(BaseModel):
    saved_filename: str
    forecast_target_role: Optional[str] = "revenue"
    forecast_horizon: Optional[int] = 7


def clean_nan_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_nan_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def get_missing_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    missing_by_column = df.isna().sum().to_dict()
    missing_by_column = {
        str(col): int(count)
        for col, count in missing_by_column.items()
        if int(count) > 0
    }

    total_missing_values = int(df.isna().sum().sum())
    columns_with_missing = len(missing_by_column)

    return {
        "total_missing_values": total_missing_values,
        "columns_with_missing": columns_with_missing,
        "missing_by_column": missing_by_column,
        "has_missing_data": total_missing_values > 0,
        "warning_message": (
            "Missing data detected. Continuing without cleaning may reduce forecast accuracy, insight quality, and decision reliability."
            if total_missing_values > 0
            else None
        ),
    }

def handle_business_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generic handling for business-meaningful missing values.
    Example: delivery date empty because order is not delivered yet.
    Do not fill or drop such columns.
    """
    df = df.copy()

    # Remove useless unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False, na=False)]

    # Standardize empty-like values
    df.replace(["", " ", "NA", "N/A", "null", "None"], pd.NA, inplace=True)

    # Normalize column name lookup
    normalized_map = {str(col).strip().lower().replace(" ", "_"): col for col in df.columns}

    delivery_col = None
    status_col = None

    for key in ["delivery_date", "invoice_date", "delivered_date"]:
        if key in normalized_map:
            delivery_col = normalized_map[key]
            break

    for key in ["status", "delivery_status", "order_status"]:
        if key in normalized_map:
            status_col = normalized_map[key]
            break

    # If delivery date is empty and status says pending/not delivered,
    # keep it empty. This is business missingness, not dirty data.
    if delivery_col and status_col:
        status_series = df[status_col].astype(str).str.strip().str.lower()

        pending_mask = status_series.isin([
            "not delivered", "pending", "open", "in transit", "processing"
        ])

        # Keep pending delivery dates empty
        df.loc[pending_mask & df[delivery_col].isna(), delivery_col] = pd.NA

        # Optional helper feature for downstream logic
        df["is_delivered"] = (~(pending_mask & df[delivery_col].isna())).astype(int)

    elif delivery_col:
        # If no status column exists, just create a safe helper flag
        df["is_delivered"] = df[delivery_col].notna().astype(int)

    return df

def _resolve_schema_column(schema_suggestions: Dict[str, Any], role: str) -> Optional[str]:
    role_value = schema_suggestions.get(role)

    if isinstance(role_value, str) and role_value.strip():
        return role_value.strip()

    if isinstance(role_value, dict):
        column_name = role_value.get("column") or role_value.get("name")
        if isinstance(column_name, str) and column_name.strip():
            return column_name.strip()

    return None


def _smart_resolve_columns(
    df: pd.DataFrame,
    schema_suggestions: Dict[str, Any],
    target_role: Optional[str],
):
    def normalize(text: str) -> str:
        return (
            str(text)
            .strip()
            .lower()
            .replace("-", "_")
            .replace("/", "_")
            .replace(" ", "_")
        )

    normalized_map = {normalize(col): col for col in df.columns}

    # ---- DATE RESOLUTION ----
    date_col = _resolve_schema_column(schema_suggestions, "date")

    if not date_col:
        for key in [
            "po_date",
            "date",
            "order_date",
            "transaction_date",
            "posting_date",
            "doc_date",
        ]:
            if key in normalized_map:
                date_col = normalized_map[key]
                break

    if not date_col:
        for norm_name, original_name in normalized_map.items():
            if "date" in norm_name and "invoice" not in norm_name and "delivery" not in norm_name:
                date_col = original_name
                break

    # ---- TARGET RESOLUTION ----
    role = (target_role or "revenue").strip().lower()
    target_col = _resolve_schema_column(schema_suggestions, role)

    if not target_col:
        if role == "revenue":
            for key in [
                "total_price",
                "revenue",
                "sales",
                "amount",
                "total_amount",
                "net_total",
                "value",
                "invoice_value",
            ]:
                if key in normalized_map:
                    target_col = normalized_map[key]
                    break

            if not target_col:
                for norm_name, original_name in normalized_map.items():
                    if any(token in norm_name for token in ["price", "revenue", "sales", "amount", "value"]):
                        target_col = original_name
                        break

        elif role == "quantity":
            for key in ["quantity", "quamtity", "qty", "units", "sold_qty"]:
                if key in normalized_map:
                    target_col = normalized_map[key]
                    break

            if not target_col:
                for norm_name, original_name in normalized_map.items():
                    if any(token in norm_name for token in ["quantity", "quamtity", "qty", "units"]):
                        target_col = original_name
                        break

        elif role == "stock":
            for key in ["stock", "inventory", "inventory_level"]:
                if key in normalized_map:
                    target_col = normalized_map[key]
                    break

    return date_col, target_col


def _build_historical_series(
    df: pd.DataFrame,
    date_column: Optional[str],
    target_column: Optional[str],
    limit: int = 30,
):
    if not date_column or not target_column:
        return []
    if date_column not in df.columns or target_column not in df.columns:
        return []

    temp = df[[date_column, target_column]].copy()
    temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce", dayfirst=True)
    temp[target_column] = pd.to_numeric(temp[target_column], errors="coerce")
    temp = temp.dropna(subset=[date_column, target_column]).copy()

    if temp.empty:
        return []

    aggregated = (
        temp.groupby(date_column, as_index=False)[target_column]
        .sum()
        .sort_values(by=date_column)
    )
    aggregated = aggregated.tail(limit)

    return [
        {
            "ds": str(row[date_column].date()) if hasattr(row[date_column], "date") else str(row[date_column]),
            "y": float(row[target_column]),
        }
        for _, row in aggregated.iterrows()
    ]


def _extract_enabled_capabilities(capabilities: Dict[str, Any]) -> list[str]:
    enabled = []
    for cap_name, cap_data in capabilities.items():
        if not isinstance(cap_data, dict):
            continue
        if cap_data.get("enabled") is True or str(cap_data.get("status", "")).lower() == "enabled":
            enabled.append(cap_name)
    return enabled


def _extract_partial_capabilities(capabilities: Dict[str, Any]) -> list[str]:
    return [
        cap_name
        for cap_name, cap_data in capabilities.items()
        if isinstance(cap_data, dict) and str(cap_data.get("status", "")).lower() in ["partial", "limited"]
    ]


def _extract_top_product_name(sales_insights: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    top_products = sales_insights.get("top_products_by_revenue", [])
    if not top_products:
        return None

    first = top_products[0]
    if not isinstance(first, dict):
        return {"name": str(first), "value": None}

    product_key = next(
        (k for k in first.keys() if k.lower() not in ["total_amount", "revenue", "sales", "sales_amount", "value"]),
        None
    )
    value_key = next(
        (k for k in first.keys() if k.lower() in ["total_amount", "revenue", "sales", "sales_amount", "value"]),
        None
    )

    if not product_key:
        return None

    return {"name": first.get(product_key), "value": first.get(value_key) if value_key else None}


def _extract_top_region_name(sales_insights: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    top_regions = sales_insights.get("top_regions_by_revenue", [])
    if not top_regions:
        return None

    first = top_regions[0]
    if not isinstance(first, dict):
        return {"name": str(first), "value": None}

    region_key = next(
        (k for k in first.keys() if k.lower() not in ["total_amount", "revenue", "sales", "sales_amount", "value"]),
        None
    )
    value_key = next(
        (k for k in first.keys() if k.lower() in ["total_amount", "revenue", "sales", "sales_amount", "value"]),
        None
    )

    if not region_key:
        return None

    return {"name": first.get(region_key), "value": first.get(value_key) if value_key else None}


def dashboard_summary_safe_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def build_dashboard_summary(
    file_summary: Dict[str, Any],
    capabilities: Dict[str, Any],
    profile: Dict[str, Any],
    insights: Dict[str, Any],
    forecast: Dict[str, Any],
    narratives: Dict[str, Any],
    decisions: Dict[str, Any],
    missing_data_summary: Dict[str, Any],
) -> Dict[str, Any]:
    sales_insights = insights.get("sales", {}) or {}
    inventory_insights = insights.get("inventory", {}) or {}
    logistics_insights = insights.get("logistics", {}) or {}

    top_decisions = decisions.get("top_decisions", []) or []
    decision_quality = decisions.get("decision_quality", {}) or {}
    decision_data_quality = decisions.get("data_quality", {}) or {}

    high_priority_count = len([d for d in top_decisions if str(d.get("priority", "")).lower() == "high"])
    medium_priority_count = len([d for d in top_decisions if str(d.get("priority", "")).lower() == "medium"])

    top_product = _extract_top_product_name(sales_insights)
    top_region = _extract_top_region_name(sales_insights)

    enabled_capabilities = _extract_enabled_capabilities(capabilities)
    partial_capabilities = _extract_partial_capabilities(capabilities)

    forecast_summary = forecast.get("forecast_summary", {}) or {}
    forecast_warnings = forecast.get("warnings", []) or []
    forecast_reliability_label = forecast.get("reliability_label")
    forecast_mode = forecast.get("forecast_mode")

    business_signals = []

    if "sales" in enabled_capabilities:
        business_signals.append("Revenue and sales visibility is available from the uploaded dataset.")
    if "inventory" in enabled_capabilities:
        business_signals.append("Inventory signals are present, allowing stock and replenishment-related analysis.")
    if "logistics" in enabled_capabilities:
        business_signals.append("Logistics flow can be reviewed using shipment, delivery, and status signals.")

    for partial_cap in partial_capabilities:
        business_signals.append(f"{partial_cap.capitalize()} capability is partially available but has incomplete supporting roles.")

    if sales_insights.get("revenue_trend_summary"):
        business_signals.append(str(sales_insights.get("revenue_trend_summary")))
    if sales_insights.get("product_concentration_summary"):
        business_signals.append(str(sales_insights.get("product_concentration_summary")))
    if sales_insights.get("weakest_region_summary"):
        business_signals.append(str(sales_insights.get("weakest_region_summary")))
    if sales_insights.get("region_revenue_pattern_summary"):
        region_pattern = sales_insights.get("region_revenue_pattern_summary")
        if isinstance(region_pattern, dict):
            business_signals.append(str(region_pattern.get("pattern_summary")))
    if sales_insights.get("pending_order_pattern_summary"):
        pending_summary = sales_insights.get("pending_order_pattern_summary")
        if isinstance(pending_summary, dict):
            business_signals.append(str(pending_summary.get("summary")))
    if inventory_insights.get("inventory_summary"):
        business_signals.append(str(inventory_insights.get("inventory_summary")))
    if logistics_insights.get("delivery_delay_summary"):
        business_signals.append(str(logistics_insights.get("delivery_delay_summary")))
    if forecast_summary.get("pattern_detected"):
        business_signals.append(f"Forecasting detected {forecast_summary.get('pattern_detected')} in recent history.")
    if forecast_reliability_label:
        business_signals.append(f"Forecast reliability is currently marked as {forecast_reliability_label}.")
    if forecast_warnings:
        business_signals.append("Forecast includes caution signals that should be reviewed before major action.")
    if missing_data_summary.get("has_missing_data"):
        business_signals.append("Missing data is present in the dataset and may affect output reliability if not handled.")
    if decision_quality.get("label"):
        business_signals.append(f"Decision intelligence quality is currently marked as {decision_quality.get('label')}.")

    executive_summary_parts = []

    if enabled_capabilities:
        executive_summary_parts.append(f"This dataset supports {', '.join(enabled_capabilities)} intelligence.")
    if partial_capabilities:
        executive_summary_parts.append(f"Partial support is also available for {', '.join(partial_capabilities)} analysis.")
    if top_product and top_product.get("name"):
        executive_summary_parts.append(f"The strongest visible product signal currently comes from {top_product['name']}.")
    if top_region and top_region.get("name"):
        executive_summary_parts.append(f"The strongest visible regional signal currently comes from {top_region['name']}.")
    if forecast_mode:
        executive_summary_parts.append(f"Forecast mode is {forecast_mode} with reliability marked as {forecast_reliability_label or 'Unknown'}.")
    if forecast_summary.get("selected_model"):
        executive_summary_parts.append(f"The forecast engine selected {forecast_summary.get('selected_model')} as the most usable model.")
    if len(top_decisions) > 0:
        executive_summary_parts.append(f"The decision engine identified {len(top_decisions)} recommendation areas for review.")
    if high_priority_count > 0:
        executive_summary_parts.append(f"{high_priority_count} of these are marked as high priority.")
    elif medium_priority_count > 0:
        executive_summary_parts.append(f"{medium_priority_count} of these are marked as medium priority.")
    if decision_quality.get("average_confidence") is not None:
        executive_summary_parts.append(f"Average decision confidence is {decision_quality.get('average_confidence')}.")
    if missing_data_summary.get("has_missing_data"):
        executive_summary_parts.append(
            f"The dataset contains {missing_data_summary.get('total_missing_values', 0)} missing values across {missing_data_summary.get('columns_with_missing', 0)} columns."
        )

    executive_summary = " ".join(executive_summary_parts).strip() or "The dataset has been analyzed successfully and key business signals are available for review."

    headline_summary = (
        narratives.get("executive_summary")
        or narratives.get("sales_summary")
        or decisions.get("summary")
        or executive_summary
    )

    recommended_next_step = (
        forecast.get("recommended_next_step")
        or dashboard_summary_safe_value(narratives.get("executive_summary"))
        or "Review the top recommendations first, then validate forecast and decision signals to prioritize the most urgent business actions."
    )

    if top_decisions:
        first_recommendation = top_decisions[0].get("recommendation")
        if first_recommendation:
            recommended_next_step = str(first_recommendation)

    if missing_data_summary.get("has_missing_data"):
        recommended_next_step = "Missing data was detected. Address data quality first, then review forecast and recommendation outputs more confidently."

    top_risk = "normal"
    if top_decisions:
        top_risk = top_decisions[0].get("risk_level") or "normal"
    elif forecast_warnings:
        top_risk = "forecast_uncertain"

    return {
        "saved_filename": file_summary.get("saved_filename"),
        "dataset_name": file_summary.get("saved_filename"),
        "rows": file_summary.get("rows"),
        "columns": len(file_summary.get("columns", [])),
        "duplicate_rows": profile.get("dataset_profile", {}).get("duplicate_rows", 0),
        "enabled_capabilities": enabled_capabilities,
        "partial_capabilities": partial_capabilities,
        "business_signals": business_signals[:12],
        "top_product": top_product,
        "top_region": top_region,
        "forecast_mode": forecast_mode,
        "forecast_model_used": forecast.get("model_used"),
        "forecast_model_type": forecast_summary.get("selected_model_type"),
        "forecast_pattern_detected": forecast_summary.get("pattern_detected"),
        "forecast_reliability_label": forecast_reliability_label,
        "forecast_reliability_score": forecast.get("reliability_score"),
        "forecast_horizon": forecast.get("forecast_horizon"),
        "forecast_warning_count": len(forecast_warnings),
        "decision_count": len(top_decisions),
        "high_priority_count": high_priority_count,
        "medium_priority_count": medium_priority_count,
        "decision_quality_label": decision_quality.get("label"),
        "decision_quality_score": decision_quality.get("score"),
        "decision_average_confidence": decision_quality.get("average_confidence"),
        "decision_average_impact": decision_quality.get("average_impact"),
        "data_quality_label": decision_data_quality.get("quality_label"),
        "data_quality_score": decision_data_quality.get("quality_score"),
        "top_risk": top_risk,
        "executive_summary": executive_summary,
        "headline_summary": headline_summary,
        "recommended_next_step": recommended_next_step,
        "missing_data_summary": missing_data_summary,
    }


def _build_fallback_forecast(
    payload: AnalyzeRequest,
    date_column: Optional[str],
    target_column: Optional[str],
    reason: str,
    warning: str,
    next_step: str,
    seasonality_detected: bool = False,
    seasonality_strength: float = 0.0,
) -> Dict[str, Any]:
    return {
        "forecast_mode": "refuse_forecast",
        "model_used": None,
        "model_selection_reason": reason,
        "forecast_values": [],
        "historical_series": [],
        "future_forecast": [],
        "validation_metrics": {},
        "reliability_label": "Not Reliable",
        "reliability_score": 0.0,
        "warnings": [warning],
        "decision_usability_flag": "not_recommended",
        "when_to_trust": [],
        "when_not_to_trust": ["Forecasting is not reliable in the current state."],
        "data_quality_summary": {},
        "recommended_next_step": next_step,
        "history_metadata": {},
        "baseline_comparison": {},
        "forecast_summary": {
            "selected_model": None,
            "selected_model_type": None,
            "confidence_band_note": "Forecast could not be generated reliably.",
            "pattern_detected": None,
            "validation_summary": reason,
            "seasonality_detected": seasonality_detected,
            "seasonality_strength": seasonality_strength,
            "trend_direction": None,
            "projected_change_percent": None,
            "commentary": reason,
        },
        "candidate_diagnostics": [],
        "debug_metadata": {},
        "target_role": payload.forecast_target_role or "revenue",
        "target_column": target_column,
        "date_column": date_column,
        "forecast_horizon": payload.forecast_horizon or 7,
        "summary": reason,
        "error": warning,
    }


@router.post("/analyze")
def analyze_dataset(payload: AnalyzeRequest):
    try:
        raw_df = DatasetStore.load_dataframe(payload.saved_filename)
        raw_df.columns = [str(col).strip() for col in raw_df.columns]

        standardized_result = standardize_client_dataset(raw_df)
        df = standardized_result["dataframe"]
        
        standardization_metadata = standardized_result["metadata"]

# Handle business-meaningful missing values before profiling/summarizing
        df = handle_business_missing_values(df)

        missing_data_summary = get_missing_data_summary(raw_df)

        file_summary = {
            "display_filename": re.sub(r"^[a-f0-9]{32}_", "", payload.saved_filename), #added this for proper file name
            "saved_filename": payload.saved_filename,
            "rows": int(raw_df.shape[0]), # change df.shape to raw_df.shape
            "columns": list(raw_df.columns), #  change df.shape to raw_df.shape
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
           # add this 
           "preview": raw_df.head(5).fillna("").astype(str).to_dict(orient="records"),
           "missing_data_summary": missing_data_summary,
            "standardization_metadata": standardization_metadata,

        }

        schema_suggestions = detect_schema(df)
        capabilities = detect_capabilities(schema_suggestions)
        profile = profile_dataframe(df)

        try:
            insights = generate_insights(
                df=df,
                schema_suggestions=schema_suggestions,
                capabilities=capabilities,
            )
        except Exception as ie:
            print("INSIGHTS ERROR:", str(ie))
            insights = {
                "sales": {},
                "inventory": {},
                "logistics": {},
                "error": str(ie),
            }

        forecast = {}
        forecast_error = None

        date_column, target_column = _smart_resolve_columns(
            df,
            schema_suggestions,
            payload.forecast_target_role,
        )

        try:
            if df.shape[0] < 10:
                forecast = _build_fallback_forecast(
                    payload=payload,
                    date_column=date_column,
                    target_column=target_column,
                    reason="Insufficient data for forecasting (less than 10 rows).",
                    warning="Dataset too small for forecasting.",
                    next_step="Upload more historical data (minimum 50–100 rows for useful forecasting).",
                )
            elif date_column and target_column and date_column in df.columns and target_column in df.columns:
                forecast = forecast_engine.run(
                    df=df,
                    date_column=date_column,
                    target_column=target_column,
                    forecast_horizon=payload.forecast_horizon or 7,
                )
                forecast["target_role"] = payload.forecast_target_role or "revenue"
                forecast["target_column"] = target_column
                forecast["date_column"] = date_column
                forecast["forecast_horizon"] = payload.forecast_horizon or 7
            else:
                forecast = _build_fallback_forecast(
                    payload=payload,
                    date_column=date_column,
                    target_column=target_column,
                    reason="Could not resolve valid date/target columns for forecasting from schema suggestions and fallback logic.",
                    warning=(
                        "Forecast could not start because date or target column was not resolved. "
                        f"Resolved date_column={date_column}, target_column={target_column}, "
                        f"requested target_role={payload.forecast_target_role or 'revenue'}."
                    ),
                    next_step="Verify schema mapping and selected forecast target before running forecast.",
                )
        except Exception as fe:
            forecast_error = str(fe)
            print("FORECAST ERROR:", forecast_error)
            forecast = _build_fallback_forecast(
                payload=payload,
                date_column=date_column,
                target_column=target_column,
                reason="Forecast execution failed.",
                warning=str(fe),
                next_step="Inspect forecasting pipeline logs and schema mapping.",
            )

        historical_series = _build_historical_series(
            df=df,
            date_column=date_column,
            target_column=target_column,
            limit=30,
        )

        try:
            narratives = generate_narrative(
                schema_suggestions=schema_suggestions,
                capabilities=capabilities,
                profile=profile,
                insights=insights,
            )
        except Exception as ne:
            print("NARRATIVE ERROR:", str(ne))
            narratives = {
                "executive_summary": "Narrative generation could not complete, but structured analysis is available.",
                "error": str(ne),
            }

        try:
            decisions = generate_decisions(
                df=df,
                schema_suggestions=schema_suggestions,
                forecast_result=forecast,
                capability_result=capabilities,
            )
        except Exception as de:
            print("DECISION ERROR:", str(de))
            decisions = {
                "summary": "Decision engine could not run due to limited data or missing supporting signals.",
                "error": str(de),
                "decision_quality": {},
                "data_quality": {},
                "top_decisions": [],
            }

        compact_insights = {
            "sales": {
                "top_products_by_revenue": insights.get("sales", {}).get("top_products_by_revenue", []),
                "top_regions_by_revenue": insights.get("sales", {}).get("top_regions_by_revenue", []),
                "top_products_by_quantity": insights.get("sales", {}).get("top_products_by_quantity", []),
                "top_regions_by_quantity": insights.get("sales", {}).get("top_regions_by_quantity", []),
                "top_products_by_region": insights.get("sales", {}).get("top_products_by_region", []),
                "top_products_by_region_status": insights.get("sales", {}).get("top_products_by_region_status", []),
                "revenue_trend": insights.get("sales", {}).get("revenue_trend", []),
                "quantity_trend": insights.get("sales", {}).get("quantity_trend", []),
                "sales_summary": insights.get("sales", {}).get("sales_summary"),
                "revenue_trend_summary": insights.get("sales", {}).get("revenue_trend_summary"),
                "quantity_trend_summary": insights.get("sales", {}).get("quantity_trend_summary"),
                "product_concentration_summary": insights.get("sales", {}).get("product_concentration_summary"),
                "region_concentration_summary": insights.get("sales", {}).get("region_concentration_summary"),
                "weakest_region_summary": insights.get("sales", {}).get("weakest_region_summary"),
                "region_revenue_pattern_summary": insights.get("sales", {}).get("region_revenue_pattern_summary"),
                "pending_order_pattern_summary": insights.get("sales", {}).get("pending_order_pattern_summary"),
            },
            "inventory": insights.get("inventory", {}),
            "logistics": insights.get("logistics", {}),
            "error": insights.get("error"),
        }

        compact_forecast = {
            "forecast_mode": forecast.get("forecast_mode"),
            "model_used": forecast.get("model_used"),
            "model_selection_reason": forecast.get("model_selection_reason"),
            "target_role": forecast.get("target_role"),
            "target_column": forecast.get("target_column"),
            "date_column": forecast.get("date_column"),
            "forecast_horizon": forecast.get("forecast_horizon"),
            "historical_series": historical_series if historical_series else forecast.get("historical_series", []),
            "forecast_values": forecast.get("forecast_values", []),
            "future_forecast": forecast.get("future_forecast", []),
            "latest_actual_value": forecast.get("latest_actual_value"),
            "average_forecast_value": forecast.get("average_forecast_value") or forecast.get("average_forecast"),
            "validation_metrics": forecast.get("validation_metrics", {}) or forecast.get("metrics", {}),
            "metrics": forecast.get("metrics", {}),
            "reliability_label": forecast.get("reliability_label"),
            "reliability_score": forecast.get("reliability_score"),
            "warnings": forecast.get("warnings", []),
            "decision_usability_flag": forecast.get("decision_usability_flag"),
            "decision_usability": forecast.get("decision_usability"),
            "forecast_recommendation": forecast.get("forecast_recommendation"),
            "recommendation": forecast.get("recommendation"),
            "when_to_trust": forecast.get("when_to_trust", []),
            "when_not_to_trust": forecast.get("when_not_to_trust", []),
            "data_quality_summary": forecast.get("data_quality_summary", {}),
            "recommended_next_step": forecast.get("recommended_next_step"),
            "history_metadata": forecast.get("history_metadata", {}),
            "baseline_comparison": forecast.get("baseline_comparison", {}),
            "forecast_summary": forecast.get("forecast_summary", {}),
            "candidate_diagnostics": forecast.get("candidate_diagnostics", []),
            "debug_metadata": forecast.get("debug_metadata", {}),
            "trend_direction": forecast.get("trend_direction"),
            "growth_percent": forecast.get("growth_percent"),
            "summary": (
                forecast.get("summary")
                or forecast.get("forecast_summary", {}).get("commentary")
                or forecast.get("forecast_summary", {}).get("validation_summary")
                or forecast.get("model_selection_reason")
                or forecast.get("recommended_next_step")
            ),
            "error": forecast_error or forecast.get("error"),
        }

        compact_decisions = {
            "summary": decisions.get("summary", ""),
            "error": decisions.get("error"),
            "decision_quality": decisions.get("decision_quality", {}),
            "data_quality": decisions.get("data_quality", {}),
            "top_decisions": decisions.get("top_decisions", []),
        }

        rag_ready_context = {
            "enabled": False,
            "retrieval_status": "not_connected",
            "retrieved_chunks": [],
            "retrieval_summary": "Phase C retrieval is not connected yet. Structured analysis context is ready for future RAG integration.",
            "recommended_next_phase": "Connect document embeddings, vector retrieval, and grounding prompts for policy/SOP/document-aware answers.",
        }

        dashboard_summary = build_dashboard_summary(
            file_summary=file_summary,
            capabilities=capabilities,
            profile=profile,
            insights=compact_insights,
            forecast=compact_forecast,
            narratives=narratives,
            decisions=compact_decisions,
            missing_data_summary=missing_data_summary,
        )

        response_payload = {
            "message": "Your dataset is ready for decision analysis.",
            "saved_filename": payload.saved_filename,
            "system_status": "partial_data_mode" if compact_forecast.get("forecast_mode") == "refuse_forecast" else "full_analysis_mode",
            "standardization_metadata": standardization_metadata,
            "dashboard_summary": dashboard_summary,
            "file_summary": file_summary,
            "schema_suggestions": schema_suggestions,
            "capabilities": capabilities,
            "profile": profile,
            "insights": compact_insights,
            "forecast": compact_forecast,
            "forecast_error": forecast_error,
            "narratives": narratives,
            "decisions": compact_decisions,
            "rag_context": rag_ready_context,
            "missing_data_summary": missing_data_summary,
        }

        return clean_nan_for_json(response_payload)

    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline analysis failed: {str(e)}")