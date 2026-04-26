from typing import Dict, Any, Optional, Tuple
import pandas as pd


def _extract_mapped_column(
    schema_suggestions: Dict[str, Any],
    role: str
) -> Tuple[Optional[str], float]:
    role_data = schema_suggestions.get(role, {})

    if isinstance(role_data, dict):
        return role_data.get("column"), float(role_data.get("confidence", 0) or 0)

    if isinstance(role_data, str):
        return role_data, 1.0

    return None, 0.0


def _resolve_date_column(
    df: pd.DataFrame,
    schema_suggestions: Dict[str, Any]
) -> Tuple[Optional[str], float, str]:
    mapped_col, confidence = _extract_mapped_column(schema_suggestions, "date")
    if mapped_col and mapped_col in df.columns:
        return mapped_col, confidence, "schema_mapping"

    fallback_candidates = [
        "date",
        "Date",
        "order_date",
        "Order_Date",
        "sales_date",
        "transaction_date"
    ]
    for col in fallback_candidates:
        if col in df.columns:
            return col, 0.75, "fallback_exact"

    for col in df.columns:
        normalized = str(col).strip().lower()
        if "date" in normalized:
            return col, 0.65, "fallback_heuristic"

    return None, 0.0, "not_found"


def _resolve_target_column(
    df: pd.DataFrame,
    schema_suggestions: Dict[str, Any],
    target_role: str
) -> Tuple[Optional[str], float, str]:
    mapped_col, confidence = _extract_mapped_column(schema_suggestions, target_role)
    if mapped_col and mapped_col in df.columns:
        return mapped_col, confidence, "schema_mapping"

    role_fallbacks = {
        "revenue": ["revenue", "sales", "amount", "total_amount", "net_total"],
        "quantity": ["sales_qty", "quantity", "qty", "units", "demand_qty"],
        "stock": ["stock", "inventory", "inventory_level"]
    }

    for col in role_fallbacks.get(target_role, []):
        if col in df.columns:
            return col, 0.75, "fallback_exact"

    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_candidates:
        return numeric_candidates[0], 0.55, "fallback_numeric"

    return None, 0.0, "not_found"


def _safe_horizon(forecast_horizon: int) -> int:
    if not isinstance(forecast_horizon, int):
        return 3
    if forecast_horizon <= 0:
        return 3
    if forecast_horizon > 12:
        return 12
    return forecast_horizon


def prepare_time_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str
) -> pd.DataFrame:
    temp_df = df.copy()

    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df[target_col] = pd.to_numeric(temp_df[target_col], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_col, target_col])

    if temp_df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    temp_df["period"] = temp_df[date_col].dt.to_period("M").dt.to_timestamp()

    ts_df = (
        temp_df.groupby("period")[target_col]
        .sum()
        .reset_index()
        .sort_values("period")
    )

    ts_df.columns = ["ds", "y"]
    ts_df["ds"] = pd.to_datetime(ts_df["ds"])

    return ts_df.reset_index(drop=True)


def moving_average_forecast(
    ts_df: pd.DataFrame,
    horizon: int = 3,
    window: int = 3
) -> pd.DataFrame:
    if ts_df.empty:
        return pd.DataFrame(columns=["ds", "yhat"])

    working_values = ts_df["y"].tolist()
    last_date = ts_df["ds"].max()

    forecasts = []
    effective_window = min(window, len(working_values))
    if effective_window == 0:
        return pd.DataFrame(columns=["ds", "yhat"])

    for step in range(1, horizon + 1):
        forecast_value = sum(working_values[-effective_window:]) / effective_window
        forecast_date = last_date + pd.DateOffset(months=step)

        forecasts.append({
            "ds": forecast_date,
            "yhat": round(float(forecast_value), 2)
        })

        working_values.append(forecast_value)

    return pd.DataFrame(forecasts)


def generate_forecast(
    df: pd.DataFrame,
    schema_suggestions: Dict[str, Any],
    target_role: str = "quantity",
    forecast_horizon: int = 3
) -> Dict[str, Any]:
    try:
        forecast_horizon = _safe_horizon(forecast_horizon)

        if df is None or df.empty:
            return {
                "target_role": target_role,
                "target_column": None,
                "date_column": None,
                "forecast_target_role": target_role,
                "forecast_target_column": None,
                "forecast_date_column": None,
                "model_type": "moving_average_baseline",
                "forecast_horizon": forecast_horizon,
                "historical_points": 0,
                "forecast_points": 0,
                "latest_actual_value": None,
                "average_forecast_value": None,
                "average_forecast": None,
                "historical_series": [],
                "forecast_series": [],
                "summary": "Forecast could not be generated because dataset is empty.",
                "error": "Empty dataset",
                "reliability_label": "low",
                "warnings": ["Dataset is empty."],
            }

        date_col, date_confidence, date_source = _resolve_date_column(df, schema_suggestions)

        # Prefer requested role, but if revenue is missing and quantity exists, fallback to quantity
        target_col, target_confidence, target_source = _resolve_target_column(df, schema_suggestions, target_role)

        if not target_col and target_role == "revenue":
            target_col, target_confidence, target_source = _resolve_target_column(df, schema_suggestions, "quantity")
            if target_col:
                target_role = "quantity"

        if not date_col:
            return {
                "target_role": target_role,
                "target_column": None,
                "date_column": None,
                "forecast_target_role": target_role,
                "forecast_target_column": None,
                "forecast_date_column": None,
                "model_type": "moving_average_baseline",
                "forecast_horizon": forecast_horizon,
                "historical_points": 0,
                "forecast_points": 0,
                "latest_actual_value": None,
                "average_forecast_value": None,
                "average_forecast": None,
                "historical_series": [],
                "forecast_series": [],
                "summary": "Forecast could not be generated because no date column was resolved.",
                "error": "No valid date column available for forecasting",
                "reliability_label": "low",
                "warnings": ["No valid date column available."],
            }

        if not target_col:
            return {
                "target_role": target_role,
                "target_column": None,
                "date_column": date_col,
                "forecast_target_role": target_role,
                "forecast_target_column": None,
                "forecast_date_column": date_col,
                "model_type": "moving_average_baseline",
                "forecast_horizon": forecast_horizon,
                "historical_points": 0,
                "forecast_points": 0,
                "latest_actual_value": None,
                "average_forecast_value": None,
                "average_forecast": None,
                "historical_series": [],
                "forecast_series": [],
                "summary": f"Forecast could not be generated because no target column was resolved for role '{target_role}'.",
                "error": f"No valid target column available for role: {target_role}",
                "reliability_label": "low",
                "warnings": [f"No valid target column available for role: {target_role}."],
            }

        ts_df = prepare_time_series(df, date_col, target_col)

        if len(ts_df) < 3:
            return {
                "target_role": target_role,
                "target_column": target_col,
                "date_column": date_col,
                "forecast_target_role": target_role,
                "forecast_target_column": target_col,
                "forecast_date_column": date_col,
                "model_type": "moving_average_baseline",
                "forecast_horizon": forecast_horizon,
                "historical_points": len(ts_df),
                "forecast_points": 0,
                "latest_actual_value": None,
                "average_forecast_value": None,
                "average_forecast": None,
                "historical_series": [],
                "forecast_series": [],
                "summary": "Forecast could not be generated because there are not enough monthly time-series points.",
                "error": "Not enough monthly time series data points for forecasting",
                "reliability_label": "low",
                "warnings": ["At least 3 monthly points are needed for forecasting."],
            }

        ts_df = (
            ts_df.set_index("ds")
            .asfreq("MS")
            .fillna(0)
            .reset_index()
        )

        forecast_df = moving_average_forecast(
            ts_df=ts_df,
            horizon=forecast_horizon,
            window=min(3, len(ts_df))
        )

        historical_series = [
            {
                "ds": row["ds"].strftime("%Y-%m-%d"),
                "y": round(float(row["y"]), 2)
            }
            for _, row in ts_df.iterrows()
        ]

        forecast_series = [
            {
                "ds": row["ds"].strftime("%Y-%m-%d"),
                "yhat": round(float(row["yhat"]), 2)
            }
            for _, row in forecast_df.iterrows()
        ]

        latest_actual = historical_series[-1]["y"] if historical_series else None
        avg_forecast = round(float(forecast_df["yhat"].mean()), 2) if not forecast_df.empty else None

        first_y = ts_df.iloc[0]["y"]
        last_y = ts_df.iloc[-1]["y"]
        if first_y == 0:
            growth_percent = 0.0
        else:
            growth_percent = ((last_y - first_y) / first_y) * 100.0

        if len(ts_df) >= 6:
            reliability_label = "medium"
        else:
            reliability_label = "low"

        warnings = []
        if len(ts_df) < 6:
            warnings.append("Forecast is based on limited monthly history.")
        if target_role == "quantity":
            warnings.append("Forecast is based on quantity because revenue was not mapped or available.")

        trend_label = "increasing" if growth_percent > 10 else "decreasing" if growth_percent < -10 else "stable"

        return {
            "target_role": target_role,
            "target_column": target_col,
            "date_column": date_col,
            "forecast_target_role": target_role,
            "forecast_target_column": target_col,
            "forecast_date_column": date_col,
            "target_confidence": round(float(target_confidence), 2),
            "date_confidence": round(float(date_confidence), 2),
            "target_resolution_source": target_source,
            "date_resolution_source": date_source,
            "model_type": "moving_average_baseline",
            "forecast_horizon": forecast_horizon,
            "historical_points": len(historical_series),
            "forecast_points": len(forecast_series),
            "latest_actual_value": latest_actual,
            "average_forecast_value": avg_forecast,
            "average_forecast": avg_forecast,
            "historical_series": historical_series,
            "forecast_series": forecast_series,
            "trend_direction": trend_label,
            "growth_percent": round(growth_percent, 2),
            "summary": f"Monthly forecast generated for {target_role} using {target_col}. The recent trend appears {trend_label}.",
            "error": None,
            "reliability_label": reliability_label,
            "warnings": warnings,
        }

    except Exception as e:
        return {
            "target_role": target_role,
            "target_column": None,
            "date_column": None,
            "forecast_target_role": target_role,
            "forecast_target_column": None,
            "forecast_date_column": None,
            "model_type": "moving_average_baseline",
            "forecast_horizon": forecast_horizon,
            "historical_points": 0,
            "forecast_points": 0,
            "latest_actual_value": None,
            "average_forecast_value": None,
            "average_forecast": None,
            "historical_series": [],
            "forecast_series": [],
            "summary": "Forecast generation failed.",
            "error": str(e),
            "reliability_label": "low",
            "warnings": [str(e)],
        }