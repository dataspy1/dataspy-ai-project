import json
import re
from typing import Dict, Any, List, Optional

from app.services.analytics.conditional_query_service import ConditionalQueryService
from app.services.llm.llm_client import LLMClient


SYSTEM_PROMPT = """
You are an Executive Decision Intelligence Assistant working inside DataSpy Decision AI.

Your role is to answer business questions like a senior strategy consultant.

You help decision-makers understand:
- what is happening
- why it matters
- what should be done
- what needs monitoring

CORE RULES:
1. Use ONLY the provided business analysis context.
2. Do NOT invent numbers, trends, risks, or business conditions.
3. Do NOT assume growth, decline, improvement, or deterioration unless clearly supported.
4. Do NOT speculate.
5. If information is missing, say exactly:
   "This is not available in the current analysis."
6. Never expose internal system details.
7. Keep answers short, business-friendly, and clear.

REASONING STYLE:
- Start with the direct answer.
- Then briefly explain why it matters.
- Link insights, forecast, and decisions where useful.
- Speak like a business advisor, not a raw data reporter.
- Prioritize business meaning over dataset description.
- If confidence is limited, say so clearly.
- If exact detail is unavailable, provide a cautious business answer only if supported.

FORECAST HANDLING:
- Use forecast when available.
- You may explain forecast quality using available forecast signals such as:
  - reliability_label
  - validation_metrics
  - warnings
  - error
  - model_used
  - recommendation / usability flags
- If the user asks why the forecast is weak, unreliable, low-confidence, not recommended, or should be used with caution:
  explain it using those forecast signals.
- Do NOT say "not available" if forecast quality signals exist.
- Only say "This is not available in the current analysis." when no meaningful forecast explanation signals are present at all.
- If direction is not clearly supported, say:
  "Forward-looking projections are available."

DECISION HANDLING:
- Use decisions as the main source for actions.
- Explain why the action matters.
- Mention what may happen if ignored only when supported.

FINAL GOAL:
Your answer should feel like a business advisor, not a chatbot.
"""

MISSING_INFO_MESSAGE = "This is not available in the current analysis."

conditional_query_service = ConditionalQueryService()


def safe_lower(text: Any) -> str:
    return str(text).lower().strip() if text is not None else ""


def get_nested(data: Dict[str, Any], *keys, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and len(value) == 0:
            continue
        return value
    return None


def normalize_bool_text(value: Optional[bool], capability_name: str) -> Optional[str]:
    if value is True:
        return f"{capability_name} capability is enabled in the current analysis."
    if value is False:
        return f"{capability_name} capability is disabled in the current analysis."
    return None


def detect_context_used(question: str) -> List[str]:
    q = safe_lower(question)
    used = []

    if any(word in q for word in [
        "product", "region", "revenue", "insight", "strongest", "top",
        "weak", "weakest", "low performing", "low-performing", "supplier",
        "risk", "opportunity", "priority", "management", "strategy"
    ]):
        used.append("insights")

    if any(word in q for word in [
        "forecast", "trend", "future", "prediction", "reliable",
        "confidence", "caution", "interpret", "recommended"
    ]):
        used.append("forecast")

    if any(word in q for word in [
        "decision", "action", "priority", "urgent", "focus",
        "increase sales", "improve sales", "what should be done", "management",
        "next steps", "strategic", "recommend"
    ]):
        used.append("decisions")

    if any(word in q for word in [
        "logistics", "sales", "inventory", "capability", "enabled", "disabled",
        "stock", "delay", "delivery", "restock", "reorder"
    ]):
        used.append("capabilities")

    if any(word in q for word in ["schema", "column", "field", "mapped", "date column", "target column"]):
        used.append("schema_suggestions")

    if any(word in q for word in ["profile", "quality", "missing", "null", "dataset", "rows", "columns", "duplicate"]):
        used.append("profile")

    if any(word in q for word in ["summary", "headline"]):
        used.append("dashboard_summary")

    return list(dict.fromkeys(used)) if used else ["general"]


def soften_overclaims(text: str) -> str:
    replacements = {
        r"\bdefinitely\b": "likely",
        r"\bcertainly\b": "based on the available analysis",
        r"\bwill surely\b": "may",
        r"\bguaranteed\b": "likely",
        r"\bobviously\b": "clearly",
    }

    cleaned = text or ""
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    return cleaned


def apply_answer_policy(text: str, question: str) -> str:
    cleaned = (text or "").strip()

    if not cleaned:
        return MISSING_INFO_MESSAGE

    lower_text = cleaned.lower()

    if "not available in the current analysis" in lower_text:
        return MISSING_INFO_MESSAGE

    unwanted_starts = [
        "based on the provided analysis context,",
        "based on the analysis context,",
        "according to the analysis context,",
        "from the analysis context,",
        "based on the available analysis context,",
    ]

    for phrase in unwanted_starts:
        if lower_text.startswith(phrase):
            cleaned = cleaned[len(phrase):].strip(" ,.-")
            break

    cleaned = cleaned.replace("\n\n", "\n").strip()

    if len(cleaned) > 900:
        cleaned = cleaned[:900].rsplit(" ", 1)[0].strip() + "..."

    if "why" in safe_lower(question) and not cleaned.endswith("."):
        cleaned += "."

    return cleaned if cleaned else MISSING_INFO_MESSAGE


def build_chat_analysis_context(analysis_context: Dict[str, Any]) -> str:
    dashboard_summary = analysis_context.get("dashboard_summary", {}) or {}
    insights = analysis_context.get("insights", {}) or {}
    sales = insights.get("sales", {}) or {}
    forecast = analysis_context.get("forecast", {}) or {}
    capabilities = analysis_context.get("capabilities", {}) or {}
    logistics = insights.get("logistics", {}) or {}

    decisions_raw = analysis_context.get("decisions", {}) or {}
    if isinstance(decisions_raw, dict):
        top_decisions = decisions_raw.get("top_decisions", [])[:5]
        decision_summary = decisions_raw.get("summary") or decisions_raw.get("decision_summary")
    elif isinstance(decisions_raw, list):
        top_decisions = decisions_raw[:5]
        decision_summary = None
    else:
        top_decisions = []
        decision_summary = None

    # IMPORTANT: keep context light to avoid token overflow
    light_context = {
        "executive_overview": {
            "dataset_name": dashboard_summary.get("dataset_name"),
            "headline_summary": dashboard_summary.get("headline_summary"),
            "executive_summary": dashboard_summary.get("executive_summary"),
            "recommended_next_step": dashboard_summary.get("recommended_next_step"),
            "enabled_capabilities": dashboard_summary.get("enabled_capabilities", []),
        },
        "sales_context": {
            "top_products_by_quantity": sales.get("top_products_by_quantity", [])[:5],
            "top_regions_by_quantity": sales.get("top_regions_by_quantity", [])[:5],

            "resolved_columns": sales.get("resolved_columns", {}),
            "top_products_by_revenue": sales.get("top_products_by_revenue", [])[:3],
            "top_regions_by_revenue": sales.get("top_regions_by_revenue", [])[:3],
            "revenue_trend_summary": sales.get("revenue_trend_summary"),
            "quantity_trend_summary": sales.get("quantity_trend_summary"),
            "product_concentration_summary": sales.get("product_concentration_summary"),
            "region_concentration_summary": sales.get("region_concentration_summary"),
            "weakest_region_summary": sales.get("weakest_region_summary"),

            
        },
        "forecast_context": {
            "target_role": first_non_empty(
                forecast.get("forecast_target_role"),
                forecast.get("target_role"),
            ),
            "target_column": first_non_empty(
                forecast.get("forecast_target_column"),
                forecast.get("target_column"),
            ),
            "forecast_horizon": forecast.get("forecast_horizon"),
            "summary": forecast.get("summary"),
            "reliability_label": forecast.get("reliability_label"),
            "model_used": first_non_empty(
                forecast.get("model_used"),
                forecast.get("model_type"),
            ),
            "validation_metrics": first_non_empty(
                forecast.get("validation_metrics"),
                forecast.get("metrics"),
                {},
            ),
            "warnings": forecast.get("warnings", [])[:3],
            "error": forecast.get("error"),
            "trend_direction": forecast.get("trend_direction"),
            "growth_percent": forecast.get("growth_percent"),
        },
        "decision_context": {
            "summary": decision_summary,
            "top_decisions": top_decisions[:3],
        },
        "logistics_context": {
            "logistics_summary": logistics.get("logistics_summary"),
            "delivery_delay_summary": logistics.get("delivery_delay_summary"),
            "supplier_delay_summary": logistics.get("supplier_delay_summary"),
            "delay_risk": logistics.get("delay_risk"),
        },
        "capabilities": {
            "sales": capabilities.get("sales", {}),
            "inventory": capabilities.get("inventory", {}),
            "logistics": capabilities.get("logistics", {}),
        },
        "forecast_error": analysis_context.get("forecast_error", ""),
    }

    return json.dumps(light_context, default=str)


def get_decision_items(analysis_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    decisions = analysis_context.get("decisions", {})

    if isinstance(decisions, list):
        return [item for item in decisions if isinstance(item, dict)]

    if isinstance(decisions, dict):
        top_decisions = decisions.get("top_decisions", [])
        if isinstance(top_decisions, list):
            return [item for item in top_decisions if isinstance(item, dict)]

    return []


def extract_top_decision_text(analysis_context: Dict[str, Any]) -> Optional[str]:
    decision_items = get_decision_items(analysis_context)
    if not decision_items:
        return None

    first_item = decision_items[0]
    return first_non_empty(
        first_item.get("decision"),
        first_item.get("action"),
        first_item.get("title"),
        first_item.get("recommendation"),
    )


def extract_top_decision_priority(analysis_context: Dict[str, Any]) -> Optional[str]:
    decision_items = get_decision_items(analysis_context)
    if not decision_items:
        return None

    first_item = decision_items[0]
    return first_non_empty(first_item.get("priority"))


def extract_recommended_actions(analysis_context: Dict[str, Any]) -> List[str]:
    decision_items = get_decision_items(analysis_context)
    actions = []

    for item in decision_items:
        action = first_non_empty(
            item.get("decision"),
            item.get("action"),
            item.get("title"),
            item.get("recommendation"),
        )
        if action:
            actions.append(str(action))

    return actions


def extract_top_product(analysis_context: Dict[str, Any]) -> Optional[str]:
    dashboard_top = get_nested(analysis_context, "dashboard_summary", "top_product")
    if isinstance(dashboard_top, dict):
        return first_non_empty(
            dashboard_top.get("name"),
            dashboard_top.get("Product"),
            dashboard_top.get("product"),
        )

    if isinstance(dashboard_top, str) and dashboard_top.strip():
        return dashboard_top.strip()

    sales_products = get_nested(analysis_context, "insights", "sales", "top_products_by_revenue", default=[])
    if isinstance(sales_products, list) and sales_products:
        first_item = sales_products[0]
        if isinstance(first_item, dict):
            return first_non_empty(
                first_item.get("Product"),
                first_item.get("product"),
                first_item.get("name"),
                first_item.get("label"),
                first_item.get("COMBINATION"),
            )

    sales_products_qty = get_nested(analysis_context, "insights", "sales", "top_products_by_quantity", default=[])
    if isinstance(sales_products_qty, list) and sales_products_qty:
        first_item = sales_products_qty[0]
        if isinstance(first_item, dict):
            return first_non_empty(
                first_item.get("Product"),
                first_item.get("product"),
                first_item.get("name"),
                first_item.get("product_name"),
                first_item.get("label"),
                first_item.get("COMBINATION"),
            )

    return None


def extract_top_region(analysis_context: Dict[str, Any]) -> Optional[str]:
    sales_regions = get_nested(analysis_context, "insights", "sales", "top_regions_by_revenue", default=[])
    if isinstance(sales_regions, list) and sales_regions:
        first_item = sales_regions[0]
        if isinstance(first_item, dict):
            return first_non_empty(
                first_item.get("City"),
                first_item.get("city"),
                first_item.get("region"),
                first_item.get("Region"),
                first_item.get("name"),
                first_item.get("DISTRICT"),
                first_item.get("PLACE"),
                first_item.get("label"),
            )

    sales_regions_qty = get_nested(analysis_context, "insights", "sales", "top_regions_by_quantity", default=[])
    if isinstance(sales_regions_qty, list) and sales_regions_qty:
        first_item = sales_regions_qty[0]
        if isinstance(first_item, dict):
            return first_non_empty(
                first_item.get("region"),
                first_item.get("Region"),
                first_item.get("city"),
                first_item.get("name"),
                first_item.get("DISTRICT"),
                first_item.get("PLACE"),
                first_item.get("label"),
            )

    dashboard_region = get_nested(analysis_context, "dashboard_summary", "top_region")
    if isinstance(dashboard_region, str) and dashboard_region.strip():
        return dashboard_region.strip()

    if isinstance(dashboard_region, dict):
        return first_non_empty(
            dashboard_region.get("region"),
            dashboard_region.get("Region"),
            dashboard_region.get("name"),
        )

    return None


def extract_missing_values(profile: Dict[str, Any]) -> Optional[int]:
    null_counts = get_nested(profile, "dataset_profile", "null_counts", default={})
    if isinstance(null_counts, dict):
        try:
            return int(sum(v for v in null_counts.values() if isinstance(v, (int, float))))
        except Exception:
            return None
    return None


def extract_row_count(analysis_context: Dict[str, Any]) -> Optional[int]:
    return first_non_empty(
        get_nested(analysis_context, "dashboard_summary", "rows"),
        get_nested(analysis_context, "profile", "dataset_profile", "total_rows"),
        get_nested(analysis_context, "file_summary", "rows"),
    )


def extract_column_count(analysis_context: Dict[str, Any]) -> Optional[int]:
    file_columns = get_nested(analysis_context, "file_summary", "columns")
    if isinstance(file_columns, list):
        return len(file_columns)

    return first_non_empty(
        get_nested(analysis_context, "dashboard_summary", "columns"),
        get_nested(analysis_context, "profile", "dataset_profile", "total_columns"),
    )


def extract_duplicate_rows(analysis_context: Dict[str, Any]) -> Optional[int]:
    return first_non_empty(
        get_nested(analysis_context, "dashboard_summary", "duplicate_rows"),
        get_nested(analysis_context, "profile", "dataset_profile", "duplicate_rows"),
    )


def extract_date_column(schema_suggestions: Dict[str, Any]) -> Optional[str]:
    return first_non_empty(get_nested(schema_suggestions, "date", "column"))


def extract_target_column(schema_suggestions: Dict[str, Any]) -> Optional[str]:
    return first_non_empty(
        get_nested(schema_suggestions, "revenue", "column"),
        get_nested(schema_suggestions, "quantity", "column"),
        get_nested(schema_suggestions, "target", "column"),
    )


def extract_main_insight(analysis_context: Dict[str, Any]) -> Optional[str]:
    return first_non_empty(
        get_nested(analysis_context, "dashboard_summary", "headline_summary"),
        get_nested(analysis_context, "narratives", "sales_summary"),
        get_nested(analysis_context, "dashboard_summary", "executive_summary"),
    )


def extract_forecast_available(analysis_context: Dict[str, Any]) -> Optional[bool]:
    forecast = analysis_context.get("forecast", {})
    forecast_error = analysis_context.get("forecast_error")

    if isinstance(forecast, dict) and len(forecast) > 0:
        return True

    if forecast_error:
        return False

    return False


def extract_forecast_trend(analysis_context: Dict[str, Any]) -> Optional[str]:
    forecast = analysis_context.get("forecast", {})
    if not forecast:
        return None

    if isinstance(forecast.get("summary"), str) and forecast.get("summary").strip():
        return forecast.get("summary")

    return first_non_empty(
        forecast.get("trend_direction"),
        forecast.get("trend"),
        get_nested(forecast, "summary", "trend_direction"),
    )


def extract_saved_filename(analysis_context: Dict[str, Any]) -> Optional[str]:
    return first_non_empty(
        analysis_context.get("saved_filename"),
        get_nested(analysis_context, "dashboard_summary", "saved_filename"),
        get_nested(analysis_context, "file_summary", "saved_filename"),
        get_nested(analysis_context, "metadata", "saved_filename"),
    )


def build_forecast_weak_explanation(forecast: Dict[str, Any]) -> Optional[str]:
    if not isinstance(forecast, dict) or not forecast:
        return None

    reliability = forecast.get("reliability_label")
    warnings = forecast.get("warnings", [])
    error = forecast.get("error")
    model_used = first_non_empty(forecast.get("model_used"), forecast.get("model_type"))
    recommendation = first_non_empty(
        forecast.get("decision_usability"),
        forecast.get("forecast_recommendation"),
        forecast.get("recommendation"),
    )
    validation_metrics = first_non_empty(
        forecast.get("validation_metrics"),
        forecast.get("metrics"),
        {},
    ) or {}

    reasons = []

    if recommendation:
        reasons.append(f"the forecast is marked as {recommendation}")

    if reliability:
        reliability_text = str(reliability).strip().lower()
        if reliability_text in {"low", "medium", "not reliable", "weak", "cautious", "moderate"}:
            reasons.append(f"the forecast reliability is {reliability}")

    metric_fragments = []
    for metric_name in ["MAE", "mae", "RMSE", "rmse", "MAPE", "mape", "sMAPE", "smape", "WAPE", "wape"]:
        metric_value = validation_metrics.get(metric_name)
        if metric_value is not None:
            metric_fragments.append(f"{metric_name.upper()} is {metric_value}")

    if error:
        reasons.append("the forecast process reported an error condition")
    elif metric_fragments:
        reasons.append("validation signals show forecasting limitations" + f" ({', '.join(metric_fragments[:2])})")

    if warnings:
        reasons.append("warnings indicate data or signal limitations")

    if model_used:
        reasons.append(f"the selected model is {model_used}")

    if not reasons:
        return None

    explanation = ", and ".join(reasons)
    return (
        f"The forecast should be treated cautiously because {explanation}. "
        f"This means it can support directional planning, but it should not be used alone for critical business decisions."
    )


def _format_supplier_delay_answer(summary: Any) -> Optional[str]:
    if isinstance(summary, str) and summary.strip():
        return summary.strip()

    if isinstance(summary, dict):
        highest_supplier = first_non_empty(
            summary.get("highest_delay_supplier"),
            summary.get("worst_supplier"),
            summary.get("supplier_name"),
        )
        highest_delay = first_non_empty(
            summary.get("highest_average_delay_days"),
            summary.get("average_delivery_days"),
            summary.get("delay_days"),
        )

        if highest_supplier and highest_delay is not None:
            return f"{highest_supplier} is showing the highest visible delay pattern, with an average delay of about {highest_delay} day(s)."

        if highest_supplier:
            return f"{highest_supplier} is showing the highest visible delay pattern in the current analysis."

    return None


def is_strategic_question(question: str) -> bool:
    q = safe_lower(question)

    strategic_keywords = [
        "risk", "risks", "business risk", "monitor", "watch closely",
        "priority", "prioritize", "management", "leadership",
        "interpret", "practical terms", "strategic", "strategy",
        "implication", "implications", "focus", "focus on",
        "what matters most", "biggest concern", "main concern",
        "what should management", "what should leadership",
        "short-term", "long-term", "next quarter", "advisory",
        "opportunity", "threat", "caution", "why it matters"
    ]

    return any(keyword in q for keyword in strategic_keywords)


def _extract_name_and_value(item: Dict[str, Any]):
    if not isinstance(item, dict):
        return None, None

    name = first_non_empty(
        item.get("Product"),
        item.get("product"),
        item.get("name"),
        item.get("label"),
        item.get("COMBINATION"),
        item.get("City"),
        item.get("city"),
        item.get("Region"),
        item.get("region"),
        item.get("DISTRICT"),
        item.get("PLACE"),
    )

    value = first_non_empty(
        item.get("value"),
        item.get("Total_Amount"),
        item.get("total_amount"),
        item.get("Revenue"),
        item.get("revenue"),
        item.get("sales"),
        item.get("Quantity"),
        item.get("quantity"),
        item.get("qty"),
        item.get("pending_orders"),
        item.get("count"),
    )

    return name, value


def try_basic_excel_queries(question: str, analysis_context: Dict[str, Any]):
    q = safe_lower(question)

    sales = get_nested(analysis_context, "insights", "sales", default={}) or {}
    inventory = get_nested(analysis_context, "insights", "inventory", default={}) or {}
    logistics = get_nested(analysis_context, "insights", "logistics", default={}) or {}

    top_products_revenue = sales.get("top_products_by_revenue", []) or []
    top_regions_revenue = sales.get("top_regions_by_revenue", []) or []
    top_products_quantity = sales.get("top_products_by_quantity", []) or []
    top_regions_quantity = sales.get("top_regions_by_quantity", []) or []
    
    revenue_trend = sales.get("revenue_trend", []) or []
    quantity_trend = sales.get("quantity_trend", []) or []
    pending_summary = sales.get("pending_order_pattern_summary", {}) or {}
    low_stock_items = inventory.get("low_stock_items", []) or []

    # total revenue
    
    if "total revenue" in q or "overall revenue" in q:
        total = 0.0
        for item in top_products_revenue:
            _, value = _extract_name_and_value(item)
            try:
                total += float(value or 0)
            except Exception:
                continue
        if total > 0:
            return f"Total visible revenue in the current analysis is {total:,.2f}."

    # top product
    if "top product" in q or "best product" in q or "strongest product" in q:
        if top_products_revenue:
            name, value = _extract_name_and_value(top_products_revenue[0])
            if name is not None:
                if value is not None:
                    return f"The top product is {name}, contributing {value} in visible revenue."
                return f"The top product is {name}."

    # top 3 products
    if "top 3 product" in q or "top three product" in q or "which three products" in q:
        names = []
        for item in top_products_revenue[:3]:
            name, _ = _extract_name_and_value(item)
            if name:
                names.append(str(name))
        if names:
            return f"The top 3 contributing products are: {', '.join(names)}."

    # top region
    if "top region" in q or "best region" in q or "strongest region" in q or "maximum revenue region" in q or "region generated maximum revenue" in q or "which region generated maximum revenue" in q or "highest revenue by region" in q:
        if top_regions_revenue:
            name, value = _extract_name_and_value(top_regions_revenue[0])
            if name is not None:
                if value is not None:
                    return f"The top region is {name}, contributing {value} in visible revenue."
                return f"The top region is {name}."

    # top 3 regions
    if "top 3 region" in q or "top three region" in q:
        names = []
        for item in top_regions_revenue[:3]:
            name, _ = _extract_name_and_value(item)
            if name:
                names.append(str(name))
        if names:
            return f"The top 3 regions are: {', '.join(names)}."

    # average value per product
    if "average value per product" in q or "average revenue per product" in q:
        values = []
        for item in top_products_revenue:
            _, value = _extract_name_and_value(item)
            try:
                values.append(float(value))
            except Exception:
                continue
        if values:
            avg = sum(values) / len(values)
            return f"The average visible revenue per product group is {avg:,.2f}."

    # total quantity
    if "total quantity" in q or "overall quantity" in q:
        total_qty = 0.0
        for item in top_products_quantity:
            _, value = _extract_name_and_value(item)
            try:
                total_qty += float(value or 0)
            except Exception:
                continue
        if total_qty > 0:
            return f"Total visible quantity in the current analysis is {total_qty:,.0f}."

    # top quantity product
    if "top quantity product" in q or "highest quantity product" in q:
        if top_products_quantity:
            name, value = _extract_name_and_value(top_products_quantity[0])
            if name is not None:
                return f"The product with the highest visible quantity is {name} at {value}."

    # top quantity region
    # if "top quantity region" in q or "highest quantity region" in q:
    #     if top_regions_quantity:
    #         name, value = _extract_name_and_value(top_regions_quantity[0])
    #         if name is not None:
    #             return f"The region with the highest visible quantity is {name} at {value}."

    if (
        "top quantity region" in q
        or "highest quantity region" in q
        or "maximum quantity" in q
        or "maximum quantity sales" in q
        or "most quantity sold region" in q
    ):
        if top_regions_quantity:
            name, value = _extract_name_and_value(top_regions_quantity[0])
            if name is not None:
                return f"The region with the highest quantity sales is {name} with total quantity of {value}."

    # low stock
    if "low stock" in q or "stock risk" in q or "stock out" in q:
        if isinstance(low_stock_items, list):
            count = len(low_stock_items)
            if count > 0:
                return f"There are {count} low-stock records currently flagged in the analysis."

    # pending orders
    if "pending order" in q or "not delivered" in q or "pending deliveries" in q:
        if isinstance(pending_summary, dict):
            pending_by_region = pending_summary.get("pending_orders_by_region", []) or []
            if pending_by_region:
                total_pending = 0
                for item in pending_by_region:
                    try:
                        total_pending += int(item.get("pending_orders", 0))
                    except Exception:
                        continue
                top_pending_region = pending_summary.get("top_pending_region")
                if top_pending_region:
                    return f"There are {total_pending} visible pending orders, with the highest concentration in {top_pending_region}."
                return f"There are {total_pending} visible pending orders in the current analysis."

    # trend questions
    if "revenue trend" in q or ("trend" in q and "revenue" in q):
        summary = sales.get("revenue_trend_summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()

    if "quantity trend" in q or ("trend" in q and "quantity" in q):
        summary = sales.get("quantity_trend_summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()

    # peak periods from trends
    if "highest revenue day" in q or "peak revenue day" in q:
        if revenue_trend:
            best = None
            for item in revenue_trend:
                date = item.get("date") or item.get("Date") or item.get("ds")
                value = item.get("value") or item.get("Revenue") or item.get("revenue") or item.get("y")
                try:
                    value = float(value)
                except Exception:
                    continue
                if best is None or value > best["value"]:
                    best = {"date": date, "value": value}
            if best:
                return f"The highest visible revenue point is on {best['date']} with a value of {best['value']:,.2f}."

    if "highest quantity day" in q or "peak quantity day" in q:
        if quantity_trend:
            best = None
            for item in quantity_trend:
                date = item.get("date") or item.get("Date") or item.get("ds")
                value = item.get("value") or item.get("Quantity") or item.get("quantity") or item.get("y")
                try:
                    value = float(value)
                except Exception:
                    continue
                if best is None or value > best["value"]:
                    best = {"date": date, "value": value}
            if best:
                return f"The highest visible quantity point is on {best['date']} with a value of {best['value']:,.0f}."

    return None


def try_direct_answer(question: str, analysis_context: Dict[str, Any]):
    q = safe_lower(question)

    if is_strategic_question(q):
        return None

    schema_suggestions = analysis_context.get("schema_suggestions", {}) or {}
    capabilities = analysis_context.get("capabilities", {}) or {}
    profile = analysis_context.get("profile", {}) or {}
    sales = get_nested(analysis_context, "insights", "sales", default={}) or {}
    inventory = get_nested(analysis_context, "insights", "inventory", default={}) or {}
    logistics = get_nested(analysis_context, "insights", "logistics", default={}) or {}
    forecast = analysis_context.get("forecast", {}) or {}

    if "stock" in q and "reorder" in q:
        return {
            "answer": (
                "Stock refers to the current inventory available, while reorder level is the minimum threshold at which replenishment should be triggered. "
                "If stock falls below reorder level, the business should consider reordering to reduce stock-out risk."
            ),
            "context_used": ["inventory", "capabilities"],
        }

    if "what is stock" in q or "meaning of stock" in q:
        return {
            "answer": (
                "Stock refers to the quantity currently available in inventory. "
                "It shows how much product is presently on hand for sales or fulfillment."
            ),
            "context_used": ["inventory"],
        }

    if "what is reorder" in q or "reorder level" in q:
        return {
            "answer": (
                "Reorder level is the minimum stock threshold that signals when new inventory should be ordered. "
                "It helps prevent stock-outs and supports smoother replenishment planning."
            ),
            "context_used": ["inventory"],
        }

    if (
        "shipment date" in q and ("missing" in q or "not available" in q)
    ) or (
        "will system work" in q and "shipment" in q
    ) or (
        "without shipment date" in q
    ):
        logistics_enabled = get_nested(capabilities, "logistics", "enabled")
        missing_roles = get_nested(capabilities, "logistics", "missing_roles", default=[])

        if logistics_enabled is False and missing_roles:
            return {
                "answer": (
                    "The system can still continue with other supported analyses, but logistics capability will be limited because required logistics fields are missing. "
                    f"Currently missing logistics roles include: {', '.join(missing_roles)}."
                ),
                "context_used": ["capabilities"],
            }

        return {
            "answer": (
                "Yes, the system can still continue with other supported analyses when shipment date is missing. "
                "However, logistics-related outputs may be limited because shipment timing is an important logistics signal."
            ),
            "context_used": ["capabilities", "logistics"],
        }

    if (
        "strongest product" in q
        or ("product" in q and "strongest" in q)
        or "top product" in q
        or "best product" in q
        or "highest revenue" in q
        or "most revenue" in q
        or "highest sales" in q
        or "top selling product" in q
    ):
        top_product = extract_top_product(analysis_context)
        if top_product:
            return {
                "answer": (
                    f"The strongest visible product in the current analysis is {top_product}. "
                    f"This indicates that {top_product} is currently the leading product signal based on available insights."
                ),
                "context_used": ["dashboard_summary", "insights"],
            }

    if (
        "weak product" in q
        or "weakest product" in q
        or "low product" in q
        or "low-performing product" in q
        or "worst product" in q
    ):
        product_concentration = sales.get("product_concentration_summary")
        return {
            "answer": (
                "Based on available insights, weaker-performing products should be reviewed for improvement, repositioning, or portfolio cleanup. "
                f"{product_concentration if isinstance(product_concentration, str) and product_concentration.strip() else 'The current analysis does not name one exact weakest product clearly, so management should review lower-performing product segments in the insights layer.'}"
            ),
            "context_used": ["insights"],
        }

    if (
        "top region" in q
        or ("region" in q and "top" in q)
        or "leading region" in q
        or "best region" in q
        or "strongest region" in q
        or "highest revenue region" in q
    ):
        top_region = extract_top_region(analysis_context)
        if top_region:
            return {
                "answer": (
                    f"The top-performing region in the current analysis is {top_region}. "
                    f"This suggests that {top_region} is currently showing the strongest regional performance signal."
                ),
                "context_used": ["insights"],
            }

    if (
        "weak region" in q
        or "weakest region" in q
        or "low region" in q
        or "worst region" in q
    ):
        weakest_region_summary = sales.get("weakest_region_summary")
        if weakest_region_summary:
            return {
                "answer": (
                    f"Based on available insights, {weakest_region_summary} "
                    f"This area may need focused review to understand whether pricing, demand, supply, or channel issues are affecting performance."
                ),
                "context_used": ["insights"],
            }

    if "urgent decision" in q or "most urgent decision" in q or "top decision" in q:
        decision_text = extract_top_decision_text(analysis_context)
        priority = extract_top_decision_priority(analysis_context)

        if decision_text and priority:
            return {
                "answer": (
                    f"The most urgent decision is {decision_text}. "
                    f"It is marked as {priority} priority, which means it should be reviewed before lower-priority actions."
                ),
                "context_used": ["decisions"],
            }

        if decision_text:
            return {
                "answer": (
                    f"The most urgent visible decision is {decision_text}. "
                    f"This should be treated as an important action area based on the current analysis."
                ),
                "context_used": ["decisions"],
            }

    if "priority" in q and "decision" in q:
        decision_text = extract_top_decision_text(analysis_context)
        priority = extract_top_decision_priority(analysis_context)

        if decision_text and priority:
            return {
                "answer": f"The top decision is {decision_text}, with {priority} priority.",
                "context_used": ["decisions"],
            }

    if "forecast" in q and "available" in q:
        available = extract_forecast_available(analysis_context)
        if available is True:
            return {
                "answer": "Yes, forecast information is available in the current analysis.",
                "context_used": ["forecast"],
            }
        return {
            "answer": "No, forecast information is not available in the current analysis.",
            "context_used": ["forecast", "forecast_error"],
        }

    if (
        ("why" in q and "forecast" in q)
        or "forecast is weak" in q
        or "weak forecast" in q
        or "forecast weak" in q
        or "why is the forecast weak" in q
        or "why forecast is weak" in q
        or "why is forecast weak" in q
        or "why is the prediction weak" in q
        or "why should this be used with caution" in q
        or "why is forecast reliability low" in q
        or "why is forecast reliability medium" in q
        or "forecast not recommended" in q
        or "not recommended for decisions" in q
    ):
        explanation = build_forecast_weak_explanation(forecast)

        if explanation:
            return {
                "answer": explanation,
                "context_used": ["forecast"],
            }

        forecast_error = analysis_context.get("forecast_error")
        if forecast_error:
            return {
                "answer": f"Forecast quality is limited because {forecast_error}.",
                "context_used": ["forecast_error"],
            }

    if "forecast" in q and ("reliable" in q or "confidence" in q):
        reliability = forecast.get("reliability_label")
        warnings = forecast.get("warnings", [])
        recommendation = first_non_empty(
            forecast.get("decision_usability"),
            forecast.get("forecast_recommendation"),
            forecast.get("recommendation"),
        )

        if reliability or recommendation:
            parts = []
            if recommendation:
                parts.append(f"The forecast is currently marked as {recommendation}.")
            if reliability:
                parts.append(f"Forecast reliability is {reliability}.")
            if warnings:
                parts.append("Warnings are present indicating potential limitations in prediction quality.")
            parts.append("This means results should be interpreted with appropriate caution.")

            return {
                "answer": " ".join(parts),
                "context_used": ["forecast"],
            }

    if "forecast trend" in q or ("trend" in q and "forecast" in q) or "explain the forecast" in q:
        trend = extract_forecast_trend(analysis_context)

        target_role = first_non_empty(
            forecast.get("forecast_target_role"),
            forecast.get("target_role"),
        )
        target_column = first_non_empty(
            forecast.get("forecast_target_column"),
            forecast.get("target_column"),
        )
        horizon = forecast.get("forecast_horizon")
        latest_actual = forecast.get("latest_actual_value")
        average_forecast = first_non_empty(
            forecast.get("average_forecast_value"),
            forecast.get("average_forecast"),
        )

        if trend:
            answer_parts = [str(trend).strip()]

            details = []
            if target_role:
                details.append(f"the selected business target is {target_role}")
            if target_column:
                details.append(f"the forecast is based on {target_column}")
            if horizon is not None:
                details.append(f"the projection covers {horizon} future period(s)")
            if latest_actual is not None:
                details.append(f"the latest actual value available is {latest_actual}")
            if average_forecast is not None:
                details.append(f"the average forecast value is {average_forecast}")

            if details:
                answer_parts.append("Based on the available forecast details, " + ", ".join(details) + ".")

            answer_parts.append("This helps management review the near-term outlook for the selected metric.")

            return {
                "answer": " ".join(answer_parts),
                "context_used": ["forecast"],
            }

        return {
            "answer": MISSING_INFO_MESSAGE,
            "context_used": ["forecast", "forecast_error"],
        }

    if "logistics" in q and ("enabled" in q or "disabled" in q):
        logistics_enabled = get_nested(capabilities, "logistics", "enabled")
        answer = normalize_bool_text(logistics_enabled, "Logistics")
        if answer:
            return {"answer": answer, "context_used": ["capabilities"]}

    if "sales" in q and ("enabled" in q or "disabled" in q):
        sales_enabled = get_nested(capabilities, "sales", "enabled")
        answer = normalize_bool_text(sales_enabled, "Sales")
        if answer:
            return {"answer": answer, "context_used": ["capabilities"]}

    if "inventory" in q and ("enabled" in q or "disabled" in q):
        inventory_enabled = get_nested(capabilities, "inventory", "enabled")
        answer = normalize_bool_text(inventory_enabled, "Inventory")
        if answer:
            return {"answer": answer, "context_used": ["capabilities"]}

    if "supplier" in q and ("delay" in q or "risk" in q or "late" in q):
        supplier_summary = first_non_empty(
            logistics.get("supplier_delay_summary"),
            logistics.get("delivery_delay_summary"),
            logistics.get("logistics_summary"),
        )

        formatted = _format_supplier_delay_answer(supplier_summary)
        if formatted:
            return {
                "answer": (
                    f"{formatted} "
                    f"This indicates that supplier performance may affect delivery timelines and should be monitored closely."
                ),
                "context_used": ["insights", "logistics"],
            }

    if "which supplier" in q and ("avoid" in q or "risky" in q):
        supplier_summary = first_non_empty(
            logistics.get("supplier_delay_summary"),
            logistics.get("delivery_delay_summary"),
            logistics.get("logistics_summary"),
        )

        formatted = _format_supplier_delay_answer(supplier_summary)
        if formatted:
            return {
                "answer": (
                    f"{formatted} "
                    f"Suppliers showing the highest delays should be reviewed or deprioritized to reduce operational risk."
                ),
                "context_used": ["insights", "logistics"],
            }

    if "stock" in q and ("risk" in q or "out" in q or "shortage" in q):
        inventory_summary = inventory.get("inventory_summary")
        pressure = inventory.get("inventory_pressure_level")

        inv_text = None
        if isinstance(inventory_summary, dict):
            inv_text = first_non_empty(
                inventory_summary.get("inventory_summary"),
                inventory_summary.get("summary"),
            )
        elif isinstance(inventory_summary, str):
            inv_text = inventory_summary

        if inv_text or pressure:
            return {
                "answer": (
                    f"{inv_text if inv_text else 'Stock conditions indicate pressure in certain areas.'} "
                    f"{'Current inventory pressure level is ' + str(pressure) + '. ' if pressure else ''}"
                    f"This suggests potential stock-out risk if demand remains firm or supply slows further."
                ),
                "context_used": ["insights", "inventory"],
            }

    if "why this decision" in q or "why this action" in q:
        decision_items = get_decision_items(analysis_context)

        if decision_items:
            first_item = decision_items[0]
            rationale = first_non_empty(
                first_item.get("rationale"),
                first_item.get("explanation"),
            )

            if rationale:
                return {
                    "answer": (
                        f"This decision is recommended because {rationale} "
                        f"This reflects the strongest actionable signal from the current analysis."
                    ),
                    "context_used": ["decisions"],
                }

    if "missing values" in q or "null values" in q or "data quality" in q:
        missing_values = extract_missing_values(profile)
        duplicate_rows = extract_duplicate_rows(analysis_context)

        if missing_values is not None and duplicate_rows is not None:
            return {
                "answer": f"The dataset contains {missing_values} missing values overall and {duplicate_rows} duplicate rows.",
                "context_used": ["profile", "dashboard_summary"],
            }

        if missing_values is not None:
            return {
                "answer": f"The dataset contains {missing_values} missing values overall.",
                "context_used": ["profile"],
            }

    if "duplicate rows" in q or "duplicates" in q:
        duplicate_rows = extract_duplicate_rows(analysis_context)
        if duplicate_rows is not None:
            return {
                "answer": f"The dataset contains {duplicate_rows} duplicate rows.",
                "context_used": ["dashboard_summary", "profile"],
            }

    if "row count" in q or "how many rows" in q:
        row_count = extract_row_count(analysis_context)
        if row_count is not None:
            return {
                "answer": f"The dataset contains {row_count} rows.",
                "context_used": ["dashboard_summary", "profile"],
            }

    if "column count" in q or "how many columns" in q:
        column_count = extract_column_count(analysis_context)
        if column_count is not None:
            return {
                "answer": f"The dataset contains {column_count} columns.",
                "context_used": ["dashboard_summary", "profile", "file_summary"],
            }

    if "dataset summary" in q or "summarize the dataset" in q:
        row_count = extract_row_count(analysis_context)
        column_count = extract_column_count(analysis_context)
        date_column = extract_date_column(schema_suggestions)
        target_column = extract_target_column(schema_suggestions)

        parts = []
        if row_count is not None:
            parts.append(f"{row_count} rows")
        if column_count is not None:
            parts.append(f"{column_count} columns")
        if date_column:
            parts.append(f"date column: {date_column}")
        if target_column:
            parts.append(f"target column: {target_column}")

        if parts:
            return {
                "answer": (
                    "The current dataset includes " + ", ".join(parts) + ". "
                    "This provides a structured base for generating insights, forecast support, and decision recommendations."
                ),
                "context_used": ["dashboard_summary", "profile", "schema_suggestions"],
            }

    if "main insight" in q or "top insight" in q or "key insight" in q or "headline summary" in q:
        summary = extract_main_insight(analysis_context)
        if summary:
            return {
                "answer": (
                    f"The main business insight from the current analysis is: {summary} "
                    f"This represents the strongest overall takeaway currently available."
                ),
                "context_used": ["dashboard_summary", "narratives"],
            }

    if "why" in q and "inventory" in q and ("not enabled" in q or "disabled" in q):
        inventory_missing = get_nested(capabilities, "inventory", "missing_roles", default=[])
        if inventory_missing:
            return {
                "answer": "Inventory capability is disabled because these required fields are missing: " + ", ".join(inventory_missing) + ".",
                "context_used": ["capabilities"],
            }

    if "why" in q and "logistics" in q and ("not enabled" in q or "disabled" in q):
        logistics_missing = get_nested(capabilities, "logistics", "missing_roles", default=[])
        if logistics_missing:
            return {
                "answer": "Logistics capability is disabled because these required fields are missing: " + ", ".join(logistics_missing) + ".",
                "context_used": ["capabilities"],
            }

    if "why" in q and "forecast" in q and ("not available" in q or "missing" in q or "failed" in q):
        forecast_error = analysis_context.get("forecast_error")
        if forecast_error:
            return {
                "answer": f"Forecast is not available because: {forecast_error}.",
                "context_used": ["forecast_error"],
            }

    return None


def build_strategic_fallback(question: str, analysis_context: Dict[str, Any]) -> Optional[str]:
    q = safe_lower(question)

    top_decision = extract_top_decision_text(analysis_context)
    top_product = extract_top_product(analysis_context)
    top_region = extract_top_region(analysis_context)
    main_insight = extract_main_insight(analysis_context)

    sales = get_nested(analysis_context, "insights", "sales", default={}) or {}
    logistics = get_nested(analysis_context, "insights", "logistics", default={}) or {}
    forecast = analysis_context.get("forecast", {}) or {}

    product_concentration = sales.get("product_concentration_summary")
    weakest_region = sales.get("weakest_region_summary")
    logistics_risk = logistics.get("delay_risk")
    forecast_reliability = forecast.get("reliability_label")
    forecast_summary = forecast.get("summary")

    if "risk" in q or "monitor" in q or "concern" in q:
        risk_parts = []

        if isinstance(product_concentration, str) and product_concentration.strip():
            risk_parts.append(product_concentration.strip())

        if isinstance(weakest_region, str) and weakest_region.strip():
            risk_parts.append(weakest_region.strip())

        if logistics_risk:
            risk_parts.append(f"Logistics delay risk is currently {logistics_risk}.")

        if forecast_reliability in {"low", "medium", "weak", "not reliable", "cautious", "moderate"}:
            risk_parts.append(f"Forecast reliability is {forecast_reliability}, so forward-looking decisions should be treated with caution.")

        if risk_parts:
            return "The main business risk to monitor is: " + " ".join(risk_parts[:3])

    if "interpret" in q and "forecast" in q:
        if forecast_summary or forecast_reliability:
            parts = []
            if forecast_summary:
                parts.append(str(forecast_summary).strip())
            if forecast_reliability:
                parts.append(f"Forecast reliability is {forecast_reliability}.")
            parts.append("Management should use this as directional guidance, not as a fully certain operating plan.")
            return " ".join(parts)

    if "priority" in q or "prioritize" in q or "management" in q or "focus" in q:
        priority_parts = []

        if top_decision:
            priority_parts.append(f"The first priority should be {top_decision}.")

        if top_product:
            priority_parts.append(f"{top_product} appears to be a strong product signal worth protecting or scaling.")

        if top_region:
            priority_parts.append(f"{top_region} appears to be the strongest visible region in the current analysis.")

        if priority_parts:
            return " ".join(priority_parts[:3])

    if main_insight:
        return str(main_insight).strip()

    return None


def generate_chat_answer(question: str, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
    if not question or not str(question).strip():
        raise ValueError("Question is required.")

    if not analysis_context or not isinstance(analysis_context, dict):
        raise ValueError("Analysis context is required.")

    question = str(question).strip()

    # =========================================
    # 1. Conditional / Excel-style analytics
    # =========================================
    saved_filename = extract_saved_filename(analysis_context)
    if saved_filename:
        try:
            conditional_result = conditional_query_service.answer_query(
                question=question,
                saved_filename=saved_filename,
                analysis_context=analysis_context,
            )
            if conditional_result:
                return conditional_result
        except Exception as e:
            print(f"[CONDITIONAL_QUERY_ERROR] {str(e)}")
            if "duplicate" in str(e).lower():
                print("[FIX] Duplicate columns detected, conditional engine skipped safely.")

    # =========================================
    # 2. Basic Excel fallback
    # =========================================
    basic_excel_answer = try_basic_excel_queries(question, analysis_context)
    if basic_excel_answer:
        return {
            "answer": basic_excel_answer,
            "context_used": ["insights"],
            "answer_source": "Direct From Data",
        }

    # =========================================
    # 3. Direct answer rules
    # =========================================
    direct_result = try_direct_answer(question, analysis_context)
    if direct_result:
        return {
            "answer": direct_result["answer"],
            "context_used": direct_result["context_used"],
            "answer_source": "Direct From Data",
        }

    # =========================================
    # 4. LLM answer
    # =========================================
    compact_context = build_chat_analysis_context(analysis_context)

    user_prompt = f"""
User Question:
{question}

Business Analysis Context:
{compact_context}

Task:
Answer the user's question using only the business analysis context above.

Answer style requirements:
- Start with the direct answer.
- Then briefly explain the business meaning if useful.
- Keep the answer natural, business-friendly, and concise.
- Do not speculate.
- Do NOT recommend discontinuing products, removing regions, or taking irreversible business actions unless the analysis explicitly supports that recommendation.
- Do not invent numbers or trends.
- If the answer is not available, say exactly: "{MISSING_INFO_MESSAGE}"
- Do not mention internal section names unless necessary for clarity.
- For strategic questions, give a management-oriented answer rather than a raw dataset summary.
- Focus on business meaning, risk, action, and implication.
- If the user asks why forecast quality is weak, low, medium, unreliable, not recommended, or should be used with caution:
  explain it using available forecast signals such as reliability, validation metrics, warnings, model_used, recommendation, or error.
- Do not reject such forecast-quality questions as unavailable when those forecast signals exist.
"""

    try:
        llm_client = LLMClient()
        llm_response = llm_client.generate_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt
        )

        final_answer = soften_overclaims((llm_response or "").strip())
        final_answer = apply_answer_policy(final_answer, question)

        return {
            "answer": final_answer,
            "context_used": detect_context_used(question),
            "answer_source": "Data Through LLM",
        }

    except Exception as e:
        print(f"[CHAT_SERVICE_ERROR] {str(e)}")

        strategic_fallback = build_strategic_fallback(question, analysis_context)
        if strategic_fallback:
            return {
                "answer": strategic_fallback,
                "context_used": detect_context_used(question),
                "answer_source": "Fallback From Data",
            }

        fallback_actions = extract_recommended_actions(analysis_context)
        top_decision = extract_top_decision_text(analysis_context)
        top_product = extract_top_product(analysis_context)
        main_insight = extract_main_insight(analysis_context)

        q = safe_lower(question)

        if (
            "decision" in q
            or "next step" in q
            or "next steps" in q
            or "action" in q
            or "management" in q
            or "top 3 decisions" in q
            or "top three decisions" in q
            or "next 2 steps" in q
            or "next two steps" in q
        ):
            if fallback_actions:
                return {
                    "answer": (
                        "The strongest management actions currently visible are: "
                        + "; ".join(fallback_actions[:3])
                        + "."
                    ),
                    "context_used": ["decisions"],
                    "answer_source": "Fallback From Data",
                }

            if top_decision:
                return {
                    "answer": f"The most visible action in the current analysis is {top_decision}.",
                    "context_used": ["decisions"],
                    "answer_source": "Fallback From Data",
                }

        if "product" in q and ("decision" in q or "action" in q or "management" in q):
            if top_product:
                return {
                    "answer": (
                        f"Management should closely review the leading product signal around {top_product} "
                        f"and align action based on current decision priorities."
                    ),
                    "context_used": ["insights", "decisions"],
                    "answer_source": "Fallback From Data",
                }

        if main_insight:
            return {
                "answer": main_insight,
                "context_used": ["dashboard_summary", "insights"],
                "answer_source": "Fallback From Data",
            }

        return {
            "answer": "Unable to generate a detailed answer right now. Please try again after re-running the analysis.",
            "context_used": detect_context_used(question),
            "answer_source": "System Fallback",
        }