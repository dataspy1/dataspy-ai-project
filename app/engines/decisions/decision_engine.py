from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


def _safe_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    temp = df.copy()
    for col in columns:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
    return temp


def _pick_column(
    schema_suggestions: Optional[Dict[str, Any]],
    role: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    if not isinstance(schema_suggestions, dict):
        return fallback

    role_data = schema_suggestions.get(role, {})

    if isinstance(role_data, dict):
        column = role_data.get("column") or role_data.get("name")
        if column:
            return str(column).strip()

    if isinstance(role_data, str) and role_data.strip():
        return role_data.strip()

    return fallback


def _get_role_confidence(
    schema_suggestions: Optional[Dict[str, Any]],
    role: str,
    default: float = 0.65,
) -> float:
    if not isinstance(schema_suggestions, dict):
        return default

    role_data = schema_suggestions.get(role, {})
    if isinstance(role_data, dict):
        value = role_data.get("confidence", default)
        try:
            value = float(value)
            if value > 1:
                value = value / 100.0
            return max(0.0, min(1.0, value))
        except Exception:
            return default

    return default


def _extract_capability_info(
    capability_result: Optional[Dict[str, Any]],
    capability_name: str,
) -> Dict[str, Any]:
    default = {
        "status": "unknown",
        "confidence": 0.60,
        "enable_reason": "",
        "role_breakdown": {},
    }

    if not isinstance(capability_result, dict):
        return default

    capabilities = capability_result.get("capabilities", capability_result)
    if not isinstance(capabilities, dict):
        return default

    info = capabilities.get(capability_name, {})
    if not isinstance(info, dict):
        return default

    confidence = info.get("confidence", 0.60)
    try:
        confidence = float(confidence)
        if confidence > 1:
            confidence = confidence / 100.0
    except Exception:
        confidence = 0.60

    return {
        "status": str(info.get("status", "unknown")).lower(),
        "confidence": max(0.0, min(1.0, confidence)),
        "enable_reason": str(info.get("enable_reason", "")),
        "role_breakdown": info.get("role_breakdown", {}) or {},
    }


def _dataset_quality_summary(
    df: pd.DataFrame,
    schema_suggestions: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "row_count": 0,
            "column_count": 0,
            "duplicate_rows": 0,
            "missing_percent_overall": 100.0,
            "quality_score": 0.0,
            "quality_label": "poor",
            "warnings": ["Dataset is empty."],
        }

    row_count = len(df)
    column_count = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())

    total_cells = max(row_count * max(column_count, 1), 1)
    total_missing = int(df.isna().sum().sum())
    missing_percent_overall = (total_missing / total_cells) * 100

    key_roles = ["date", "product", "region", "revenue", "quantity", "order_id", "stock", "reorder_level"]
    role_confidences = [
        _get_role_confidence(schema_suggestions, role, 0.60)
        for role in key_roles
        if _pick_column(schema_suggestions, role) is not None
    ]
    avg_schema_conf = sum(role_confidences) / len(role_confidences) if role_confidences else 0.60

    duplicate_penalty = min(0.20, duplicate_rows / max(row_count, 1))
    missing_penalty = min(0.45, missing_percent_overall / 100.0)

    quality_score = 1.0
    quality_score -= missing_penalty
    quality_score -= duplicate_penalty
    quality_score = quality_score * 0.65 + avg_schema_conf * 0.35
    quality_score = max(0.0, min(1.0, quality_score))

    if quality_score >= 0.80:
        quality_label = "high"
    elif quality_score >= 0.60:
        quality_label = "medium"
    else:
        quality_label = "low"

    warnings: List[str] = []
    if missing_percent_overall > 15:
        warnings.append("Dataset has notable missing values.")
    if duplicate_rows > 0:
        warnings.append("Dataset contains duplicate rows.")
    if avg_schema_conf < 0.65:
        warnings.append("Schema confidence is weak for some important roles.")

    return {
        "row_count": row_count,
        "column_count": column_count,
        "duplicate_rows": duplicate_rows,
        "missing_percent_overall": round(missing_percent_overall, 2),
        "quality_score": round(quality_score, 4),
        "quality_label": quality_label,
        "avg_schema_confidence": round(avg_schema_conf, 4),
        "warnings": warnings,
    }


def _extract_forecast_values(forecast_result: Dict[str, Any]) -> List[float]:
    values: List[float] = []

    forecast_values = forecast_result.get("forecast_values", [])
    for point in forecast_values:
        if isinstance(point, dict):
            value = point.get("predicted_value")
            if value is not None:
                try:
                    values.append(float(value))
                except Exception:
                    pass
        else:
            try:
                values.append(float(point))
            except Exception:
                pass

    forecast_series = forecast_result.get("forecast_series", [])
    for point in forecast_series:
        if isinstance(point, dict):
            value = point.get("yhat")
            if value is not None:
                try:
                    values.append(float(value))
                except Exception:
                    pass

    return values


def _calculate_growth_percentage(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0

    start = values[0]
    end = values[-1]

    if start == 0:
        return 0.0

    return ((end - start) / start) * 100.0


def _detect_trend(values: List[float], forecast_result: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(forecast_result, dict):
        trend_direction = str(forecast_result.get("trend_direction", "")).strip().lower()
        if trend_direction in {"increasing", "decreasing", "stable"}:
            return trend_direction

    if len(values) < 2:
        return "stable"

    start = values[0]
    end = values[-1]

    if end > start * 1.10:
        return "increasing"
    if end < start * 0.90:
        return "decreasing"
    return "stable"


def _forecast_signal_strength(growth: float) -> float:
    magnitude = abs(growth)
    if magnitude >= 25:
        return 1.0
    if magnitude >= 15:
        return 0.8
    if magnitude >= 8:
        return 0.6
    if magnitude >= 3:
        return 0.4
    return 0.2


def _reliability_to_score(reliability_label: str, fallback_score: Any = 0.5) -> float:
    label = str(reliability_label or "").lower()

    if label in {"high", "reliable"}:
        return 0.85
    if label in {"medium", "moderate"}:
        return 0.65
    if label in {"low", "not reliable"}:
        return 0.35

    try:
        value = float(fallback_score)
        if value > 1:
            value = value / 100.0
        return max(0.0, min(1.0, value))
    except Exception:
        return 0.5


def _detect_forecast_risk(
    trend: str,
    growth: float,
    reliability_label: str,
    usability_flag: str,
) -> str:
    reliability_label = str(reliability_label or "").lower()
    usability_flag = str(usability_flag or "").lower()

    if usability_flag == "not_recommended" or reliability_label in {"low", "not reliable"}:
        return "forecast_uncertain"

    if trend == "increasing" and growth >= 20:
        return "demand_spike_risk"
    if trend == "decreasing" and growth <= -15:
        return "demand_softening_risk"
    if abs(growth) >= 8:
        return "demand_shift_risk"

    return "normal"


def _priority_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _impact_label_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def _build_explanation_text(
    reasoning_why: str,
    forecast_support: Optional[str],
    rationale: Optional[str],
) -> str:
    parts = []

    if reasoning_why:
        parts.append(reasoning_why.strip())

    if rationale and rationale.strip() and rationale.strip() not in reasoning_why:
        parts.append(rationale.strip())

    if forecast_support and forecast_support.strip():
        parts.append(f"Forecast support: {forecast_support.strip()}")

    return " ".join(parts).strip()


def _build_decision(
    *,
    title: str,
    category: str,
    trigger_source: str,
    reasoning_why: str,
    what_if_ignored: str,
    recommendation: str,
    evidence: Dict[str, Any],
    confidence_score: float,
    impact_score: float,
    risk_level: str,
    rationale: Optional[str] = None,
    forecast_support: Optional[str] = None,
) -> Dict[str, Any]:
    confidence_score = max(0.0, min(1.0, confidence_score))
    impact_score = max(0.0, min(1.0, impact_score))

    explanation = _build_explanation_text(
        reasoning_why=reasoning_why,
        forecast_support=forecast_support,
        rationale=rationale,
    )

    return {
        "title": title,
        "category": category,
        "decision_type": category,
        "trigger_source": trigger_source,
        "priority": _priority_from_score((confidence_score * 0.45) + (impact_score * 0.55)),
        "impact_estimate": {
            "label": _impact_label_from_score(impact_score),
            "score": round(impact_score, 3),
        },
        "confidence": {
            "label": _confidence_label(confidence_score),
            "score": round(confidence_score, 3),
        },
        "risk_level": risk_level,
        "recommendation": recommendation,
        "rationale": rationale or reasoning_why,
        "explanation": explanation,
        "forecast_support": forecast_support,
        "reasoning": {
            "why": reasoning_why,
            "what_happens_if_ignored": what_if_ignored,
        },
        "evidence": evidence,
        "supporting_evidence": evidence,
    }


def _build_forecast_support_text(
    trend: str,
    growth: float,
    reliability_label: str,
    usability_flag: str,
) -> str:
    label = str(reliability_label or "").strip()
    usability = str(usability_flag or "").strip().replace("_", " ")

    if trend == "increasing":
        return (
            f"Available forecast signals point to an increasing pattern over the forecast horizon "
            f"with estimated movement of {round(growth, 2)}%. Reliability is marked as {label or 'unknown'} "
            f"and usability is {usability or 'unknown'}."
        )
    if trend == "decreasing":
        return (
            f"Available forecast signals point to a decreasing pattern over the forecast horizon "
            f"with estimated movement of {round(growth, 2)}%. Reliability is marked as {label or 'unknown'} "
            f"and usability is {usability or 'unknown'}."
        )
    return (
        f"Available forecast signals suggest a relatively stable near-term pattern. "
        f"Reliability is marked as {label or 'unknown'} and usability is {usability or 'unknown'}."
    )


def _resolve_primary_value_column(
    df: pd.DataFrame,
    schema_suggestions: Optional[Dict[str, Any]]
) -> Tuple[Optional[str], str]:
    revenue_col = _pick_column(schema_suggestions, "revenue", None)
    quantity_col = _pick_column(schema_suggestions, "quantity", None)

    if revenue_col and revenue_col in df.columns:
        return revenue_col, "revenue"

    if quantity_col and quantity_col in df.columns:
        return quantity_col, "quantity"

    # sensible fallbacks
    for col in ["sales_qty", "quantity", "qty", "units", "demand_qty", "stock"]:
        if col in df.columns:
            return col, "quantity" if col != "stock" else "stock"

    return None, "unknown"


def _build_forecast_decisions(
    forecast_result: Optional[Dict[str, Any]],
    data_quality: Dict[str, Any],
    capability_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []

    if not isinstance(forecast_result, dict) or not forecast_result:
        return decisions

    forecast_mode = str(forecast_result.get("forecast_mode", "")).lower()
    reliability_label = str(forecast_result.get("reliability_label", "Not Reliable"))
    reliability_score = _reliability_to_score(
        reliability_label,
        forecast_result.get("reliability_score", 0.5),
    )
    usability_flag = str(forecast_result.get("decision_usability_flag", "usable_with_caution")).lower()
    model_used = forecast_result.get("model_used") or forecast_result.get("model_type")
    validation_metrics = forecast_result.get("validation_metrics", {})
    warnings = forecast_result.get("warnings", [])
    candidate_diagnostics = forecast_result.get("candidate_diagnostics", [])

    values = _extract_forecast_values(forecast_result)
    trend = _detect_trend(values, forecast_result)
    growth = _calculate_growth_percentage(values) if values else float(forecast_result.get("growth_percent", 0.0) or 0.0)
    risk = _detect_forecast_risk(trend, growth, reliability_label, usability_flag)
    signal_strength = _forecast_signal_strength(growth)

    sales_cap = _extract_capability_info(capability_result, "sales")
    inventory_cap = _extract_capability_info(capability_result, "inventory")

    quality_score = float(data_quality.get("quality_score", 0.6))

    base_confidence = (
        reliability_score * 0.45
        + quality_score * 0.25
        + signal_strength * 0.15
        + sales_cap["confidence"] * 0.15
    )
    base_confidence = max(0.0, min(1.0, base_confidence))

    forecast_support_text = _build_forecast_support_text(
        trend=trend,
        growth=growth,
        reliability_label=reliability_label,
        usability_flag=usability_flag,
    )

    common_evidence = {
        "forecast_reference": {
            "forecast_mode": forecast_mode,
            "model_used": model_used,
            "trend": trend,
            "growth_percent": round(growth, 2),
            "reliability_label": reliability_label,
            "reliability_score": round(reliability_score, 3),
            "decision_usability_flag": usability_flag,
            "validation_metrics": validation_metrics,
            "warnings": warnings,
            "candidate_diagnostics": candidate_diagnostics,
        },
        "data_reference": {
            "dataset_quality_label": data_quality.get("quality_label"),
            "dataset_quality_score": data_quality.get("quality_score"),
            "missing_percent_overall": data_quality.get("missing_percent_overall"),
            "duplicate_rows": data_quality.get("duplicate_rows"),
        },
        "capability_reference": {
            "sales": sales_cap,
            "inventory": inventory_cap,
        },
    }

    if forecast_mode == "refuse_forecast":
        decisions.append(
            _build_decision(
                title="Do not use forecast for operational commitments yet",
                category="sales",
                trigger_source="forecast",
                reasoning_why="The forecasting layer did not produce a planning-grade forecast, so the current forward signal is not strong enough for direct operational commitment.",
                what_if_ignored="Operational decisions may be made on unstable signal, which can create avoidable planning, stock, or revenue risk.",
                recommendation="Improve forecast readiness before using forecast-driven action for operational commitments.",
                evidence=common_evidence,
                confidence_score=max(0.40, quality_score * 0.55),
                impact_score=0.85,
                risk_level="forecast_uncertain",
                rationale="Forecast usability is too weak for confident execution.",
                forecast_support="A reliable operational forecast is not available at this stage.",
            )
        )
        return decisions

    if forecast_mode == "trend_projection":
        decisions.append(
            _build_decision(
                title=f"Use forecast only for directional review ({trend})",
                category="sales",
                trigger_source="forecast",
                reasoning_why="A directional pattern is visible, but the signal is not strong enough for high-confidence operational action.",
                what_if_ignored="The business may overreact to a weak forecast signal and make commitments not fully supported by validation strength.",
                recommendation="Use the forecast as directional guidance only and validate it with business judgment before major action.",
                evidence=common_evidence,
                confidence_score=base_confidence * 0.80,
                impact_score=0.55,
                risk_level=risk,
                rationale="This forecast is more suitable for management review than direct execution.",
                forecast_support=forecast_support_text,
            )
        )
        return decisions

    if forecast_mode in {"predictive_forecast", "", "monthly_forecast"}:
        if trend == "increasing":
            inventory_boost = 0.10 if inventory_cap["status"] in {"enabled", "partial"} else 0.0
            decisions.append(
                _build_decision(
                    title="Prepare for short-term demand increase",
                    category="inventory",
                    trigger_source="forecast",
                    reasoning_why="Forward-looking estimates suggest near-term upward pressure, which may require tighter stock and supply planning.",
                    what_if_ignored="The business may face stock stress, missed sales, or service gaps if supply and replenishment do not adapt in time.",
                    recommendation="Review stock levels, replenishment timing, and near-term supply capacity for likely high-demand items.",
                    evidence=common_evidence,
                    confidence_score=min(1.0, base_confidence + inventory_boost),
                    impact_score=min(1.0, 0.50 + signal_strength * 0.45),
                    risk_level=risk,
                    rationale="Planning should adapt where forecast signals indicate stronger short-term activity.",
                    forecast_support=forecast_support_text,
                )
            )

        elif trend == "decreasing":
            decisions.append(
                _build_decision(
                    title="Protect against short-term demand softening",
                    category="sales",
                    trigger_source="forecast",
                    reasoning_why="Forward-looking estimates indicate softer near-term movement, so aggressive allocation may create avoidable exposure.",
                    what_if_ignored="The business may over-allocate stock, budget, or commercial effort into weaker demand conditions.",
                    recommendation="Review demand assumptions and tighten short-term allocation until the pattern becomes clearer.",
                    evidence=common_evidence,
                    confidence_score=base_confidence,
                    impact_score=min(1.0, 0.50 + signal_strength * 0.40),
                    risk_level=risk,
                    rationale="This helps prevent over-commitment when projected demand signals weaken.",
                    forecast_support=forecast_support_text,
                )
            )

        else:
            decisions.append(
                _build_decision(
                    title="Maintain stable planning posture and monitor movement",
                    category="sales",
                    trigger_source="forecast",
                    reasoning_why="The available forecast suggests no strong directional movement, so a balanced planning posture is more appropriate than aggressive change.",
                    what_if_ignored="The team may either overreact to normal variation or miss subtle changes that still require monitoring.",
                    recommendation="Maintain the current plan, but continue monitoring the forecast and key business metrics for shifts.",
                    evidence=common_evidence,
                    confidence_score=base_confidence * 0.95,
                    impact_score=0.35,
                    risk_level=risk,
                    rationale="A stable signal supports monitoring rather than immediate major intervention.",
                    forecast_support=forecast_support_text,
                )
            )

        if reliability_label.lower() in {"low", "not reliable"} or usability_flag in {"usable_with_caution", "not_recommended"}:
            decisions.append(
                _build_decision(
                    title="Apply forecast with caution",
                    category="sales",
                    trigger_source="forecast",
                    reasoning_why="A forecast is available, but reliability or usability indicators still require controlled usage.",
                    what_if_ignored="Large decisions may be made with more confidence than the current data and validation support.",
                    recommendation="Use the forecast for guided planning, not blind execution. Validate it against business context before major action.",
                    evidence=common_evidence,
                    confidence_score=min(0.75, base_confidence * 0.85),
                    impact_score=0.50,
                    risk_level="forecast_uncertain",
                    rationale="Forecast signal exists, but reliability conditions still limit how strongly it should be used.",
                    forecast_support=forecast_support_text,
                )
            )

    return decisions


def _build_product_decisions(
    df: pd.DataFrame,
    schema_suggestions: Optional[Dict[str, Any]],
    data_quality: Dict[str, Any],
    capability_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []

    product_col = _pick_column(schema_suggestions, "product", "Product")
    if not product_col or product_col not in df.columns:
        return decisions

    value_col, value_type = _resolve_primary_value_column(df, schema_suggestions)
    if not value_col or value_col not in df.columns:
        return decisions

    temp = _safe_numeric(df, [value_col])
    temp = temp.dropna(subset=[product_col, value_col]).copy()

    if temp.empty:
        return decisions

    sales_cap = _extract_capability_info(capability_result, "sales")
    quality_score = float(data_quality.get("quality_score", 0.6))
    product_conf = _get_role_confidence(schema_suggestions, "product", 0.65)
    value_conf = _get_role_confidence(schema_suggestions, "revenue" if value_type == "revenue" else "quantity", 0.75)

    product_summary = (
        temp.groupby(product_col, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    if product_summary.empty:
        return decisions

    top_products = product_summary.head(5)
    total_value = float(product_summary[value_col].sum())
    top_5_value = float(top_products[value_col].sum())
    top_5_share = (top_5_value / total_value * 100.0) if total_value else 0.0

    decision_confidence = (
        quality_score * 0.35
        + product_conf * 0.20
        + value_conf * 0.25
        + sales_cap["confidence"] * 0.20
    )
    decision_confidence = max(0.0, min(1.0, decision_confidence))

    evidence = {
        "insight_reference": {
            "metric_type": value_type,
            "metric_column": value_col,
            "top_products": top_products[product_col].astype(str).tolist(),
            "top_5_total_value": round(top_5_value, 2),
            "top_5_share_percent": round(top_5_share, 2),
            "product_count": int(product_summary.shape[0]),
        },
        "data_reference": {
            "product_column": product_col,
            "value_column": value_col,
            "dataset_quality_label": data_quality.get("quality_label"),
            "dataset_quality_score": data_quality.get("quality_score"),
        },
        "schema_reference": {
            "product_role_confidence": round(product_conf, 3),
            "value_role_confidence": round(value_conf, 3),
        },
        "capability_reference": {
            "sales": sales_cap,
        },
    }

    decisions.append(
        _build_decision(
            title="Focus attention on top-performing products",
            category="sales",
            trigger_source="insights",
            reasoning_why=f"A small group of products is contributing a meaningful share of {value_type} and deserves focused commercial and operational attention.",
            what_if_ignored="The business may under-support the products that matter most and miss optimization opportunities in stock, promotion, or planning.",
            recommendation="Prioritize inventory, promotion, and planning around the strongest contributing products.",
            evidence=evidence,
            confidence_score=decision_confidence,
            impact_score=min(1.0, 0.40 + (top_5_share / 100.0) * 0.50),
            risk_level="normal",
            rationale=f"Top-performing products are currently a key {value_type} concentration area.",
            forecast_support=None,
        )
    )

    if top_5_share >= 70:
        decisions.append(
            _build_decision(
                title="Reduce concentration risk across products",
                category="sales",
                trigger_source="insights",
                reasoning_why=f"The top 5 products contribute {round(top_5_share, 2)}% of total {value_type}, indicating a concentrated business base.",
                what_if_ignored="A change in demand for a small number of products could disproportionately affect overall business performance.",
                recommendation="Review concentration risk and explore diversification or protection strategies across the product mix.",
                evidence=evidence,
                confidence_score=decision_confidence,
                impact_score=min(1.0, 0.55 + (top_5_share / 100.0) * 0.35),
                risk_level="concentration_risk",
                rationale="High dependence on a limited product set raises resilience concerns.",
                forecast_support=None,
            )
        )

    return decisions


def _build_region_decisions(
    df: pd.DataFrame,
    schema_suggestions: Optional[Dict[str, Any]],
    data_quality: Dict[str, Any],
    capability_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []

    region_col = _pick_column(schema_suggestions, "region", "Region")
    if not region_col or region_col not in df.columns:
        return decisions

    value_col, value_type = _resolve_primary_value_column(df, schema_suggestions)
    if not value_col or value_col not in df.columns:
        return decisions

    temp = _safe_numeric(df, [value_col])
    temp = temp.dropna(subset=[region_col, value_col]).copy()

    if temp.empty:
        return decisions

    sales_cap = _extract_capability_info(capability_result, "sales")
    quality_score = float(data_quality.get("quality_score", 0.6))
    region_conf = _get_role_confidence(schema_suggestions, "region", 0.65)
    value_conf = _get_role_confidence(schema_suggestions, "revenue" if value_type == "revenue" else "quantity", 0.75)

    region_summary = (
        temp.groupby(region_col, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    if region_summary.shape[0] < 2:
        return decisions

    top_region = str(region_summary.iloc[0][region_col])
    bottom_region = str(region_summary.iloc[-1][region_col])
    top_value = float(region_summary.iloc[0][value_col])
    bottom_value = float(region_summary.iloc[-1][value_col])

    if top_value <= 0:
        return decisions

    gap_ratio = (top_value - bottom_value) / top_value

    confidence_score = (
        quality_score * 0.35
        + region_conf * 0.25
        + value_conf * 0.20
        + sales_cap["confidence"] * 0.20
    )
    confidence_score = max(0.0, min(1.0, confidence_score))

    evidence = {
        "insight_reference": {
            "metric_type": value_type,
            "top_region": top_region,
            "top_region_value": round(top_value, 2),
            "bottom_region": bottom_region,
            "bottom_region_value": round(bottom_value, 2),
            "regional_gap_ratio": round(gap_ratio, 3),
        },
        "data_reference": {
            "region_column": region_col,
            "value_column": value_col,
            "dataset_quality_label": data_quality.get("quality_label"),
            "dataset_quality_score": data_quality.get("quality_score"),
        },
        "schema_reference": {
            "region_role_confidence": round(region_conf, 3),
            "value_role_confidence": round(value_conf, 3),
        },
        "capability_reference": {
            "sales": sales_cap,
        },
    }

    if gap_ratio >= 0.40:
        decisions.append(
            _build_decision(
                title="Review weak-performing regions",
                category="sales",
                trigger_source="insights",
                reasoning_why="Regional performance appears uneven, with a notable gap between the strongest and weakest regions.",
                what_if_ignored="The business may continue underperforming in weaker regions without identifying root causes or targeted corrective action.",
                recommendation="Investigate weaker regions and compare demand, supply, pricing, or channel conditions against stronger regions.",
                evidence=evidence,
                confidence_score=confidence_score,
                impact_score=min(1.0, 0.45 + gap_ratio * 0.45),
                risk_level="regional_performance_risk",
                rationale="This helps management identify where targeted intervention may improve regional balance.",
                forecast_support=None,
            )
        )

    return decisions


def _build_inventory_decisions(
    df: pd.DataFrame,
    schema_suggestions: Optional[Dict[str, Any]],
    data_quality: Dict[str, Any],
    capability_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []

    product_col = _pick_column(schema_suggestions, "product", None)
    stock_col = _pick_column(schema_suggestions, "stock", None)
    reorder_col = _pick_column(schema_suggestions, "reorder_level", None)

    if not product_col or not stock_col or not reorder_col:
        return decisions

    if product_col not in df.columns or stock_col not in df.columns or reorder_col not in df.columns:
        return decisions

    temp = _safe_numeric(df, [stock_col, reorder_col])
    temp = temp.dropna(subset=[product_col, stock_col, reorder_col]).copy()

    if temp.empty:
        return decisions

    temp["stock_gap"] = temp[stock_col] - temp[reorder_col]
    low_stock_df = temp[temp["stock_gap"] <= 0].copy()

    if low_stock_df.empty:
        return decisions

    inventory_cap = _extract_capability_info(capability_result, "inventory")
    quality_score = float(data_quality.get("quality_score", 0.6))
    stock_conf = _get_role_confidence(schema_suggestions, "stock", 0.75)
    reorder_conf = _get_role_confidence(schema_suggestions, "reorder_level", 0.75)

    low_stock_count = int(low_stock_df.shape[0])
    top_product = str(low_stock_df.iloc[0][product_col])

    confidence_score = (
        quality_score * 0.35
        + inventory_cap["confidence"] * 0.30
        + stock_conf * 0.20
        + reorder_conf * 0.15
    )
    confidence_score = max(0.0, min(1.0, confidence_score))

    evidence = {
        "insight_reference": {
            "low_stock_count": low_stock_count,
            "example_product": top_product,
        },
        "data_reference": {
            "product_column": product_col,
            "stock_column": stock_col,
            "reorder_level_column": reorder_col,
            "dataset_quality_label": data_quality.get("quality_label"),
            "dataset_quality_score": data_quality.get("quality_score"),
        },
        "schema_reference": {
            "stock_role_confidence": round(stock_conf, 3),
            "reorder_level_role_confidence": round(reorder_conf, 3),
        },
        "capability_reference": {
            "inventory": inventory_cap,
        },
    }

    decisions.append(
        _build_decision(
            title="Replenish items at or below reorder level",
            category="inventory",
            trigger_source="insights",
            reasoning_why="Current stock levels show one or more items at or below reorder threshold, indicating replenishment pressure.",
            what_if_ignored="The business may face stock-out risk, missed sales, or service disruption if replenishment is delayed.",
            recommendation="Prioritize replenishment review for products with stock at or below reorder level.",
            evidence=evidence,
            confidence_score=confidence_score,
            impact_score=min(1.0, 0.45 + min(low_stock_count / 20.0, 0.45)),
            risk_level="stock_risk",
            rationale="Low stock pressure is visible in the current analysis.",
            forecast_support=None,
        )
    )

    return decisions


def _build_supplier_delay_decisions(
    df: pd.DataFrame,
    schema_suggestions: Optional[Dict[str, Any]],
    data_quality: Dict[str, Any],
    capability_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []

    shipment_col = _pick_column(schema_suggestions, "shipment_date", None)
    delivery_col = _pick_column(schema_suggestions, "delivery_date", None)

    supplier_col = None
    for candidate in ["supplier_name", "supplier", "vendor_name", "vendor", "Supplier"]:
        if candidate in df.columns:
            supplier_col = candidate
            break

    if not shipment_col or not delivery_col or not supplier_col:
        return decisions

    if shipment_col not in df.columns or delivery_col not in df.columns or supplier_col not in df.columns:
        return decisions

    temp = df.copy()
    temp[shipment_col] = pd.to_datetime(temp[shipment_col], errors="coerce")
    temp[delivery_col] = pd.to_datetime(temp[delivery_col], errors="coerce")
    temp = temp.dropna(subset=[supplier_col, shipment_col, delivery_col])

    if temp.empty:
        return decisions

    temp["delay_days"] = (temp[delivery_col] - temp[shipment_col]).dt.days
    temp = temp[temp["delay_days"] >= 0]

    if temp.empty:
        return decisions

    supplier_delay = (
        temp.groupby(supplier_col)["delay_days"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    if supplier_delay.empty:
        return decisions

    worst_supplier = str(supplier_delay.iloc[0][supplier_col])
    worst_delay = float(supplier_delay.iloc[0]["delay_days"])

    logistics_cap = _extract_capability_info(capability_result, "logistics")
    quality_score = float(data_quality.get("quality_score", 0.6))

    confidence_score = (
        quality_score * 0.35
        + logistics_cap["confidence"] * 0.35
        + 0.30
    )
    confidence_score = max(0.0, min(1.0, confidence_score))

    evidence = {
        "insight_reference": {
            "highest_delay_supplier": worst_supplier,
            "highest_average_delay_days": round(worst_delay, 2),
            "supplier_delay_ranking": supplier_delay.to_dict(orient="records"),
        },
        "data_reference": {
            "supplier_column": supplier_col,
            "shipment_column": shipment_col,
            "delivery_column": delivery_col,
            "dataset_quality_label": data_quality.get("quality_label"),
            "dataset_quality_score": data_quality.get("quality_score"),
        },
        "capability_reference": {
            "logistics": logistics_cap,
        },
    }

    if worst_delay >= 5:
        decisions.append(
            _build_decision(
                title=f"Review supplier delay risk for {worst_supplier}",
                category="logistics",
                trigger_source="insights",
                reasoning_why=f"{worst_supplier} is showing the highest average delivery delay in the current data.",
                what_if_ignored="Persistent supplier delays can reduce service levels and increase stock-out or fulfillment risk.",
                recommendation="Review supplier performance and build mitigation plans for delayed supplier flow.",
                evidence=evidence,
                confidence_score=confidence_score,
                impact_score=min(1.0, 0.45 + min(worst_delay / 10.0, 0.45)),
                risk_level="supplier_delay_risk",
                rationale="Supplier delay risk is materially visible in the available logistics data.",
                forecast_support=None,
            )
        )

    return decisions


def _priority_rank(priority: str) -> int:
    mapping = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }
    return mapping.get(str(priority or "low").lower(), 1)


def _deduplicate_decisions(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []

    for item in decisions:
        key = (
            str(item.get("title", "")).strip().lower(),
            str(item.get("category", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)

    return output


def _decision_sort_key(item: Dict[str, Any]) -> Tuple[int, float, float]:
    priority = _priority_rank(item.get("priority", "low"))
    impact_score = float(item.get("impact_estimate", {}).get("score", 0.0))
    confidence_score = float(item.get("confidence", {}).get("score", 0.0))
    return (priority, impact_score, confidence_score)


def _build_management_summary(decisions: List[Dict[str, Any]], data_quality: Dict[str, Any]) -> str:
    if not decisions:
        return "No strong evidence-based decisions could be generated from the available data."

    high_count = sum(1 for d in decisions if str(d.get("priority", "")).lower() == "high")
    medium_count = sum(1 for d in decisions if str(d.get("priority", "")).lower() == "medium")
    quality_label = data_quality.get("quality_label", "unknown")

    if high_count > 0:
        return (
            f"Generated {len(decisions)} structured decision(s), including {high_count} high-priority item(s). "
            f"These recommendations are supported by available business signals, forecast evidence where available, "
            f"and data quality adjusted confidence scoring. Current dataset quality is {quality_label}."
        )

    return (
        f"Generated {len(decisions)} structured decision(s), with {medium_count} medium-priority item(s) and "
        f"data-quality-adjusted confidence scoring. Current dataset quality is {quality_label}."
    )


def generate_decisions(
    df: pd.DataFrame,
    schema_suggestions: Optional[Dict[str, Any]],
    forecast_result: Optional[Dict[str, Any]] = None,
    capability_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        if df is None or df.empty:
            return {
                "top_decisions": [],
                "summary": "No data available for decision generation.",
                "decision_quality": {
                    "label": "low",
                    "score": 0.0,
                },
                "error": "Empty dataset",
            }

        schema_suggestions = schema_suggestions or {}
        data_quality = _dataset_quality_summary(df, schema_suggestions)

        decisions: List[Dict[str, Any]] = []

        if forecast_result:
            decisions.extend(
                _build_forecast_decisions(
                    forecast_result=forecast_result,
                    data_quality=data_quality,
                    capability_result=capability_result,
                )
            )

        decisions.extend(
            _build_product_decisions(
                df=df,
                schema_suggestions=schema_suggestions,
                data_quality=data_quality,
                capability_result=capability_result,
            )
        )

        decisions.extend(
            _build_region_decisions(
                df=df,
                schema_suggestions=schema_suggestions,
                data_quality=data_quality,
                capability_result=capability_result,
            )
        )

        decisions.extend(
            _build_inventory_decisions(
                df=df,
                schema_suggestions=schema_suggestions,
                data_quality=data_quality,
                capability_result=capability_result,
            )
        )

        decisions.extend(
            _build_supplier_delay_decisions(
                df=df,
                schema_suggestions=schema_suggestions,
                data_quality=data_quality,
                capability_result=capability_result,
            )
        )

        decisions = _deduplicate_decisions(decisions)
        decisions = sorted(decisions, key=_decision_sort_key, reverse=True)

        if not decisions:
            return {
                "top_decisions": [],
                "summary": "No strong evidence-based decisions could be generated from the available data.",
                "decision_quality": {
                    "label": data_quality.get("quality_label", "low"),
                    "score": data_quality.get("quality_score", 0.0),
                },
                "data_quality": data_quality,
                "error": None,
            }

        avg_conf = sum(
            float(d.get("confidence", {}).get("score", 0.0))
            for d in decisions
        ) / max(len(decisions), 1)

        avg_impact = sum(
            float(d.get("impact_estimate", {}).get("score", 0.0))
            for d in decisions
        ) / max(len(decisions), 1)

        decision_quality_score = (avg_conf * 0.60) + (data_quality.get("quality_score", 0.0) * 0.40)

        if decision_quality_score >= 0.80:
            decision_quality_label = "high"
        elif decision_quality_score >= 0.60:
            decision_quality_label = "medium"
        else:
            decision_quality_label = "low"

        return {
            "top_decisions": decisions[:10],
            "summary": _build_management_summary(decisions, data_quality),
            "decision_quality": {
                "label": decision_quality_label,
                "score": round(decision_quality_score, 3),
                "average_confidence": round(avg_conf, 3),
                "average_impact": round(avg_impact, 3),
            },
            "data_quality": data_quality,
            "error": None,
        }

    except Exception as e:
        return {
            "top_decisions": [],
            "summary": "Decision generation failed.",
            "decision_quality": {
                "label": "low",
                "score": 0.0,
            },
            "error": str(e),
        }