from __future__ import annotations

from typing import Dict, Any, List


CAPABILITY_RULES = {
    "sales": {
        "roles": ["date", "product", "revenue", "quantity", "region"],
        "min_required": 3,
        "must_have": ["revenue"],
        "important_roles": ["date", "product", "region"],
        "description": "Sales analytics requires revenue plus enough business dimensions for trend and breakdown analysis.",
    },
    "inventory": {
        "roles": ["product", "stock", "reorder_level", "lead_time", "region"],
        "min_required": 2,
        "must_have": ["stock"],
        "important_roles": ["product", "reorder_level", "lead_time"],
        "description": "Inventory analytics requires stock visibility and ideally reorder or lead-time support.",
    },
    "logistics": {
        "roles": ["order_id", "shipment_date", "delivery_date", "status", "region"],
        "min_required": 3,
        "must_have": ["order_id"],
        "important_roles": ["shipment_date", "delivery_date", "status"],
        "description": "Logistics analytics requires order-level tracking and event/status coverage.",
    },
}


def _extract_suggestions(schema_suggestions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports multiple input formats:

    1. {
         "date": {"column": "...", "confidence": 1.0},
         ...
       }

    2. {
         "suggestions": {
             "date": {"column": "...", "confidence": 1.0},
             ...
         }
       }
    """
    if not isinstance(schema_suggestions, dict):
        return {}

    if "suggestions" in schema_suggestions and isinstance(schema_suggestions["suggestions"], dict):
        return schema_suggestions["suggestions"]

    return schema_suggestions


def _normalize_role_payload(role_data: Any) -> Dict[str, Any]:
    """
    Normalize different role payload shapes into:
    {
        "column": Optional[str],
        "confidence": float
    }
    """
    if not isinstance(role_data, dict):
        return {"column": None, "confidence": 0.0}

    column = role_data.get("column")
    if column is None:
        column = role_data.get("column_name")

    if "confidence" in role_data:
        confidence = role_data.get("confidence", 0.0)
        if confidence is None:
            confidence = 1.0 if column is not None else 0.0
    else:
        confidence = 1.0 if column is not None else 0.0

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(confidence, 1.0))

    return {
        "column": column,
        "confidence": confidence,
    }


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    if score >= 0.4:
        return "low"
    return "very_low"


def _build_enable_reason(
    status: str,
    matched_roles: List[str],
    min_required: int,
    must_have: List[str],
    must_have_ok: bool,
) -> str:
    if status == "enabled":
        reason_parts = ["full capability available"]
        if must_have:
            reason_parts.append("required core roles present")
        return "; ".join(reason_parts)

    if status == "partial":
        reason_parts = ["partial capability available"]
        if must_have_ok:
            reason_parts.append("core required roles present")
        reason_parts.append("limited output should still be allowed")
        return "; ".join(reason_parts)

    if status == "limited":
        return (
            "very limited capability available; fallback mode enabled from available columns"
        )

    reason_parts = ["insufficient data for this capability"]

    if must_have and not must_have_ok:
        missing_must_have = [r for r in must_have if r not in matched_roles]
        if missing_must_have:
            reason_parts.append(
                f"missing required roles: {', '.join(missing_must_have)}"
            )

    if len(matched_roles) < min_required:
        reason_parts.append(
            f"matched only {len(matched_roles)} role(s), below minimum required {min_required}"
        )

    return "; ".join(reason_parts)


def detect_capabilities(
    schema_suggestions: Dict[str, Any],
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    suggestions = _extract_suggestions(schema_suggestions)
    detected: Dict[str, Any] = {}

    # Always keep insights available if dataset exists and schema mapping has started.
    detected["insights"] = {
        "enabled": True,
        "status": "enabled",
        "coverage": 1.0,
        "confidence_score": 1.0,
        "confidence_label": "high",
        "matched_roles": [],
        "weak_roles": [],
        "missing_roles": [],
        "must_have_roles": [],
        "must_have_satisfied": True,
        "role_details": {},
        "description": "Insights can be generated from available dataset even with partial schema coverage.",
        "enable_reason": "Insights are always enabled as long as dataset is available.",
    }

    for capability, config in CAPABILITY_RULES.items():
        roles: List[str] = config.get("roles", [])
        min_required: int = config.get("min_required", 0)
        must_have: List[str] = config.get("must_have", [])
        important_roles: List[str] = config.get("important_roles", [])

        matched_roles: List[str] = []
        missing_roles: List[str] = []
        weak_roles: List[str] = []
        role_details: Dict[str, Any] = {}

        weighted_score = 0.0
        weighted_total = 0.0

        for role in roles:
            raw_role_data = suggestions.get(role, {})
            normalized_role_data = _normalize_role_payload(raw_role_data)

            column = normalized_role_data["column"]
            confidence = normalized_role_data["confidence"]

            role_weight = 1.0
            if role in must_have:
                role_weight = 1.5
            elif role in important_roles:
                role_weight = 1.2

            weighted_total += role_weight

            meets_threshold = bool(column is not None and confidence >= confidence_threshold)

            role_payload = {
                "column": column,
                "confidence": round(confidence, 2),
                "meets_threshold": meets_threshold,
                "confidence_label": _confidence_label(confidence),
            }
            role_details[role] = role_payload

            if column is not None:
                weighted_score += role_weight * confidence

            if meets_threshold:
                matched_roles.append(role)
            elif column is not None and confidence > 0:
                weak_roles.append(role)
            else:
                missing_roles.append(role)

        must_have_ok = all(role in matched_roles for role in must_have)
        coverage = len(matched_roles) / len(roles) if roles else 0.0
        confidence_score = weighted_score / weighted_total if weighted_total > 0 else 0.0

        # Flexible fallback logic:
        # 1. enabled  -> must-have roles satisfied and minimum role count satisfied
        # 2. partial  -> must-have satisfied but minimum role count not fully reached
        # 3. limited  -> near-threshold fallback so system can still run reduced logic
        # 4. not_enabled -> too little usable structure
        if must_have_ok and len(matched_roles) >= min_required:
            enabled = True
            status = "enabled"
        elif must_have_ok:
            enabled = True
            status = "partial"
        elif len(matched_roles) >= max(1, min_required - 1):
            enabled = True
            status = "limited"
        else:
            enabled = False
            status = "not_enabled"

        detected[capability] = {
            "enabled": enabled,
            "status": status,
            "coverage": round(coverage, 2),
            "confidence_score": round(confidence_score, 2),
            "confidence_label": _confidence_label(confidence_score),
            "matched_roles": matched_roles,
            "weak_roles": weak_roles,
            "missing_roles": missing_roles,
            "must_have_roles": must_have,
            "must_have_satisfied": must_have_ok,
            "role_details": role_details,
            "description": config.get("description", ""),
            "enable_reason": _build_enable_reason(
                status=status,
                matched_roles=matched_roles,
                min_required=min_required,
                must_have=must_have,
                must_have_ok=must_have_ok,
            ),
        }

    return detected