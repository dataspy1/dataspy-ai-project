from typing import Dict, List, Optional, Tuple
import re
import pandas as pd

from app.utils.data_cleaning import clean_dataframe, parse_dates


ROLE_SYNONYMS = {
    "date": [
        "date",
        "order_date",
        "po_date",
        "po date",
        "purchase_order_date",
        "purchase order date",
        "posting_date",
        "doc_date",
        "transaction_date",
        "document_date",
        "booking_date",
    ],
    "product": [
        "product",
        "item",
        "sku",
        "material",
        "material_description",
        "product_name",
        "description",
    ],
    "region": [
        "region",
        "city",
        "state",
        "country",
        "zone",
        "plant",
        "location",
        "branch",
    ],
    "revenue": [
        "revenue",
        "sales",
        "amount",
        "net_value",
        "net_total",
        "total_amount",
        "total amount",
        "value",
        "invoice_value",
        "total_price",
        "total price",
        "price",
        "sales_value",
        "order_value",
    ],
    "quantity": [
        "quantity",
        "quamtity",
        "qty",
        "units",
        "volume",
        "sold_qty",
        "sold quantity",
    ],
    "order_id": [
        "order_id",
        "order",
        "invoice_id",
        "sales_order",
        "document_no",
        "doc_no",
        "po_number",
        "po_no",
        "purchase_order_no",
    ],
    "delivery_date": [
        "delivery_date",
        "delivery date",
        "delivered_on",
        "delivery_dt",
        "invoice_date",
        "invoice date",
        "billing_date",
        "bill_date",
    ],
    "shipment_date": [
        "shipment_date",
        "shipment date",
        "ship_date",
        "ship date",
        "dispatch_date",
        "dispatch date",
    ],
    "status": [
        "status",
        "delivery_status",
        "shipment_status",
        "state_code",
        "order_status",
    ],
    "stock": [
        "stock",
        "inventory",
        "available_qty",
        "available_stock",
        "on_hand",
    ],
    "reorder_level": [
        "reorder_level",
        "min_stock",
        "safety_stock",
        "threshold",
    ],
    "lead_time": [
        "lead_time",
        "delivery_lead_time",
        "procurement_days",
        "lead_days",
    ],
}


class SchemaMapper:
    @staticmethod
    def _normalize(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    @staticmethod
    def _dtype_hints(series: pd.Series) -> List[str]:
        hints = []

        if pd.api.types.is_datetime64_any_dtype(series):
            hints.append("datetime_detected")
            return hints

        if pd.api.types.is_numeric_dtype(series):
            hints.append("numeric_detected")
            return hints

        sample = series.dropna().astype(str).head(20)

        if len(sample) == 0:
            return hints

        try:
            converted = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            valid_ratio = converted.notna().mean()
            if valid_ratio >= 0.6:
                hints.append("date_like_values")
        except Exception:
            pass

        return hints

    @staticmethod
    def _score_column(column_name: str, series: pd.Series) -> Tuple[str, float, List[str]]:
        normalized = SchemaMapper._normalize(column_name)
        best_role = "unknown"
        best_score = 0.1
        reasons: List[str] = []

        dtype_hints = SchemaMapper._dtype_hints(series)

        for role, synonyms in ROLE_SYNONYMS.items():
            role_score = 0.0
            role_reasons = []

            for synonym in synonyms:
                syn = SchemaMapper._normalize(synonym)

                if normalized == syn:
                    role_score += 0.75
                    role_reasons.append(f"exact_name_match:{synonym}")
                elif syn in normalized or normalized in syn:
                    role_score += 0.45
                    role_reasons.append(f"partial_name_match:{synonym}")

            if role in ["revenue", "quantity", "stock", "reorder_level", "lead_time"] and "numeric_detected" in dtype_hints:
                role_score += 0.15
                role_reasons.append("numeric_dtype_support")

            if role in ["date", "delivery_date", "shipment_date"] and (
                "datetime_detected" in dtype_hints or "date_like_values" in dtype_hints
            ):
                role_score += 0.2
                role_reasons.append("date_pattern_support")

            if role == "status":
                sample_text = " ".join(series.dropna().astype(str).head(20).str.lower().tolist())
                status_keywords = [
                    "pending",
                    "delivered",
                    "shipped",
                    "cancelled",
                    "canceled",
                    "processing",
                    "in process",
                    "delayed",
                    "dispatch",
                    "not delivered",
                ]
                if any(keyword in sample_text for keyword in status_keywords):
                    role_score += 0.2
                    role_reasons.append("status_value_pattern_support")

            # Strong business-specific boosts
            if normalized in {"po_date", "po_date_", "purchase_order_date", "po"}:
                if role == "date":
                    role_score += 0.5
                    role_reasons.append("business_rule_po_date_primary_date")
                if role == "delivery_date":
                    role_score -= 0.25

            if normalized in {"invoice_date", "billing_date", "bill_date"}:
                if role == "delivery_date":
                    role_score += 0.45
                    role_reasons.append("business_rule_invoice_date_delivery_date")
                if role == "date":
                    role_score -= 0.25

            if normalized in {"total_price", "total_amount", "net_value", "sales_value"} and role == "revenue":
                role_score += 0.35
                role_reasons.append("business_rule_revenue_match")

            if normalized in {"quantity", "quamtity", "qty", "sold_qty"} and role == "quantity":
                role_score += 0.30
                role_reasons.append("business_rule_quantity_match")

            if role_score > best_score:
                best_role = role
                best_score = min(role_score, 1.0)
                reasons = role_reasons

        return best_role, round(best_score, 2), reasons

    @staticmethod
    def suggest_roles(df: pd.DataFrame) -> List[Dict]:
        df = clean_dataframe(df)
        df = parse_dates(df)

        suggestions = []

        for column in df.columns:
            detected_role, confidence, reasons = SchemaMapper._score_column(column, df[column])
            suggestions.append(
                {
                    "column_name": column,
                    "detected_role": detected_role,
                    "confidence": confidence,
                    "reasons": reasons,
                }
            )

        return suggestions

    @staticmethod
    def role_to_column_map(suggestions: List[Dict]) -> Dict[str, Optional[str]]:
        role_map: Dict[str, Tuple[Optional[str], float]] = {}

        for item in suggestions:
            role = item["detected_role"]
            column = item["column_name"]
            confidence = item["confidence"]

            if role == "unknown":
                continue

            current = role_map.get(role)
            if current is None or confidence > current[1]:
                role_map[role] = (column, confidence)

        final_map = {role: column for role, (column, _) in role_map.items()}

        for role in ROLE_SYNONYMS.keys():
            final_map.setdefault(role, None)

        return final_map


def _normalized_column_lookup(df: pd.DataFrame) -> Dict[str, str]:
    return {SchemaMapper._normalize(col): col for col in df.columns}


def _apply_business_role_overrides(
    df: pd.DataFrame,
    extracted: Dict[str, Dict],
) -> Dict[str, Dict]:
    """
    Final business-safe override layer to make schema stable for client datasets.

    Priority:
    - PO DATE / ORDER DATE => date
    - INVOICE DATE => delivery_date
    - TOTAL PRICE / TOTAL AMOUNT => revenue
    - quantity / quamtity => quantity
    """
    normalized_map = _normalized_column_lookup(df)

    def set_role(role: str, column_name: str, confidence: float, reason: str):
        if column_name in df.columns:
            extracted[role] = {
                "column": column_name,
                "confidence": confidence,
                "reasons": [reason],
            }

    # 1. Primary business date
    for key in [
        "po_date",
        "po date",
        "order_date",
        "order date",
        "purchase_order_date",
        "posting_date",
        "transaction_date",
    ]:
        norm = SchemaMapper._normalize(key)
        if norm in normalized_map:
            set_role("date", normalized_map[norm], 0.99, "business_override_primary_business_date")
            break

    # 2. Delivery-style date
    for key in [
        "invoice_date",
        "invoice date",
        "delivery_date",
        "delivery date",
        "billing_date",
        "bill_date",
    ]:
        norm = SchemaMapper._normalize(key)
        if norm in normalized_map:
            set_role("delivery_date", normalized_map[norm], 0.98, "business_override_delivery_date")
            break

    # 3. Revenue
    for key in [
        "total_price",
        "total price",
        "total_amount",
        "total amount",
        "net_value",
        "sales_value",
        "revenue",
    ]:
        norm = SchemaMapper._normalize(key)
        if norm in normalized_map:
            set_role("revenue", normalized_map[norm], 0.99, "business_override_revenue")
            break

    # 4. Quantity
    for key in [
        "quantity",
        "quamtity",
        "qty",
        "units",
        "sold_qty",
    ]:
        norm = SchemaMapper._normalize(key)
        if norm in normalized_map:
            set_role("quantity", normalized_map[norm], 0.99, "business_override_quantity")
            break

    # 5. Status
    for key in [
        "status",
        "order_status",
        "delivery_status",
        "shipment_status",
    ]:
        norm = SchemaMapper._normalize(key)
        if norm in normalized_map:
            set_role("status", normalized_map[norm], 0.95, "business_override_status")
            break

    return extracted


def detect_schema(df: pd.DataFrame) -> Dict[str, Dict]:
    df = clean_dataframe(df)
    df = parse_dates(df)

    suggestions = SchemaMapper.suggest_roles(df)

    extracted: Dict[str, Dict] = {}

    for item in suggestions:
        role = item["detected_role"]
        column = item["column_name"]
        confidence = item["confidence"]
        reasons = item.get("reasons", [])

        if role == "unknown":
            continue

        if role not in extracted or confidence > extracted[role]["confidence"]:
            extracted[role] = {
                "column": column,
                "confidence": confidence,
                "reasons": reasons,
            }

    # Final override layer for business datasets
    extracted = _apply_business_role_overrides(df, extracted)

    # Ensure all expected roles are present
    for role in ROLE_SYNONYMS.keys():
        extracted.setdefault(
            role,
            {
                "column": None,
                "confidence": 0.0,
                "reasons": [],
            },
        )

    return extracted