from typing import Dict, Any, List, Optional
import pandas as pd


def _safe_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    temp = df.copy()
    for col in columns:
        if col in temp.columns:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
    return temp


def _safe_datetime(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    temp = df.copy()
    for col in columns:
        if col in temp.columns:
            temp[col] = pd.to_datetime(temp[col], errors="coerce", dayfirst=True)
    return temp


def _normalize(text: Any) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def _resolve_schema_column_value(role_value: Any) -> Optional[str]:
    if isinstance(role_value, str) and role_value.strip():
        return role_value.strip()

    if isinstance(role_value, dict):
        column = role_value.get("column") or role_value.get("name")
        if isinstance(column, str) and column.strip():
            return column.strip()

    return None


def _build_normalized_column_map(df: pd.DataFrame) -> Dict[str, str]:
    return {_normalize(col): col for col in df.columns}


def _resolve_role_column(
    df: pd.DataFrame,
    schema_suggestions: Dict[str, Any],
    role_names: List[str],
    fallback_names: List[str],
) -> Optional[str]:
    normalized_map = _build_normalized_column_map(df)

    for role in role_names:
        if role in schema_suggestions:
            resolved = _resolve_schema_column_value(schema_suggestions.get(role))
            if resolved and resolved in df.columns:
                return resolved

    for fallback in fallback_names:
        if _normalize(fallback) in normalized_map:
            return normalized_map[_normalize(fallback)]

    for norm_col, original_col in normalized_map.items():
        for fallback in fallback_names:
            fallback_norm = _normalize(fallback)
            if fallback_norm in norm_col or norm_col in fallback_norm:
                return original_col

    return None


def format_inr_short(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "₹0"

    if value >= 10000000:
        return f"₹{value / 10000000:.2f} Cr"
    if value >= 100000:
        return f"₹{value / 100000:.2f} L"
    return f"₹{value:,.0f}"


def build_revenue_trend_summary_text(summary: Dict[str, Any]) -> str:
    if not summary:
        return "Revenue trend summary not available."

    trend = str(summary.get("trend_label", "stable")).lower()
    growth = float(summary.get("growth_percent", 0) or 0)
    first_val = format_inr_short(summary.get("first_value", 0))
    last_val = format_inr_short(summary.get("last_value", 0))
    avg_val = format_inr_short(summary.get("average_value", 0))
    points = int(summary.get("points_count", 0) or 0)

    if trend == "increasing":
        direction_text = "shows a clear upward trajectory"
    elif trend == "decreasing":
        direction_text = "shows a declining pattern"
    else:
        direction_text = "remains relatively stable"

    return (
        f"Revenue {direction_text}, moving from {first_val} to {last_val} "
        f"across {points} periods, with an overall change of {growth:.2f}%. "
        f"The average revenue observed during this period was {avg_val}."
    )


def build_quantity_trend_summary_text(summary: Dict[str, Any]) -> str:
    if not summary:
        return "Quantity trend summary not available."

    trend = str(summary.get("trend_label", "stable")).lower()
    growth = float(summary.get("growth_percent", 0) or 0)
    first_val = summary.get("first_value", 0)
    last_val = summary.get("last_value", 0)
    avg_val = float(summary.get("average_value", 0) or 0)
    points = int(summary.get("points_count", 0) or 0)

    if trend == "increasing":
        direction_text = "is trending upward"
    elif trend == "decreasing":
        direction_text = "is showing a decline"
    else:
        direction_text = "is relatively stable"

    return (
        f"Quantity demand {direction_text}, moving from {first_val} to {last_val} "
        f"across {points} observed periods, with an overall change of {growth:.2f}%. "
        f"The average quantity observed was {avg_val:.2f}."
    )


def build_concentration_summary_text(summary: Dict[str, Any], label: str) -> str:
    if not summary:
        return f"{label} concentration insight not available."

    top1 = float(summary.get("top_1_share_percent", 0) or 0)
    top3 = float(summary.get("top_3_share_percent", 0) or 0)
    risk = str(summary.get("concentration_risk", "unknown")).lower()

    if risk == "high":
        risk_text = "a high concentration risk, which suggests dependency on a limited set of contributors"
    elif risk == "medium":
        risk_text = "a medium concentration risk, indicating moderate dependency on top contributors"
    else:
        risk_text = "a low concentration risk, showing relatively diversified contribution"

    return (
        f"{label} concentration analysis shows that the top contributor accounts for "
        f"{top1:.2f}% of the total, while the top three contributors together account for "
        f"{top3:.2f}%. This indicates {risk_text}."
    )


def build_weakest_region_summary_text(summary: Dict[str, Any]) -> str:
    if not summary:
        return "Weak region insight not available."

    weakest_group = summary.get("weakest_group")
    weakest_value = summary.get("weakest_value")
    strongest_group = summary.get("strongest_group")
    strongest_value = summary.get("strongest_value")

    if weakest_group is None or strongest_group is None:
        return "Weak region insight not available."

    return (
        f"{weakest_group} is currently the weakest performing region with revenue of "
        f"{format_inr_short(weakest_value)}, while {strongest_group} is the strongest "
        f"with revenue of {format_inr_short(strongest_value)}."
    )


def safe_group_sum(
    df: pd.DataFrame,
    group_col: Optional[str],
    value_col: Optional[str],
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not group_col or not value_col:
        return []
    if group_col not in df.columns or value_col not in df.columns:
        return []

    temp = _safe_numeric(df, [value_col])
    temp = temp.dropna(subset=[group_col, value_col])

    if temp.empty:
        return []

    grouped = (
        temp.groupby(group_col, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    if top_n is not None and top_n > 0:
        grouped = grouped.head(top_n)

    return grouped.to_dict(orient="records")


def safe_group_count(
    df: pd.DataFrame,
    group_col: Optional[str],
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not group_col or group_col not in df.columns:
        return []

    grouped = (
        df[group_col]
        .astype(str)
        .value_counts(dropna=False)
        .reset_index()
    )
    grouped.columns = [group_col, "count"]

    if top_n is not None and top_n > 0:
        grouped = grouped.head(top_n)

    return grouped.to_dict(orient="records")


def safe_time_series_sum(
    df: pd.DataFrame,
    date_col: Optional[str],
    value_col: Optional[str],
    freq: str = "D",
) -> List[Dict[str, Any]]:
    if not date_col or not value_col:
        return []
    if date_col not in df.columns or value_col not in df.columns:
        return []

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce", dayfirst=True)
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col])

    if temp.empty:
        return []

    if freq == "M":
        temp["period"] = temp[date_col].dt.to_period("M").dt.to_timestamp()
    else:
        temp["period"] = temp[date_col].dt.floor("D")

    grouped = (
        temp.groupby("period")[value_col]
        .sum()
        .reset_index()
        .sort_values(by="period")
    )

    grouped["date"] = grouped["period"].dt.strftime("%Y-%m-%d")
    grouped["value"] = grouped[value_col]

    return grouped[["date", "value"]].to_dict(orient="records")


def safe_group_sum_by_two_keys(
    df: pd.DataFrame,
    group_col_1: Optional[str],
    group_col_2: Optional[str],
    value_col: Optional[str],
    top_n_per_group: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not group_col_1 or not group_col_2 or not value_col:
        return []
    if group_col_1 not in df.columns or group_col_2 not in df.columns or value_col not in df.columns:
        return []

    temp = _safe_numeric(df, [value_col])
    temp = temp.dropna(subset=[group_col_1, group_col_2, value_col])

    if temp.empty:
        return []

    grouped = (
        temp.groupby([group_col_1, group_col_2], dropna=False)[value_col]
        .sum()
        .reset_index()
        .sort_values(by=[group_col_1, value_col], ascending=[True, False])
    )

    if top_n_per_group is not None and top_n_per_group > 0:
        grouped = grouped.groupby(group_col_1, dropna=False).head(top_n_per_group).reset_index(drop=True)

    return grouped.to_dict(orient="records")


def safe_group_sum_by_three_keys(
    df: pd.DataFrame,
    group_col_1: Optional[str],
    group_col_2: Optional[str],
    group_col_3: Optional[str],
    value_col: Optional[str],
    top_n_per_group: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not group_col_1 or not group_col_2 or not group_col_3 or not value_col:
        return []
    needed = {group_col_1, group_col_2, group_col_3, value_col}
    if not needed.issubset(df.columns):
        return []

    temp = _safe_numeric(df, [value_col])
    temp = temp.dropna(subset=[group_col_1, group_col_2, group_col_3, value_col])

    if temp.empty:
        return []

    grouped = (
        temp.groupby([group_col_1, group_col_2, group_col_3], dropna=False)[value_col]
        .sum()
        .reset_index()
        .sort_values(by=[group_col_1, group_col_2, value_col], ascending=[True, True, False])
    )

    if top_n_per_group is not None and top_n_per_group > 0:
        grouped = grouped.groupby([group_col_1, group_col_2], dropna=False).head(top_n_per_group).reset_index(drop=True)

    return grouped.to_dict(orient="records")


def summarize_region_product_pattern(
    top_products_by_region: List[Dict[str, Any]],
    region_col: str,
    product_col: str,
    metric_col: str
) -> Dict[str, Any]:
    if not top_products_by_region:
        return {}

    temp = pd.DataFrame(top_products_by_region)
    required_cols = {region_col, product_col, metric_col}
    if temp.empty or not required_cols.issubset(set(temp.columns)):
        return {}

    temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce")
    temp = temp.dropna(subset=[region_col, product_col, metric_col])

    if temp.empty:
        return {}

    region_totals = (
        temp.groupby(region_col)[metric_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    unique_top_products = temp[product_col].nunique()
    unique_regions = temp[region_col].nunique()

    highest_region = region_totals.iloc[0]
    lowest_region = region_totals.iloc[-1]

    repeated_products = (
        temp.groupby(product_col)[region_col]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
    )
    repeated_products = repeated_products[repeated_products[region_col] > 1]

    return {
        "top_region_for_top_products": str(highest_region[region_col]),
        "top_region_top_products_revenue": round(float(highest_region[metric_col]), 2),
        "lowest_region_for_top_products": str(lowest_region[region_col]),
        "lowest_region_top_products_revenue": round(float(lowest_region[metric_col]), 2),
        "regions_covered": int(unique_regions),
        "unique_top_products_count": int(unique_top_products),
        "products_repeating_across_regions": repeated_products.to_dict(orient="records"),
        "pattern_summary": (
            f"{str(highest_region[region_col])} contributes the highest revenue among top-performing regional products."
            if unique_regions > 1 else
            "Only one region is available for top-product regional comparison."
        ),
    }


def summarize_status_region_pattern(
    df: pd.DataFrame,
    status_col: Optional[str],
    region_col: Optional[str],
    product_col: Optional[str],
    order_id_col: Optional[str] = None,
) -> Dict[str, Any]:
    if not status_col or not region_col or not product_col:
        return {}
    needed = {status_col, region_col, product_col}
    if not needed.issubset(df.columns):
        return {}

    temp = df[[status_col, region_col, product_col] + ([order_id_col] if order_id_col and order_id_col in df.columns else [])].copy()
    temp[status_col] = temp[status_col].astype(str).str.strip().str.lower()
    temp[region_col] = temp[region_col].astype(str).str.strip()
    temp[product_col] = temp[product_col].astype(str).str.strip()
    temp = temp.dropna(subset=[status_col, region_col, product_col])

    if temp.empty:
        return {}

    pending_df = temp[temp[status_col].isin(["pending", "not delivered", "not_delivered"])]
    if pending_df.empty:
        return {
            "pending_orders_by_region": [],
            "top_pending_region": None,
            "top_pending_products": [],
            "summary": "No pending-order concentration is visible in the current analysis.",
        }

    if order_id_col and order_id_col in pending_df.columns:
        pending_by_region = (
            pending_df.groupby(region_col)[order_id_col]
            .count()
            .sort_values(ascending=False)
            .reset_index(name="pending_orders")
        )
    else:
        pending_by_region = (
            pending_df.groupby(region_col)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="pending_orders")
        )

    pending_products = (
        pending_df.groupby([region_col, product_col])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="pending_count")
    )

    top_region = pending_by_region.iloc[0][region_col] if not pending_by_region.empty else None

    return {
        "pending_orders_by_region": pending_by_region.to_dict(orient="records"),
        "top_pending_region": str(top_region) if top_region is not None else None,
        "top_pending_products": pending_products.head(10).to_dict(orient="records"),
        "summary": (
            f"Pending orders are most concentrated in {top_region}."
            if top_region is not None else
            "Pending-order concentration could not be determined."
        ),
    }


def detect_delivery_delays(
    df: pd.DataFrame,
    shipment_col: Optional[str],
    delivery_col: Optional[str]
) -> Dict[str, Any]:
    if not shipment_col or not delivery_col:
        return {}
    if shipment_col not in df.columns or delivery_col not in df.columns:
        return {}

    temp = _safe_datetime(df, [shipment_col, delivery_col])
    temp = temp.dropna(subset=[shipment_col, delivery_col])

    if temp.empty:
        return {}

    temp["delivery_delay_days"] = (temp[delivery_col] - temp[shipment_col]).dt.days
    temp = temp.dropna(subset=["delivery_delay_days"])
    temp = temp[temp["delivery_delay_days"] >= 0]

    if temp.empty:
        return {}

    avg_delay = float(temp["delivery_delay_days"].mean())
    max_delay = float(temp["delivery_delay_days"].max())
    min_delay = float(temp["delivery_delay_days"].min())
    p90_delay = float(temp["delivery_delay_days"].quantile(0.90))

    return {
        "average_delivery_days": round(avg_delay, 2),
        "max_delivery_days": int(max_delay),
        "min_delivery_days": int(min_delay),
        "p90_delivery_days": round(p90_delay, 2),
        "delayed_shipments_count": int((temp["delivery_delay_days"] > avg_delay).sum()),
    }


def detect_supplier_delay_risk(
    df: pd.DataFrame,
    supplier_col: Optional[str],
    shipment_col: Optional[str],
    delivery_col: Optional[str]
) -> Dict[str, Any]:
    if not supplier_col or not shipment_col or not delivery_col:
        return {}
    if supplier_col not in df.columns or shipment_col not in df.columns or delivery_col not in df.columns:
        return {}

    temp = _safe_datetime(df, [shipment_col, delivery_col])
    temp = temp.dropna(subset=[supplier_col, shipment_col, delivery_col])

    if temp.empty:
        return {}

    temp["delay_days"] = (temp[delivery_col] - temp[shipment_col]).dt.days
    temp = temp[temp["delay_days"] >= 0]

    if temp.empty:
        return {}

    grouped = (
        temp.groupby(supplier_col)["delay_days"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    if grouped.empty:
        return {}

    worst = grouped.iloc[0]
    best = grouped.iloc[-1]

    return {
        "supplier_delay_ranking": grouped.to_dict(orient="records"),
        "highest_delay_supplier": str(worst[supplier_col]),
        "highest_average_delay_days": round(float(worst["delay_days"]), 2),
        "lowest_delay_supplier": str(best[supplier_col]),
        "lowest_average_delay_days": round(float(best["delay_days"]), 2),
    }


def summarize_trend(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not time_series or len(time_series) < 2:
        return {}

    try:
        values = [float(row.get("value", 0)) for row in time_series]
    except Exception:
        return {}

    if len(values) < 2:
        return {}

    first_value = values[0]
    last_value = values[-1]
    avg_value = sum(values) / len(values)

    if first_value == 0:
        growth_percent = 0.0
    else:
        growth_percent = ((last_value - first_value) / first_value) * 100.0

    if last_value > first_value * 1.10:
        trend_label = "increasing"
    elif last_value < first_value * 0.90:
        trend_label = "decreasing"
    else:
        trend_label = "stable"

    return {
        "trend_label": trend_label,
        "growth_percent": round(growth_percent, 2),
        "first_value": round(first_value, 2),
        "last_value": round(last_value, 2),
        "average_value": round(avg_value, 2),
        "points_count": len(values),
    }


def summarize_concentration(
    grouped_records: List[Dict[str, Any]],
    value_col: str
) -> Dict[str, Any]:
    if not grouped_records:
        return {}

    try:
        values = [float(item.get(value_col, 0)) for item in grouped_records]
    except Exception:
        return {}

    total = sum(values)
    if total <= 0:
        return {}

    top_share = (values[0] / total) * 100.0 if values else 0.0
    top_3_share = (sum(values[:3]) / total) * 100.0 if values else 0.0

    return {
        "top_1_share_percent": round(top_share, 2),
        "top_3_share_percent": round(top_3_share, 2),
        "concentration_risk": (
            "high" if top_3_share >= 75 else
            "medium" if top_3_share >= 50 else
            "low"
        ),
    }


def find_weakest_group(
    df: pd.DataFrame,
    group_col: Optional[str],
    value_col: Optional[str]
) -> Dict[str, Any]:
    if not group_col or not value_col:
        return {}
    if group_col not in df.columns or value_col not in df.columns:
        return {}

    temp = _safe_numeric(df, [value_col])
    temp = temp.dropna(subset=[group_col, value_col])

    if temp.empty:
        return {}

    grouped = (
        temp.groupby(group_col, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )

    if grouped.empty:
        return {}

    weakest = grouped.iloc[0]
    strongest = grouped.iloc[-1]

    return {
        "weakest_group": str(weakest[group_col]),
        "weakest_value": round(float(weakest[value_col]), 2),
        "strongest_group": str(strongest[group_col]),
        "strongest_value": round(float(strongest[value_col]), 2),
    }


def build_business_summary(
    product_records: List[Dict[str, Any]],
    region_records: List[Dict[str, Any]],
    trend_summary: Dict[str, Any],
    metric_label: str
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}

    if product_records:
        first_product = product_records[0]
        non_value_keys = [k for k in first_product.keys() if k != metric_label]
        if non_value_keys and metric_label in first_product:
            summary["top_product_name"] = str(first_product[non_value_keys[0]])
            summary[f"top_product_{metric_label}"] = round(float(first_product[metric_label]), 2)

    if region_records:
        first_region = region_records[0]
        non_value_keys = [k for k in first_region.keys() if k != metric_label]
        if non_value_keys and metric_label in first_region:
            summary["top_region_name"] = str(first_region[non_value_keys[0]])
            summary[f"top_region_{metric_label}"] = round(float(first_region[metric_label]), 2)

    if trend_summary:
        summary["trend_label"] = trend_summary.get("trend_label")
        summary["growth_percent"] = trend_summary.get("growth_percent")

    return summary


def generate_insights(
    df: pd.DataFrame,
    schema_suggestions: Dict[str, Any],
    capabilities: Dict[str, Any]
) -> Dict[str, Any]:
    insights = {
        "sales": {},
        "inventory": {},
        "logistics": {}
    }

    date_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["date"],
        fallback_names=["po date", "po_date", "order date", "order_date", "transaction date", "date"],
    )

    product_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["product"],
        fallback_names=["product", "item", "sku", "material", "product_name", "description"],
    )

    region_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["region"],
        fallback_names=["region", "city", "state", "country", "zone", "location", "branch"],
    )

    revenue_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["revenue", "sales"],
        fallback_names=["total price", "total_price", "revenue", "sales", "amount", "total amount", "total_amount", "net_value", "value"],
    )

    quantity_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["quantity"],
        fallback_names=["quantity", "quamtity", "qty", "units", "sold_qty"],
    )

    status_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["order_status", "status"],
        fallback_names=["status", "delivery_status", "order_status", "shipment_status"],
    )

    order_id_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["order_id"],
        fallback_names=["order_id", "order id", "sales_id", "invoice_id", "sales_order", "document_no", "doc_no"],
    )

    shipment_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["shipment_date"],
        fallback_names=["shipment_date", "shipment date", "dispatch_date", "dispatch date"],
    )

    delivery_col = _resolve_role_column(
        df,
        schema_suggestions,
        role_names=["delivery_date"],
        fallback_names=["delivery_date", "delivery date", "invoice_date", "invoice date"],
    )

    sales_minimum_ready = any([product_col, region_col, revenue_col, quantity_col, date_col])

    if sales_minimum_ready:
        top_products_revenue = safe_group_sum(df, product_col, revenue_col, top_n=None) if product_col and revenue_col else []
        top_regions_revenue = safe_group_sum(df, region_col, revenue_col, top_n=None) if region_col and revenue_col else []
        top_products_quantity = safe_group_sum(df, product_col, quantity_col, top_n=None) if product_col and quantity_col else []
        top_regions_quantity = safe_group_sum(df, region_col, quantity_col, top_n=None) if region_col and quantity_col else []

        print("REGION COL:", region_col)
        print("QUANTITY COL:", quantity_col)
        print("DF COLUMNS:", list(df.columns))
        print("TOP REGIONS QUANTITY TEST:", safe_group_sum(df, region_col, quantity_col, top_n=None)[:3])

        revenue_trend = safe_time_series_sum(df, date_col, revenue_col, freq="D") if date_col and revenue_col else []
        quantity_trend = safe_time_series_sum(df, date_col, quantity_col, freq="D") if date_col and quantity_col else []

        revenue_trend_summary_raw = summarize_trend(revenue_trend)
        quantity_trend_summary_raw = summarize_trend(quantity_trend)

        product_concentration_raw = summarize_concentration(top_products_revenue, revenue_col) if revenue_col else {}
        region_concentration_raw = summarize_concentration(top_regions_revenue, revenue_col) if revenue_col else {}
        weakest_region_summary_raw = find_weakest_group(df, region_col, revenue_col) if region_col and revenue_col else {}

        revenue_trend_summary = build_revenue_trend_summary_text(revenue_trend_summary_raw)
        quantity_trend_summary = build_quantity_trend_summary_text(quantity_trend_summary_raw)
        product_concentration = build_concentration_summary_text(product_concentration_raw, "Product")
        region_concentration = build_concentration_summary_text(region_concentration_raw, "Region")
        weakest_region_summary = build_weakest_region_summary_text(weakest_region_summary_raw)

        sales_summary = build_business_summary(
            top_products_revenue,
            top_regions_revenue,
            revenue_trend_summary_raw,
            revenue_col,
        ) if revenue_col else {}

        top_products_by_region = (
            safe_group_sum_by_two_keys(df, region_col, product_col, revenue_col, top_n_per_group=None)
            if region_col and product_col and revenue_col else []
        )

        top_products_by_region_status = (
            safe_group_sum_by_three_keys(df, region_col, status_col, product_col, revenue_col, top_n_per_group=None)
            if region_col and status_col and product_col and revenue_col else []
        )

        region_revenue_pattern_summary = (
            summarize_region_product_pattern(
                top_products_by_region=top_products_by_region,
                region_col=region_col,
                product_col=product_col,
                metric_col=revenue_col,
            )
            if top_products_by_region and region_col and product_col and revenue_col else {}
        )

        pending_order_pattern_summary = summarize_status_region_pattern(
            df=df,
            status_col=status_col,
            region_col=region_col,
            product_col=product_col,
            order_id_col=order_id_col,
        )

        insights["sales"] = {
            "resolved_columns": {
                "date": date_col,
                "product": product_col,
                "region": region_col,
                "revenue": revenue_col,
                "quantity": quantity_col,
                "status": status_col,
                "order_id": order_id_col,
            },
            "analysis_metric": "revenue" if revenue_col else "quantity" if quantity_col else None,
            "top_products": top_products_revenue if revenue_col else top_products_quantity,
            "top_regions": top_regions_revenue if revenue_col else top_regions_quantity,
            "top_products_by_revenue": top_products_revenue,
            "top_regions_by_revenue": top_regions_revenue,
            "top_products_by_quantity": top_products_quantity,
            "top_regions_by_quantity": top_regions_quantity,
            "top_products_by_region": top_products_by_region,
            "top_products_by_region_status": top_products_by_region_status,
            "primary_trend": revenue_trend if revenue_col else quantity_trend,
            "revenue_trend": revenue_trend,
            "quantity_trend": quantity_trend,
            "primary_trend_summary": revenue_trend_summary if revenue_col else quantity_trend_summary,
            "revenue_trend_summary_raw": revenue_trend_summary_raw,
            "revenue_trend_summary": revenue_trend_summary,
            "quantity_trend_summary_raw": quantity_trend_summary_raw,
            "quantity_trend_summary": quantity_trend_summary,
            "product_concentration_summary_raw": product_concentration_raw,
            "product_concentration_summary": product_concentration,
            "region_concentration_summary_raw": region_concentration_raw,
            "region_concentration_summary": region_concentration,
            "weakest_region_summary_raw": weakest_region_summary_raw,
            "weakest_region_summary": weakest_region_summary,
            "region_revenue_pattern_summary": region_revenue_pattern_summary,
            "pending_order_pattern_summary": pending_order_pattern_summary,
            "sales_summary": sales_summary,
        }

    if capabilities.get("inventory", {}).get("enabled") or all(col in df.columns for col in ["product", "stock", "reorder_level"]):
        stock_col = _resolve_role_column(
            df,
            schema_suggestions,
            role_names=["stock"],
            fallback_names=["stock", "inventory", "inventory_level"],
        )
        reorder_col = _resolve_role_column(
            df,
            schema_suggestions,
            role_names=["reorder_level"],
            fallback_names=["reorder_level", "reorder level", "min_stock", "minimum_stock"],
        )

        low_stock_items = []
        inventory_summary = {}

        if product_col and stock_col and reorder_col and all(col in df.columns for col in [product_col, stock_col, reorder_col]):
            temp = _safe_numeric(df, [stock_col, reorder_col])
            temp = temp.dropna(subset=[product_col, stock_col, reorder_col])

            if not temp.empty:
                low_stock_df = temp[temp[stock_col] <= temp[reorder_col]].copy()

                low_stock_items = (
                    low_stock_df[[product_col, stock_col, reorder_col]]
                    .to_dict(orient="records")
                )

                inventory_summary = {
                    "low_stock_count": int(low_stock_df.shape[0]),
                    "inventory_pressure_level": (
                        "high" if low_stock_df.shape[0] >= 20 else
                        "medium" if low_stock_df.shape[0] >= 5 else
                        "low"
                    ),
                    "inventory_summary": (
                        f"{int(low_stock_df.shape[0])} records are at or below reorder level."
                        if low_stock_df.shape[0] > 0 else
                        "No immediate low-stock pressure is visible in the current analysis."
                    )
                }

        insights["inventory"] = {
            "low_stock_items": low_stock_items,
            "inventory_summary": inventory_summary,
            "inventory_pressure_level": inventory_summary.get("inventory_pressure_level"),
        }

    if capabilities.get("logistics", {}).get("enabled") or any([status_col, shipment_col, delivery_col]):
        supplier_col = None
        for candidate in ["supplier_name", "supplier", "vendor_name", "vendor", "Supplier"]:
            if candidate in df.columns:
                supplier_col = candidate
                break

        status_distribution = safe_group_count(df, status_col, top_n=None) if status_col else []
        region_distribution = safe_group_count(df, region_col, top_n=None) if region_col else []
        delivery_delay_summary = detect_delivery_delays(df, shipment_col, delivery_col) if shipment_col and delivery_col else {}
        supplier_delay_summary = detect_supplier_delay_risk(df, supplier_col, shipment_col, delivery_col) if supplier_col and shipment_col and delivery_col else {}

        avg_delay = delivery_delay_summary.get("average_delivery_days", 0)

        logistics_summary = {
            "top_status": status_distribution[0] if status_distribution else {},
            "top_region": region_distribution[0] if region_distribution else {},
            "delay_risk_level": (
                "high" if avg_delay >= 7 else
                "medium" if avg_delay >= 3 else
                "low"
            ),
            "highest_delay_supplier": supplier_delay_summary.get("highest_delay_supplier"),
            "highest_average_delay_days": supplier_delay_summary.get("highest_average_delay_days"),
        }

        insights["logistics"] = {
            "resolved_columns": {
                "status": status_col,
                "shipment_date": shipment_col,
                "delivery_date": delivery_col,
                "region": region_col,
            },
            "status_distribution": status_distribution,
            "region_distribution": region_distribution,
            "delivery_delay_summary": delivery_delay_summary,
            "supplier_delay_summary": supplier_delay_summary,
            "logistics_summary": logistics_summary,
            "delay_risk": logistics_summary.get("delay_risk_level"),
        }

    return insights