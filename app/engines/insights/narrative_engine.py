from typing import Dict, Any, List, Optional


def format_currency(value: float) -> str:
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def get_top_item(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if items and isinstance(items, list) and len(items) > 0:
        return items[0]
    return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _join_sentences(lines: List[str]) -> str:
    cleaned = [line.strip() for line in lines if isinstance(line, str) and line.strip()]
    return " ".join(cleaned)


def _append_summary_or_dict_lines(
    lines: List[str],
    summary_value: Any,
    dict_handler,
) -> None:
    """
    Handles both:
    - new string-style summaries
    - old dict-style summaries
    """
    if isinstance(summary_value, str) and summary_value.strip():
        lines.append(summary_value.strip())
    elif isinstance(summary_value, dict):
        dict_handler(summary_value)


def generate_narrative(
    schema_suggestions: Dict[str, Any],
    capabilities: Dict[str, Any],
    profile: Dict[str, Any],
    insights: Dict[str, Any]
) -> Dict[str, Any]:
    narratives = {
        "dataset_summary": "",
        "capability_summary": "",
        "sales_summary": "",
        "inventory_summary": "",
        "logistics_summary": "",
        "data_quality_summary": "",
        "executive_summary": "",
    }

    dataset_profile = profile.get("dataset_profile", {}) or {}

    total_rows = _safe_int(dataset_profile.get("total_rows", 0))
    total_columns = _safe_int(dataset_profile.get("total_columns", 0))
    duplicate_rows = _safe_int(dataset_profile.get("duplicate_rows", 0))
    datetime_cols = dataset_profile.get("datetime_candidate_columns", []) or []
    numeric_cols = dataset_profile.get("numeric_columns", []) or []
    categorical_cols = dataset_profile.get("categorical_columns", []) or []
    null_counts = dataset_profile.get("null_counts", {}) or {}
    total_nulls = sum(null_counts.values()) if isinstance(null_counts, dict) else 0

    # ---------------- DATASET SUMMARY ----------------
    dataset_lines = [
        f"The uploaded dataset contains {total_rows} rows and {total_columns} columns.",
        f"It includes {len(numeric_cols)} numeric columns, {len(categorical_cols)} categorical columns, and {len(datetime_cols)} date-like columns.",
    ]
    narratives["dataset_summary"] = _join_sentences(dataset_lines)

    # ---------------- CAPABILITY SUMMARY ----------------
    enabled_caps = []
    partial_caps = []

    for cap_name, cap_data in capabilities.items():
        if not isinstance(cap_data, dict):
            continue

        if cap_data.get("enabled") is True or str(cap_data.get("status", "")).lower() == "enabled":
            enabled_caps.append(cap_name)
        elif str(cap_data.get("status", "")).lower() == "partial":
            partial_caps.append(cap_name)

    capability_lines = []
    if enabled_caps:
        capability_lines.append(
            f"The dataset strongly supports the following business analysis areas: {', '.join(enabled_caps)}."
        )
    else:
        capability_lines.append(
            "The dataset does not strongly support any predefined business capability yet."
        )

    if partial_caps:
        capability_lines.append(
            f"Partial support is also available for: {', '.join(partial_caps)}."
        )

    narratives["capability_summary"] = _join_sentences(capability_lines)

    # ---------------- DATA QUALITY SUMMARY ----------------
    quality_lines = [
        f"The dataset contains {duplicate_rows} duplicate rows and {total_nulls} missing values overall."
    ]

    if total_nulls > 0:
        quality_lines.append(
            "Missing values may reduce the reliability of trend, forecast, or decision outputs if they affect important mapped fields."
        )

    if duplicate_rows > 0:
        quality_lines.append(
            "Duplicate rows should be reviewed because they may distort aggregate business signals."
        )

    narratives["data_quality_summary"] = _join_sentences(quality_lines)

    # ---------------- SALES SUMMARY ----------------
    sales_cap = capabilities.get("sales", {}) or {}
    sales_insights = insights.get("sales", {}) or {}

    if sales_cap.get("enabled") is True or str(sales_cap.get("status", "")).lower() == "enabled":
        sales_lines = ["Sales analysis is enabled for this dataset."]

        sales_summary = sales_insights.get("sales_summary", {}) or {}
        top_products = sales_insights.get("top_products_by_revenue", []) or []
        top_regions = sales_insights.get("top_regions_by_revenue", []) or []

        top_product_name = sales_summary.get("top_product_name") if isinstance(sales_summary, dict) else None
        top_product_revenue = sales_summary.get("top_product_revenue") if isinstance(sales_summary, dict) else None
        if top_product_name is not None and top_product_revenue is not None:
            sales_lines.append(
                f"The top-performing product is {top_product_name}, with total revenue of {format_currency(top_product_revenue)}."
            )
        else:
            top_product = get_top_item(top_products)
            if top_product:
                product_key = next(
                    (k for k in top_product.keys() if k.lower() not in ["total_amount", "revenue", "sales", "sales_amount", "value"]),
                    None
                )
                value_key = next(
                    (k for k in top_product.keys() if k.lower() in ["total_amount", "revenue", "sales", "sales_amount", "value"]),
                    None
                )
                if product_key and value_key:
                    sales_lines.append(
                        f"The top-performing product is {top_product[product_key]}, with total revenue of {format_currency(top_product[value_key])}."
                    )

        top_region_name = sales_summary.get("top_region_name") if isinstance(sales_summary, dict) else None
        top_region_revenue = sales_summary.get("top_region_revenue") if isinstance(sales_summary, dict) else None
        if top_region_name is not None and top_region_revenue is not None:
            sales_lines.append(
                f"The strongest visible region is {top_region_name}, contributing revenue of {format_currency(top_region_revenue)}."
            )
        else:
            top_region = get_top_item(top_regions)
            if top_region:
                region_key = next(
                    (k for k in top_region.keys() if k.lower() not in ["total_amount", "revenue", "sales", "sales_amount", "value"]),
                    None
                )
                value_key = next(
                    (k for k in top_region.keys() if k.lower() in ["total_amount", "revenue", "sales", "sales_amount", "value"]),
                    None
                )
                if region_key and value_key:
                    sales_lines.append(
                        f"The strongest visible region is {top_region[region_key]}, contributing revenue of {format_currency(top_region[value_key])}."
                    )

        revenue_trend_summary = sales_insights.get("revenue_trend_summary")

        def _handle_revenue_trend_summary(summary_dict: Dict[str, Any]) -> None:
            trend_label = summary_dict.get("trend_label")
            growth_percent = summary_dict.get("growth_percent")
            points_count = summary_dict.get("points_count")

            if trend_label:
                if growth_percent is not None:
                    sales_lines.append(
                        f"Revenue trend appears {trend_label} across {points_count or 0} time points, with approximately {growth_percent}% movement over the observed series."
                    )
                else:
                    sales_lines.append(
                        f"Revenue trend appears {trend_label} across the observed time series."
                    )

        _append_summary_or_dict_lines(sales_lines, revenue_trend_summary, _handle_revenue_trend_summary)

        quantity_trend_summary = sales_insights.get("quantity_trend_summary")

        def _handle_quantity_trend_summary(summary_dict: Dict[str, Any]) -> None:
            trend_label = summary_dict.get("trend_label")
            growth_percent = summary_dict.get("growth_percent")
            points_count = summary_dict.get("points_count")

            if trend_label:
                if growth_percent is not None:
                    sales_lines.append(
                        f"Quantity trend appears {trend_label} across {points_count or 0} time points, with approximately {growth_percent}% movement over the observed series."
                    )
                else:
                    sales_lines.append(
                        f"Quantity trend appears {trend_label} across the observed time series."
                    )

        _append_summary_or_dict_lines(sales_lines, quantity_trend_summary, _handle_quantity_trend_summary)

        product_concentration = sales_insights.get("product_concentration_summary")

        def _handle_product_concentration(summary_dict: Dict[str, Any]) -> None:
            concentration_risk = summary_dict.get("concentration_risk")
            top_3_share = summary_dict.get("top_3_share_percent")

            if concentration_risk and top_3_share is not None:
                sales_lines.append(
                    f"Product revenue concentration risk is currently {concentration_risk}, with the top three visible products contributing about {top_3_share}% of the grouped revenue shown in the current summary."
                )

        _append_summary_or_dict_lines(sales_lines, product_concentration, _handle_product_concentration)

        region_concentration = sales_insights.get("region_concentration_summary")

        def _handle_region_concentration(summary_dict: Dict[str, Any]) -> None:
            concentration_risk = summary_dict.get("concentration_risk")
            top_3_share = summary_dict.get("top_3_share_percent")

            if concentration_risk and top_3_share is not None:
                sales_lines.append(
                    f"Regional revenue concentration risk is currently {concentration_risk}, with the top three visible regions contributing about {top_3_share}% of the grouped revenue shown in the current summary."
                )

        _append_summary_or_dict_lines(sales_lines, region_concentration, _handle_region_concentration)

        weakest_region_summary = sales_insights.get("weakest_region_summary")

        def _handle_weakest_region(summary_dict: Dict[str, Any]) -> None:
            weakest_group = summary_dict.get("weakest_group")
            strongest_group = summary_dict.get("strongest_group")

            if weakest_group and strongest_group:
                sales_lines.append(
                    f"Regional comparison also shows a weaker area in {weakest_group}, while {strongest_group} appears strongest in the current grouped revenue view."
                )

        _append_summary_or_dict_lines(sales_lines, weakest_region_summary, _handle_weakest_region)

        region_revenue_pattern_summary = sales_insights.get("region_revenue_pattern_summary")

        def _handle_region_pattern(summary_dict: Dict[str, Any]) -> None:
            pattern_summary = summary_dict.get("pattern_summary")
            if pattern_summary:
                sales_lines.append(pattern_summary)

        _append_summary_or_dict_lines(sales_lines, region_revenue_pattern_summary, _handle_region_pattern)

        pending_order_pattern_summary = sales_insights.get("pending_order_pattern_summary")

        def _handle_pending_order_pattern(summary_dict: Dict[str, Any]) -> None:
            summary_text = summary_dict.get("summary")
            if summary_text:
                sales_lines.append(summary_text)

        _append_summary_or_dict_lines(sales_lines, pending_order_pattern_summary, _handle_pending_order_pattern)

        narratives["sales_summary"] = _join_sentences(sales_lines)
    else:
        narratives["sales_summary"] = (
            "Sales analysis is not fully supported because key sales fields are missing or weakly mapped."
        )

    # ---------------- INVENTORY SUMMARY ----------------
    inventory_cap = capabilities.get("inventory", {}) or {}
    inventory_insights = insights.get("inventory", {}) or {}

    if inventory_cap.get("enabled") is True or str(inventory_cap.get("status", "")).lower() == "enabled":
        inventory_lines = ["Inventory analysis is enabled for this dataset."]

        low_stock_items = inventory_insights.get("low_stock_items", []) or []
        inventory_summary = inventory_insights.get("inventory_summary", {}) or {}

        low_stock_count = inventory_summary.get("low_stock_count") if isinstance(inventory_summary, dict) else None
        pressure_level = inventory_summary.get("inventory_pressure_level") if isinstance(inventory_summary, dict) else None

        if low_stock_count is not None:
            inventory_lines.append(
                f"There are {low_stock_count} low-stock records currently flagged for possible reorder attention."
            )
        elif low_stock_items:
            inventory_lines.append(
                f"There are {len(low_stock_items)} low-stock items currently flagged for possible reorder attention."
            )
        else:
            inventory_lines.append(
                "No immediate low-stock items were identified in the current output."
            )

        if pressure_level:
            inventory_lines.append(
                f"Overall inventory pressure is currently assessed as {pressure_level}."
            )

        if isinstance(inventory_summary, str) and inventory_summary.strip():
            inventory_lines.append(inventory_summary.strip())

        narratives["inventory_summary"] = _join_sentences(inventory_lines)
    else:
        narratives["inventory_summary"] = (
            "Inventory analysis is not available because stock, reorder level, or related fields are missing."
        )

    # ---------------- LOGISTICS SUMMARY ----------------
    logistics_cap = capabilities.get("logistics", {}) or {}
    logistics_insights = insights.get("logistics", {}) or {}

    if logistics_cap.get("enabled") is True or str(logistics_cap.get("status", "")).lower() == "enabled":
        logistics_lines = ["Logistics analysis is enabled for this dataset."]

        delay_summary = logistics_insights.get("delivery_delay_summary", {}) or {}
        logistics_summary = logistics_insights.get("logistics_summary", {}) or {}
        status_distribution = logistics_insights.get("status_distribution", []) or []

        avg_days = delay_summary.get("average_delivery_days") if isinstance(delay_summary, dict) else None
        p90_days = delay_summary.get("p90_delivery_days") if isinstance(delay_summary, dict) else None
        if avg_days is not None:
            logistics_lines.append(
                f"The average delivery time is {avg_days} days."
            )
        if p90_days is not None:
            logistics_lines.append(
                f"Higher-delay shipments can extend to around {p90_days} days at the 90th percentile."
            )

        top_status = logistics_summary.get("top_status", {}) if isinstance(logistics_summary, dict) else {}
        top_status = top_status or get_top_item(status_distribution) or {}
        if isinstance(top_status, dict) and top_status:
            status_key = next((k for k in top_status.keys() if k != "count"), None)
            if status_key:
                logistics_lines.append(
                    f"The most common logistics status currently visible is {top_status[status_key]}."
                )

        delay_risk_level = logistics_summary.get("delay_risk_level") if isinstance(logistics_summary, dict) else None
        if delay_risk_level:
            logistics_lines.append(
                f"Logistics delay risk is currently assessed as {delay_risk_level}."
            )

        if isinstance(logistics_summary, str) and logistics_summary.strip():
            logistics_lines.append(logistics_summary.strip())

        narratives["logistics_summary"] = _join_sentences(logistics_lines)
    else:
        narratives["logistics_summary"] = (
            "Logistics analysis is not fully available because shipment date, delivery date, status, or related regional fields are missing."
        )

    # ---------------- EXECUTIVE SUMMARY ----------------
    executive_lines = [
        narratives["dataset_summary"],
        narratives["capability_summary"],
    ]

    if narratives["sales_summary"]:
        executive_lines.append(narratives["sales_summary"])

    if narratives["inventory_summary"]:
        executive_lines.append(narratives["inventory_summary"])

    if narratives["logistics_summary"]:
        executive_lines.append(narratives["logistics_summary"])

    if narratives["data_quality_summary"]:
        executive_lines.append(narratives["data_quality_summary"])

    narratives["executive_summary"] = _join_sentences(executive_lines)

    return narratives