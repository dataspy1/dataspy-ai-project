from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from app.services.data.dataset_store import DatasetStore


MISSING_INFO_MESSAGE = "This is not available in the current analysis."


@dataclass
class ConditionalQueryResult:
    matched: bool
    answer: str
    context_used: List[str]
    answer_source: str = "Computed From Data"


class ConditionalQueryService:
    def __init__(self) -> None:
        self.status_aliases = {
            "delivered": "Delivered",
            "pending": "Pending",
            "in process": "In Process",
            "in-process": "In Process",
            "processing": "In Process",
            "delayed": "Delayed",
            "cancelled": "Cancelled",
            "canceled": "Cancelled",
            "shipped": "Shipped",
            "not delivered": "Not Delivered",
            "undelivered": "Not Delivered",
        }

    def answer_query(
        self,
        question: str,
        saved_filename: str,
        analysis_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not question or not str(question).strip():
            return None

        if not saved_filename or not str(saved_filename).strip():
            return None

        try:
            df = DatasetStore.load_dataframe(saved_filename)
        except Exception:
            return None

        if df is None or df.empty:
            return None

        q = self._normalize_text(question)
        working_df = self._prepare_dataframe(df)

        handlers = [
            self._handle_total_revenue,
            self._handle_total_quantity,
            self._handle_total_revenue_last_n_days,
            self._handle_average_revenue_per_order,
            self._handle_revenue_trend_summary,
            self._handle_revenue_direction,
            self._handle_highest_revenue_day,
            self._handle_lowest_revenue_day,
            self._handle_top_product,
            self._handle_top_3_products,
            self._handle_top_region,
            self._handle_top_3_regions,
            self._handle_average_value_per_product,
            self._handle_average_value_per_region,
            self._handle_product_revenue_in_region,
            self._handle_revenue_by_region,
            self._handle_revenue_by_product,
            self._handle_not_delivered_orders,
            self._handle_pending_orders_count,
            self._handle_delivered_orders_count,
            self._handle_status_counts,
            self._handle_average_delivery_time,
            self._handle_region_with_highest_delivery_delay,
            self._handle_products_with_pending_orders,
            self._handle_low_stock_products,
            self._handle_reorder_questions,
            self._handle_forecast_next_7_days_revenue_from_context,
            self._handle_forecast_direction_from_context,
            self._handle_top_risks_from_context,
            self._handle_management_steps_from_context,
            self._handle_business_summary,
        ]

        for handler in handlers:
            try:
                result = handler(q, working_df, analysis_context or {})
                if result and result.matched:
                    return {
                        "answer": result.answer,
                        "context_used": result.context_used,
                        "answer_source": result.answer_source,
                    }
            except Exception:
                continue

        return None

    # =========================================================
    # Data preparation
    # =========================================================
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        temp = df.copy()

        temp = temp.loc[:, ~temp.columns.duplicated()]
        temp.columns = [str(col).strip() for col in temp.columns]

        rename_map: Dict[str, str] = {}

        for col in temp.columns:
            normalized = self._normalize_column_name(col)

            if normalized in {
                "invoice_date", "date", "order_date", "sales_date",
                "transaction_date", "po_date"
            }:
                rename_map[col] = "Invoice_Date"
            elif normalized in {"delivery_date"}:
                rename_map[col] = "Delivery_Date"
            elif normalized in {"shipment_date", "dispatch_date"}:
                rename_map[col] = "Shipment_Date"
            elif normalized in {"product", "product_name", "item", "sku", "combination"}:
                rename_map[col] = "Product"
            elif normalized in {"region", "city", "location"}:
                rename_map[col] = "Region"
            elif normalized in {"district"}:
                rename_map[col] = "District"
            elif normalized in {"place"}:
                rename_map[col] = "Place"
            elif normalized in {"tehsil"}:
                rename_map[col] = "Tehsil"
            elif normalized in {
                "revenue", "sales", "amount", "total_amount", "net_total",
                "value", "total_price"
            }:
                rename_map[col] = "Revenue"
            elif normalized in {"quantity", "qty", "units", "sales_qty", "quamtity"}:
                rename_map[col] = "Quantity"
            elif normalized in {"order_id", "order", "invoice_id", "sales_order", "sale_order_number"}:
                rename_map[col] = "Order_ID"
            elif normalized in {"status", "order_status", "delivery_status"}:
                rename_map[col] = "Status"
            elif normalized in {"stock", "inventory", "inventory_level"}:
                rename_map[col] = "Stock"
            elif normalized in {"reorder_level", "reorder_stock", "reorder"}:
                rename_map[col] = "Reorder_Level"

        temp = temp.rename(columns=rename_map)

        for col in ["Invoice_Date", "Delivery_Date", "Shipment_Date"]:
            if col in temp.columns:
                temp[col] = pd.to_datetime(temp[col], errors="coerce", dayfirst=True)

        for col in ["Revenue", "Quantity", "Stock", "Reorder_Level"]:
            if col in temp.columns:
                temp[col] = pd.to_numeric(temp[col], errors="coerce")

        for col in [
            "Status", "Region", "Product", "Order_ID",
            "District", "Place", "Tehsil"
        ]:
            if col in temp.columns:
                temp[col] = temp[col].astype(str).str.strip()

        if "Invoice_Date" in temp.columns and "Delivery_Date" in temp.columns:
            temp["Delivery_Days"] = (temp["Delivery_Date"] - temp["Invoice_Date"]).dt.days

        if "Revenue" in temp.columns and "Quantity" in temp.columns:
            qty = temp["Quantity"].replace(0, pd.NA)
            temp["Revenue_Per_Unit"] = temp["Revenue"] / qty

        return temp

    # =========================================================
    # Helpers
    # =========================================================
    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text).strip().lower())

    def _normalize_column_name(self, col: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip("_")

    def _format_number(self, value: Any) -> str:
        try:
            num = float(value)
            if pd.isna(num):
                return "0"
            if num.is_integer():
                return f"{int(num):,}"
            return f"{num:,.2f}"
        except Exception:
            return str(value)

    def _safe_share_percent(self, part: float, total: float) -> float:
        if total in [0, None] or pd.isna(total):
            return 0.0
        return (float(part) / float(total)) * 100.0

    def _get_revenue_col(self, df: pd.DataFrame) -> Optional[str]:
        return "Revenue" if "Revenue" in df.columns else None

    def _get_quantity_col(self, df: pd.DataFrame) -> Optional[str]:
        return "Quantity" if "Quantity" in df.columns else None

    def _get_product_col(self, df: pd.DataFrame) -> Optional[str]:
        return "Product" if "Product" in df.columns else None

    def _get_region_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["Region", "District", "Place", "Tehsil"]:
            if col in df.columns:
                return col
        return None

    def _get_date_col(self, df: pd.DataFrame) -> Optional[str]:
        return "Invoice_Date" if "Invoice_Date" in df.columns else None

    def _extract_region(self, q: str, df: pd.DataFrame) -> Optional[str]:
        region_col = self._get_region_col(df)
        if not region_col:
            return None

        values = {
            str(v).strip().lower(): str(v).strip()
            for v in df[region_col].dropna().unique()
            if str(v).strip() and str(v).strip().lower() != "nan"
        }

        for region_lower, region_actual in sorted(values.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(rf"\b{re.escape(region_lower)}\b", q):
                return region_actual
            if region_lower in q:
                return region_actual
        return None

    def _extract_product(self, q: str, df: pd.DataFrame) -> Optional[str]:
        product_col = self._get_product_col(df)
        if not product_col:
            return None

        products = sorted(
            [str(v).strip() for v in df[product_col].dropna().unique() if str(v).strip() and str(v).strip().lower() != "nan"],
            key=len,
            reverse=True,
        )

        for product in products:
            product_lower = product.lower()
            if re.search(rf"\b{re.escape(product_lower)}\b", q):
                return product
            if product_lower in q:
                return product
        return None

    def _extract_status(self, q: str) -> Optional[str]:
        for alias, canonical in sorted(self.status_aliases.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return canonical
        return None

    def _scope_text(self, q: str, df: pd.DataFrame) -> str:
        region = self._extract_region(q, df)
        status = self._extract_status(q)
        product = self._extract_product(q, df)

        parts = []
        if product:
            parts.append(product)
        if status:
            parts.append(status)
        if region:
            parts.append(region)

        return " for " + " | ".join(parts) if parts else ""

    def _apply_date_range_filter(self, df: pd.DataFrame, q: str) -> pd.DataFrame:
        date_col = self._get_date_col(df)
        if not date_col:
            return df

        temp = df.copy()
        temp = temp[temp[date_col].notna()]

        if temp.empty:
            return temp

        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp[temp[date_col].notna()]

        if temp.empty:
            return temp

        max_date = temp[date_col].max()

        if "last 7 days" in q:
            return temp[temp[date_col] >= max_date - pd.Timedelta(days=7)]

        if "last 10 days" in q:
            return temp[temp[date_col] >= max_date - pd.Timedelta(days=10)]

        if "this week" in q:
            return temp[temp[date_col] >= max_date - pd.Timedelta(days=7)]

        if "last 30 days" in q:
            return temp[temp[date_col] >= max_date - pd.Timedelta(days=30)]

        return temp

    def _apply_common_filters(self, df: pd.DataFrame, q: str) -> pd.DataFrame:
        filtered = df.copy()

        region_col = self._get_region_col(filtered)
        region = self._extract_region(q, filtered)
        if region and region_col:
            filtered = filtered[
                filtered[region_col].astype(str).str.strip().str.lower() == region.strip().lower()
            ]

        status = self._extract_status(q)
        if status and "Status" in filtered.columns:
            filtered = filtered[
                filtered["Status"].astype(str).str.strip().str.lower() == status.strip().lower()
            ]

        product_col = self._get_product_col(filtered)
        product = self._extract_product(q, filtered)
        if product and product_col:
            filtered = filtered[
                filtered[product_col].astype(str).str.strip().str.lower() == product.strip().lower()
            ]

        filtered = self._apply_date_range_filter(filtered, q)
        return filtered

    # =========================================================
    # Core answers
    # =========================================================
    def _handle_total_revenue(self, q, df, analysis_context):
        if "revenue" not in q or not any(term in q for term in ["total", "overall", "sum", "generated"]):
            return None
        revenue_col = self._get_revenue_col(df)
        if not revenue_col:
            return None

        filtered = self._apply_common_filters(df, q)
        if filtered.empty:
            filtered = df.copy()

        total_revenue = filtered[revenue_col].dropna().sum()
        return ConditionalQueryResult(True, f"Total revenue{self._scope_text(q, filtered)} is ₹{self._format_number(total_revenue)}.", ["computed_metrics", "revenue"])

    def _handle_total_quantity(self, q, df, analysis_context):
        if "quantity" not in q or not any(term in q for term in ["total", "overall", "sum"]):
            return None
        quantity_col = self._get_quantity_col(df)
        if not quantity_col:
            return None

        filtered = self._apply_common_filters(df, q)
        if filtered.empty:
            filtered = df.copy()

        total_quantity = filtered[quantity_col].dropna().sum()
        return ConditionalQueryResult(True, f"Total quantity{self._scope_text(q, filtered)} is {self._format_number(total_quantity)} units.", ["computed_metrics", "quantity"])

    def _handle_total_revenue_last_n_days(self, q, df, analysis_context):
        if not any(term in q for term in ["last 7 days", "last 10 days", "last 30 days", "this week"]):
            return None
        if "revenue" not in q:
            return None
        revenue_col = self._get_revenue_col(df)
        if not revenue_col:
            return None

        filtered = self._apply_common_filters(df, q)
        if filtered.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "revenue", "date_range"])

        total_revenue = filtered[revenue_col].dropna().sum()
        return ConditionalQueryResult(True, f"Total revenue in the selected recent period is ₹{self._format_number(total_revenue)}.", ["computed_metrics", "revenue", "date_range"])

    def _handle_average_revenue_per_order(self, q, df, analysis_context):
        if not any(term in q for term in ["average revenue", "average order value", "avg revenue", "avg order value"]):
            return None
        revenue_col = self._get_revenue_col(df)
        if not revenue_col:
            return None

        filtered = self._apply_common_filters(df, q)
        if filtered.empty:
            filtered = df.copy()

        if filtered.empty or filtered[revenue_col].dropna().empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "revenue"])

        avg = filtered[revenue_col].mean()
        return ConditionalQueryResult(True, f"Average revenue per order is ₹{self._format_number(avg)}.", ["computed_metrics", "revenue"])

    def _handle_revenue_trend_summary(self, q, df, analysis_context):
        if not any(term in q for term in ["revenue trend over time", "revenue trend", "trend over time"]):
            return None

        sales = ((analysis_context or {}).get("insights") or {}).get("sales", {}) or {}
        summary = sales.get("revenue_trend_summary")
        if isinstance(summary, str) and summary.strip():
            return ConditionalQueryResult(True, summary.strip(), ["insights", "revenue", "trend"], answer_source="Structured Insights")

        revenue_col = self._get_revenue_col(df)
        date_col = self._get_date_col(df)
        if not revenue_col or not date_col:
            return None

        temp = df[[date_col, revenue_col]].copy()
        temp = temp.dropna(subset=[date_col, revenue_col])
        if temp.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "revenue", "trend"])

        grouped = temp.groupby(temp[date_col].dt.date)[revenue_col].sum().sort_index()
        if len(grouped) < 2:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "revenue", "trend"])

        first_val = float(grouped.iloc[0])
        last_val = float(grouped.iloc[-1])
        direction = "stable"
        if last_val > first_val:
            direction = "increasing"
        elif last_val < first_val:
            direction = "decreasing"

        return ConditionalQueryResult(True, f"The revenue trend over time appears {direction}.", ["computed_metrics", "revenue", "trend"])

    def _handle_revenue_direction(self, q, df, analysis_context):
        if not any(term in q for term in ["revenue increasing or decreasing", "is my revenue increasing", "is my revenue decreasing"]):
            return None

        sales = ((analysis_context or {}).get("insights") or {}).get("sales", {}) or {}
        raw = sales.get("revenue_trend_summary_raw", {}) or {}
        label = raw.get("trend_label")
        if label:
            return ConditionalQueryResult(True, f"Revenue is {label}.", ["insights", "revenue", "trend"], answer_source="Structured Insights")

        return self._handle_revenue_trend_summary("revenue trend over time", df, analysis_context)

    def _handle_highest_revenue_day(self, q, df, analysis_context):
        if not any(term in q for term in ["highest revenue day", "which day had highest revenue", "peak revenue day"]):
            return None

        revenue_col = self._get_revenue_col(df)
        date_col = self._get_date_col(df)
        if not revenue_col or not date_col:
            return None

        filtered = self._apply_common_filters(df, q)
        filtered = filtered.dropna(subset=[date_col, revenue_col])
        if filtered.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "revenue", "date"])

        grouped = filtered.groupby(filtered[date_col].dt.date)[revenue_col].sum().sort_values(ascending=False)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "revenue", "date"])

        return ConditionalQueryResult(True, f"The day with highest revenue is {grouped.index[0]}, with revenue ₹{self._format_number(grouped.iloc[0])}.", ["computed_metrics", "revenue", "date"])

    def _handle_lowest_revenue_day(self, q, df, analysis_context):
        if not any(term in q for term in ["lowest revenue day", "which day had lowest revenue"]):
            return None

        revenue_col = self._get_revenue_col(df)
        date_col = self._get_date_col(df)
        if not revenue_col or not date_col:
            return None

        filtered = self._apply_common_filters(df, q)
        filtered = filtered.dropna(subset=[date_col, revenue_col])
        if filtered.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "revenue", "date"])

        grouped = filtered.groupby(filtered[date_col].dt.date)[revenue_col].sum().sort_values(ascending=True)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "revenue", "date"])

        return ConditionalQueryResult(True, f"The day with lowest revenue is {grouped.index[0]}, with revenue ₹{self._format_number(grouped.iloc[0])}.", ["computed_metrics", "revenue", "date"])

    def _handle_top_product(self, q, df, analysis_context):
        if not any(term in q for term in ["top product", "best product", "strongest product", "highest revenue product"]):
            return None
        revenue_col = self._get_revenue_col(df)
        product_col = self._get_product_col(df)
        if not revenue_col or not product_col:
            return None

        grouped = self._apply_common_filters(df, q).groupby(product_col)[revenue_col].sum().sort_values(ascending=False)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "product", "revenue"])

        return ConditionalQueryResult(True, f"The top product is {grouped.index[0]}, with revenue ₹{self._format_number(grouped.iloc[0])}.", ["computed_metrics", "product", "revenue"])

    def _handle_top_3_products(self, q, df, analysis_context):
        if not any(term in q for term in ["top 3 products", "top three products", "which are the top 3 products"]):
            return None
        revenue_col = self._get_revenue_col(df)
        product_col = self._get_product_col(df)
        if not revenue_col or not product_col:
            return None

        grouped = self._apply_common_filters(df, q).groupby(product_col)[revenue_col].sum().sort_values(ascending=False)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "product", "revenue"])

        items = [f"{idx}: ₹{self._format_number(val)}" for idx, val in grouped.head(3).items()]
        return ConditionalQueryResult(True, "Top 3 products by revenue are " + "; ".join(items) + ".", ["computed_metrics", "product", "revenue"])

    def _handle_top_region(self, q, df, analysis_context):
        if not any(term in q for term in ["top region", "best region", "strongest region", "highest revenue region"]):
            return None
        revenue_col = self._get_revenue_col(df)
        region_col = self._get_region_col(df)
        if not revenue_col or not region_col:
            return None

        grouped = self._apply_common_filters(df, q).groupby(region_col)[revenue_col].sum().sort_values(ascending=False)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "region", "revenue"])

        return ConditionalQueryResult(True, f"The top region is {grouped.index[0]}, with revenue ₹{self._format_number(grouped.iloc[0])}.", ["computed_metrics", "region", "revenue"])

    def _handle_top_3_regions(self, q, df, analysis_context):
        if not any(term in q for term in ["top 3 regions", "top three regions"]):
            return None
        revenue_col = self._get_revenue_col(df)
        region_col = self._get_region_col(df)
        if not revenue_col or not region_col:
            return None

        grouped = self._apply_common_filters(df, q).groupby(region_col)[revenue_col].sum().sort_values(ascending=False)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "region", "revenue"])

        items = [f"{idx}: ₹{self._format_number(val)}" for idx, val in grouped.head(3).items()]
        return ConditionalQueryResult(True, "Top 3 regions by revenue are " + "; ".join(items) + ".", ["computed_metrics", "region", "revenue"])

    def _handle_average_value_per_product(self, q, df, analysis_context):
        if not any(term in q for term in ["average value per product", "average revenue per product", "avg revenue per product"]):
            return None
        revenue_col = self._get_revenue_col(df)
        product_col = self._get_product_col(df)
        if not revenue_col or not product_col:
            return None

        grouped = self._apply_common_filters(df, q).groupby(product_col)[revenue_col].sum()
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "product", "revenue"])

        avg_value = grouped.mean()
        return ConditionalQueryResult(True, f"Average visible revenue per product group is ₹{self._format_number(avg_value)}.", ["computed_metrics", "product", "revenue"])

    def _handle_average_value_per_region(self, q, df, analysis_context):
        if not any(term in q for term in ["average value per region", "average revenue per region"]):
            return None
        revenue_col = self._get_revenue_col(df)
        region_col = self._get_region_col(df)
        if not revenue_col or not region_col:
            return None

        grouped = self._apply_common_filters(df, q).groupby(region_col)[revenue_col].sum()
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "region", "revenue"])

        avg_value = grouped.mean()
        return ConditionalQueryResult(True, f"Average visible revenue per region is ₹{self._format_number(avg_value)}.", ["computed_metrics", "region", "revenue"])

    def _handle_product_revenue_in_region(self, q, df, analysis_context):
        if "revenue" not in q:
            return None
        region = self._extract_region(q, df)
        product = self._extract_product(q, df)
        if not region or not product:
            return None
        revenue_col = self._get_revenue_col(df)
        if not revenue_col:
            return None

        filtered = self._apply_common_filters(df, q)
        if filtered.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "product", "region", "revenue"])

        total = filtered[revenue_col].sum()
        return ConditionalQueryResult(True, f"Revenue for product {product} in {region} is ₹{self._format_number(total)}.", ["computed_metrics", "product", "region", "revenue"])

    def _handle_revenue_by_region(self, q, df, analysis_context):
        if not any(term in q for term in ["revenue by region", "sales by region", "group revenue by region"]):
            return None
        revenue_col = self._get_revenue_col(df)
        region_col = self._get_region_col(df)
        if not revenue_col or not region_col:
            return None

        grouped = self._apply_common_filters(df, q).groupby(region_col)[revenue_col].sum().sort_values(ascending=False)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "region", "revenue"])

        items = [f"{idx}: ₹{self._format_number(val)}" for idx, val in grouped.head(10).items()]
        return ConditionalQueryResult(True, "Revenue by region: " + "; ".join(items) + ".", ["computed_metrics", "region", "revenue"])

    def _handle_revenue_by_product(self, q, df, analysis_context):
        if not any(term in q for term in ["revenue by product", "sales by product", "group revenue by product"]):
            return None
        revenue_col = self._get_revenue_col(df)
        product_col = self._get_product_col(df)
        if not revenue_col or not product_col:
            return None

        grouped = self._apply_common_filters(df, q).groupby(product_col)[revenue_col].sum().sort_values(ascending=False)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "product", "revenue"])

        items = [f"{idx}: ₹{self._format_number(val)}" for idx, val in grouped.head(10).items()]
        return ConditionalQueryResult(True, "Revenue by product: " + "; ".join(items) + ".", ["computed_metrics", "product", "revenue"])

    def _handle_not_delivered_orders(self, q, df, analysis_context):
        if not any(term in q for term in ["not delivered", "orders not delivered", "pending delivery"]):
            return None

        if "Delivery_Date" in df.columns:
            filtered = self._apply_common_filters(df, q)
            count = filtered["Delivery_Date"].isna().sum()
            return ConditionalQueryResult(True, f"There are {self._format_number(count)} orders not delivered yet.", ["computed_metrics", "delivery"])

        if "Status" in df.columns:
            filtered = self._apply_common_filters(df, q)
            count = filtered["Status"].astype(str).str.lower().isin(["pending", "not delivered"]).sum()
            return ConditionalQueryResult(True, f"There are {self._format_number(count)} orders not delivered yet.", ["computed_metrics", "status", "delivery"])

        return None

    def _handle_pending_orders_count(self, q, df, analysis_context):
        if not any(term in q for term in ["pending orders", "how many pending", "count pending"]):
            return None
        if "Status" not in df.columns:
            return None

        filtered = self._apply_common_filters(df, q)
        count = filtered["Status"].astype(str).str.lower().isin(["pending", "not delivered"]).sum()
        return ConditionalQueryResult(True, f"There are {self._format_number(count)} pending orders{self._scope_text(q, filtered)}.", ["computed_metrics", "status"])

    def _handle_delivered_orders_count(self, q, df, analysis_context):
        if not any(term in q for term in ["delivered orders", "how many delivered", "count delivered"]):
            return None
        if "Status" not in df.columns:
            return None

        filtered = self._apply_common_filters(df, q)
        count = filtered["Status"].astype(str).str.lower().eq("delivered").sum()
        return ConditionalQueryResult(True, f"There are {self._format_number(count)} delivered orders{self._scope_text(q, filtered)}.", ["computed_metrics", "status"])

    def _handle_status_counts(self, q, df, analysis_context):
        if not any(term in q for term in ["status wise", "status breakdown", "order status breakdown"]):
            return None
        if "Status" not in df.columns:
            return None

        filtered = self._apply_common_filters(df, q)
        counts = filtered["Status"].fillna("Unknown").value_counts()
        if counts.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "status"])

        parts = [f"{status}: {count}" for status, count in counts.items()]
        return ConditionalQueryResult(True, "Order status breakdown is " + ", ".join(parts) + ".", ["computed_metrics", "status"])

    def _handle_average_delivery_time(self, q, df, analysis_context):
        if not any(term in q for term in ["average delivery time", "avg delivery time"]):
            return None
        if "Delivery_Days" not in df.columns:
            return None

        filtered = self._apply_common_filters(df, q)
        filtered = filtered[filtered["Delivery_Days"].notna()]
        if filtered.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "delivery"])

        avg_days = filtered["Delivery_Days"].mean()
        return ConditionalQueryResult(True, f"Average delivery time{self._scope_text(q, filtered)} is {self._format_number(avg_days)} day(s).", ["computed_metrics", "delivery"])

    def _handle_region_with_highest_delivery_delay(self, q, df, analysis_context):
        if not any(term in q for term in ["which region has highest delivery delay", "maximum delivery delay region", "region with highest delivery delay"]):
            return None
        region_col = self._get_region_col(df)
        if not region_col or "Delivery_Days" not in df.columns:
            return None

        filtered = self._apply_common_filters(df, q)
        filtered = filtered[filtered["Delivery_Days"].notna()]
        grouped = filtered.groupby(region_col)["Delivery_Days"].mean().sort_values(ascending=False)
        if grouped.empty:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "delivery", "region"])

        return ConditionalQueryResult(True, f"The region with the highest delivery delay is {grouped.index[0]}, with average delay of {self._format_number(grouped.iloc[0])} day(s).", ["computed_metrics", "delivery", "region"])

    def _handle_products_with_pending_orders(self, q, df, analysis_context):
        if not any(term in q for term in ["products with pending orders", "which products are pending", "products pending"]):
            return None
        product_col = self._get_product_col(df)
        if not product_col:
            return None

        if "Status" in df.columns:
            pending_df = self._apply_common_filters(df, q)
            pending_df = pending_df[pending_df["Status"].astype(str).str.strip().str.lower().isin(["pending", "not delivered"])]
        elif "Delivery_Date" in df.columns:
            pending_df = self._apply_common_filters(df, q)
            pending_df = pending_df[pending_df["Delivery_Date"].isna()]
        else:
            return None

        if pending_df.empty:
            return ConditionalQueryResult(True, "No products currently have pending orders in the filtered data.", ["computed_metrics", "product", "status"])

        grouped = pending_df.groupby(product_col).size().sort_values(ascending=False)
        items = [f"{product} ({self._format_number(count)} pending records)" for product, count in grouped.head(10).items()]
        return ConditionalQueryResult(True, "Products with pending orders are " + "; ".join(items) + ".", ["computed_metrics", "product", "status"])

    def _handle_low_stock_products(self, q, df, analysis_context):
        if not any(term in q for term in ["low stock products", "which products are low stock", "stock out risk products"]):
            return None
        product_col = self._get_product_col(df)
        if not product_col or "Stock" not in df.columns:
            return None

        filtered = self._apply_common_filters(df, q)
        if "Reorder_Level" in filtered.columns:
            low_stock_df = filtered[filtered["Stock"] <= filtered["Reorder_Level"]]
        else:
            threshold = filtered["Stock"].median()
            low_stock_df = filtered[filtered["Stock"] <= threshold]

        if low_stock_df.empty:
            return ConditionalQueryResult(True, "No low-stock products stand out in the current filtered data.", ["computed_metrics", "inventory", "product"])

        grouped = low_stock_df.groupby(product_col)["Stock"].mean().sort_values(ascending=True)
        items = [f"{idx} (avg stock={self._format_number(val)})" for idx, val in grouped.head(10).items()]
        return ConditionalQueryResult(True, "Low-stock products are " + "; ".join(items) + ".", ["computed_metrics", "inventory", "product"])

    def _handle_reorder_questions(self, q, df, analysis_context):
        if not ("reorder" in q or "restock" in q):
            return None
        product_col = self._get_product_col(df)
        if not product_col or "Stock" not in df.columns:
            return None

        filtered = self._apply_common_filters(df, q)
        if "Reorder_Level" in filtered.columns:
            candidate_df = filtered[filtered["Stock"] < filtered["Reorder_Level"]]
        else:
            stock_threshold = filtered["Stock"].median() if not filtered["Stock"].dropna().empty else None
            if stock_threshold is None:
                return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["computed_metrics", "inventory", "product"])
            candidate_df = filtered[filtered["Stock"] <= stock_threshold]

        if candidate_df.empty:
            return ConditionalQueryResult(True, "No products currently appear to need urgent reordering in the filtered data.", ["computed_metrics", "inventory", "product"])

        grouped = candidate_df.groupby(product_col).agg(avg_stock=("Stock", "mean")).sort_values("avg_stock", ascending=True)
        items = [f"{idx} (avg stock={self._format_number(row['avg_stock'])})" for idx, row in grouped.head(5).iterrows()]
        return ConditionalQueryResult(True, "Products that may need reordering are " + "; ".join(items) + ".", ["computed_metrics", "inventory", "product"])

    # =========================================================
    # Forecast / risk / management from context
    # =========================================================
    def _handle_forecast_next_7_days_revenue_from_context(self, q, df, analysis_context):
        if not any(term in q for term in ["expected revenue in next 7 days", "next 7 days revenue", "forecast revenue next 7 days"]):
            return None

        forecast = analysis_context.get("forecast", {}) or {}
        future_forecast = forecast.get("future_forecast", []) or forecast.get("forecast_values", []) or []

        if not future_forecast:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["forecast"])

        total_pred = 0.0
        count = 0
        for item in future_forecast[:7]:
            val = item.get("predicted") or item.get("predicted_value") or item.get("yhat") or item.get("value")
            try:
                total_pred += float(val or 0)
                count += 1
            except Exception:
                continue

        if count == 0:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["forecast"])

        reliability = forecast.get("reliability_label")
        caution = f" Forecast reliability is {reliability}." if reliability else ""
        return ConditionalQueryResult(True, f"Projected revenue for the next 7 days is ₹{self._format_number(total_pred)}.{caution}", ["forecast"])

    def _handle_forecast_direction_from_context(self, q, df, analysis_context):
        if not any(term in q for term in ["demand expected to increase", "demand expected to decrease", "forecast trend", "future demand"]):
            return None

        forecast = analysis_context.get("forecast", {}) or {}
        trend_direction = str(forecast.get("trend_direction") or "").lower()
        growth_percent = forecast.get("growth_percent")

        if trend_direction == "increasing":
            return ConditionalQueryResult(True, f"Demand is expected to increase based on available forecast signals, with growth signal around {self._format_number(growth_percent)}%." if growth_percent is not None else "Demand is expected to increase based on available forecast signals.", ["forecast"])
        if trend_direction == "decreasing":
            return ConditionalQueryResult(True, f"Demand is expected to decrease based on available forecast signals, with growth signal around {self._format_number(growth_percent)}%." if growth_percent is not None else "Demand is expected to decrease based on available forecast signals.", ["forecast"])
        if trend_direction == "stable":
            return ConditionalQueryResult(True, "Demand appears stable based on available forecast signals.", ["forecast"])

        summary = forecast.get("summary")
        if isinstance(summary, str) and summary.strip():
            return ConditionalQueryResult(True, summary.strip(), ["forecast"], answer_source="Structured Insights")

        return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["forecast"])

    def _handle_top_risks_from_context(self, q, df, analysis_context):
        if not any(term in q for term in ["top risks", "what are the top risks", "business risks"]):
            return None

        sales = ((analysis_context or {}).get("insights") or {}).get("sales", {}) or {}
        logistics = ((analysis_context or {}).get("insights") or {}).get("logistics", {}) or {}
        inventory = ((analysis_context or {}).get("insights") or {}).get("inventory", {}) or {}
        forecast = analysis_context.get("forecast", {}) or {}

        risks = []

        product_conc = sales.get("product_concentration_summary")
        if isinstance(product_conc, str) and product_conc.strip():
            risks.append(product_conc.strip())

        weakest_region = sales.get("weakest_region_summary")
        if isinstance(weakest_region, str) and weakest_region.strip():
            risks.append(weakest_region.strip())

        inv_summary = inventory.get("inventory_summary")
        if isinstance(inv_summary, dict):
            txt = inv_summary.get("inventory_summary")
            if txt:
                risks.append(str(txt))
        elif isinstance(inv_summary, str) and inv_summary.strip():
            risks.append(inv_summary.strip())

        delay_risk = logistics.get("delay_risk")
        if delay_risk:
            risks.append(f"Logistics delay risk is currently {delay_risk}.")

        reliability = forecast.get("reliability_label")
        if reliability and str(reliability).lower() in {"low", "medium", "not reliable"}:
            risks.append(f"Forecast reliability is {reliability}, so forward-looking decisions should be treated with caution.")

        if not risks:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["insights", "forecast", "risk"])

        return ConditionalQueryResult(True, "Top risks are: " + " ".join(risks[:4]), ["insights", "forecast", "risk"], answer_source="Structured Insights")

    def _handle_management_steps_from_context(self, q, df, analysis_context):
        if not any(term in q for term in ["what steps should the management take", "what should management do", "management steps", "next steps"]):
            return None

        decisions = analysis_context.get("decisions", {})
        decision_items = []

        if isinstance(decisions, dict):
            decision_items = decisions.get("top_decisions", []) or []
        elif isinstance(decisions, list):
            decision_items = decisions

        actions = []
        for item in decision_items[:3]:
            if isinstance(item, dict):
                action = item.get("decision") or item.get("action") or item.get("title") or item.get("recommendation")
                if action:
                    actions.append(str(action))

        if actions:
            return ConditionalQueryResult(True, "Management should focus on: " + "; ".join(actions) + ".", ["decisions"], answer_source="Structured Insights")

        return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["decisions"])

    def _handle_business_summary(self, q, df, analysis_context):
        if not any(term in q for term in ["business summary", "overall summary", "executive summary"]):
            return None

        revenue_col = self._get_revenue_col(df)
        product_col = self._get_product_col(df)
        region_col = self._get_region_col(df)

        if not revenue_col:
            return ConditionalQueryResult(True, MISSING_INFO_MESSAGE, ["summary"])

        total_revenue = df[revenue_col].dropna().sum()
        avg_order_value = df[revenue_col].dropna().mean()

        top_product_text = "N/A"
        if product_col:
            grouped = df.groupby(product_col)[revenue_col].sum().sort_values(ascending=False)
            if not grouped.empty:
                top_product_text = f"{grouped.index[0]} (₹{self._format_number(grouped.iloc[0])})"

        top_region_text = "N/A"
        if region_col:
            grouped = df.groupby(region_col)[revenue_col].sum().sort_values(ascending=False)
            if not grouped.empty:
                top_region_text = f"{grouped.index[0]} (₹{self._format_number(grouped.iloc[0])})"

        return ConditionalQueryResult(
            True,
            f"Business summary: total revenue is ₹{self._format_number(total_revenue)}, average order value is ₹{self._format_number(avg_order_value)}, top product is {top_product_text}, and top region is {top_region_text}.",
            ["summary", "revenue", "product", "region"],
        )