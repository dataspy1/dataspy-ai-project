from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd


EXPECTED_COLUMNS = [
    "date",
    "product",
    "region",
    "revenue",
    "quantity",
    "order_id",
    "delivery_date",
    "shipment_date",
    "status",
    "stock",
    "reorder_level",
    "lead_time",
]


def _clean_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False, na=False)]

    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Replace blank strings with NA
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Drop fully blank rows
    df = df.dropna(how="all").reset_index(drop=True)

    return df


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    normalized = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]

    return None


def _safe_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_already_standardized(df: pd.DataFrame) -> bool:
    normalized_cols = {str(col).strip().lower() for col in df.columns}
    core_required = {"date", "product", "region", "revenue"}
    return core_required.issubset(normalized_cols)


def _standardize_existing_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    If file is already in DataSpy-standard shape, preserve it instead of remapping raw client columns.
    """
    temp = _clean_base_dataframe(df).copy()
    temp.columns = [str(col).strip().lower() for col in temp.columns]

    standardized = pd.DataFrame(index=temp.index)

    standardized["date"] = _safe_date(temp["date"]) if "date" in temp.columns else pd.NaT
    standardized["product"] = temp["product"] if "product" in temp.columns else pd.NA
    standardized["region"] = temp["region"] if "region" in temp.columns else pd.NA
    standardized["revenue"] = _safe_numeric(temp["revenue"]) if "revenue" in temp.columns else pd.NA
    standardized["quantity"] = _safe_numeric(temp["quantity"]) if "quantity" in temp.columns else 1
    standardized["order_id"] = temp["order_id"] if "order_id" in temp.columns else pd.NA
    standardized["delivery_date"] = _safe_date(temp["delivery_date"]) if "delivery_date" in temp.columns else pd.NaT
    standardized["shipment_date"] = _safe_date(temp["shipment_date"]) if "shipment_date" in temp.columns else pd.NaT
    standardized["status"] = temp["status"] if "status" in temp.columns else pd.NA
    standardized["stock"] = _safe_numeric(temp["stock"]) if "stock" in temp.columns else 100
    standardized["reorder_level"] = _safe_numeric(temp["reorder_level"]) if "reorder_level" in temp.columns else 20
    standardized["lead_time"] = _safe_numeric(temp["lead_time"]) if "lead_time" in temp.columns else 2

    # Fill smart defaults only where missing
    standardized["quantity"] = standardized["quantity"].fillna(1)
    standardized["stock"] = standardized["stock"].fillna(100)
    standardized["reorder_level"] = standardized["reorder_level"].fillna(20)
    standardized["lead_time"] = standardized["lead_time"].fillna(2)

    # If shipment_date missing but delivery_date available
    shipment_missing = standardized["shipment_date"].isna() & standardized["delivery_date"].notna()
    standardized.loc[shipment_missing, "shipment_date"] = (
        standardized.loc[shipment_missing, "delivery_date"] - pd.Timedelta(days=2)
    )

    keep_mask = (
        standardized["revenue"].notna()
        | standardized["date"].notna()
        | standardized["delivery_date"].notna()
        | standardized["product"].notna()
    )
    standardized = standardized[keep_mask].reset_index(drop=True)

    for col in EXPECTED_COLUMNS:
        if col not in standardized.columns:
            standardized[col] = pd.NA

    standardized = standardized[EXPECTED_COLUMNS]

    metadata = {
        "standardization_applied": True,
        "input_format": "already_standardized",
        "source_columns_detected": {col: col for col in temp.columns if col in EXPECTED_COLUMNS},
        "filled_defaults": {
            "quantity": "filled_missing_with_1_only",
            "shipment_date_rule": "delivery_date_minus_2_days_if_missing",
            "stock": "filled_missing_with_100_only",
            "reorder_level": "filled_missing_with_20_only",
            "lead_time": "filled_missing_with_2_only",
        },
        "row_count_after_standardization": int(standardized.shape[0]),
        "columns_after_standardization": list(standardized.columns),
    }

    return {
        "dataframe": standardized,
        "metadata": metadata,
    }


def _standardize_raw_client_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert raw client dataset into standardized internal format expected by DataSpy Decision AI.
    """
    df = _clean_base_dataframe(df)

    # Detect likely client columns
    po_date_col = _find_column(df, ["PO DATE", "PO_DATE", "ORDER DATE", "ORDER_DATE"])
    invoice_date_col = _find_column(df, ["INVOICE DATE", "INVOICE_DATE", "DELIVERY DATE", "DELIVERY_DATE"])
    product_col = _find_column(df, ["COMBINATION", "PRODUCT", "PRODUCT NAME", "ITEM"])
    region_col = _find_column(df, ["DISTRICT", "REGION", "CITY", "PLACE"])
    revenue_col = _find_column(df, ["TOTAL PRICE", "TOTAL_AMOUNT", "REVENUE", "AMOUNT", "TOTAL"])
    quantity_col = _find_column(df, ["QUANTITY", "QTY", "UNITS"])
    order_id_col = _find_column(df, ["SALE ORDER NUMBER", "ORDER_ID", "ORDER NUMBER", "SALES ORDER"])
    status_col = _find_column(df, ["STATUS"])

    standardized = pd.DataFrame(index=df.index)

    # Core mappings
    standardized["date"] = _safe_date(df[po_date_col]) if po_date_col else pd.NaT
    standardized["product"] = df[product_col] if product_col else pd.NA
    standardized["region"] = df[region_col] if region_col else pd.NA
    standardized["revenue"] = _safe_numeric(df[revenue_col]) if revenue_col else pd.NA
    standardized["quantity"] = _safe_numeric(df[quantity_col]) if quantity_col else 1
    standardized["order_id"] = df[order_id_col] if order_id_col else pd.NA
    standardized["status"] = df[status_col] if status_col else pd.NA

    # Business assumptions
    standardized["delivery_date"] = _safe_date(df[invoice_date_col]) if invoice_date_col else pd.NaT
    standardized["shipment_date"] = standardized["delivery_date"] - pd.Timedelta(days=2)
    standardized["stock"] = 100
    standardized["reorder_level"] = 20
    standardized["lead_time"] = 2

    standardized["quantity"] = standardized["quantity"].fillna(1)

    # Keep meaningful rows
    keep_mask = (
        standardized["revenue"].notna()
        | standardized["date"].notna()
        | standardized["delivery_date"].notna()
        | standardized["product"].notna()
    )
    standardized = standardized[keep_mask].reset_index(drop=True)

    # Ensure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in standardized.columns:
            standardized[col] = pd.NA

    standardized = standardized[EXPECTED_COLUMNS]

    metadata = {
        "standardization_applied": True,
        "input_format": "raw_client_file",
        "source_columns_detected": {
            "po_date": po_date_col,
            "invoice_date": invoice_date_col,
            "product": product_col,
            "region": region_col,
            "revenue": revenue_col,
            "quantity": quantity_col,
            "order_id": order_id_col,
            "status": status_col,
        },
        "filled_defaults": {
            "quantity": 1 if quantity_col is None else "client_provided_or_filled_missing_with_1",
            "shipment_date_rule": "delivery_date_minus_2_days",
            "stock": 100,
            "reorder_level": 20,
            "lead_time": 2,
        },
        "row_count_after_standardization": int(standardized.shape[0]),
        "columns_after_standardization": list(standardized.columns),
    }

    return {
        "dataframe": standardized,
        "metadata": metadata,
    }


def standardize_client_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Entry point:
    - preserve already-standardized datasets
    - transform raw client datasets into standard structure
    """
    cleaned_df = _clean_base_dataframe(df)

    if _is_already_standardized(cleaned_df):
        return _standardize_existing_dataset(cleaned_df)

    return _standardize_raw_client_dataset(cleaned_df)