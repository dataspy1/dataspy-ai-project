from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.engines.forecasting.categorical_handler import CategoricalHandler


@dataclass
class FeatureEngineeringResult:
    model_df: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    date_column: str
    categorical_summary: Dict[str, str]
    used_categorical_columns: List[str]
    dropped_categorical_columns: List[str]
    warnings: List[str]


class ForecastFeatureEngineer:
    def __init__(self) -> None:
        self.categorical_handler = CategoricalHandler()

    def prepare_forecasting_frame(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        schema_mapping: Optional[Dict[str, str]] = None,
    ) -> FeatureEngineeringResult:
        working_df = df.copy()
        warnings: List[str] = []

        # Global cleanup
        working_df = self._clean_general_columns(working_df)

        if date_col not in working_df.columns:
            raise ValueError(f"Date column '{date_col}' not found in dataset.")

        if target_col not in working_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        # Handle business-meaningful missing values before forecasting prep
        working_df, business_warnings = self._handle_business_missing_values(
            df=working_df,
            schema_mapping=schema_mapping,
        )
        warnings.extend(business_warnings)

        # Date and target cleaning
        working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce", dayfirst=True)
        working_df = working_df.dropna(subset=[date_col])

        working_df[target_col] = pd.to_numeric(working_df[target_col], errors="coerce")
        working_df = working_df.dropna(subset=[target_col])

        if working_df.empty:
            raise ValueError("No valid rows available after date/target cleaning.")

        selected_cats, dropped_cats = self.categorical_handler.select_categorical_columns(
            df=working_df,
            target_col=target_col,
            date_col=date_col,
            schema_mapping=schema_mapping,
        )

        grouped = self._aggregate_by_date(
            df=working_df,
            date_col=date_col,
            target_col=target_col,
            categorical_columns=selected_cats,
        )

        if len(grouped) < 15:
            warnings.append(
                "Dataset has limited historical rows after time aggregation; forecast reliability may be lower."
            )

        grouped = self._add_calendar_features(grouped, date_col)
        grouped = self._add_lag_features(grouped, target_col)
        grouped = self._add_rolling_features(grouped, target_col)
        grouped = self._add_difference_features(grouped, target_col)

        cat_result = self.categorical_handler.encode_categorical_columns(
            df=grouped,
            categorical_columns=selected_cats,
        )

        final_df = cat_result.encoded_df.dropna().reset_index(drop=True)

        if final_df.empty:
            raise ValueError("Feature engineering produced no usable rows after lag/rolling generation.")

        feature_columns = [
            col for col in final_df.columns
            if col not in {date_col, target_col}
        ]

        return FeatureEngineeringResult(
            model_df=final_df,
            feature_columns=feature_columns,
            target_column=target_col,
            date_column=date_col,
            categorical_summary=cat_result.encoding_summary,
            used_categorical_columns=cat_result.used_categorical_columns,
            dropped_categorical_columns=dropped_cats + cat_result.dropped_categorical_columns,
            warnings=warnings,
        )

    def _clean_general_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Remove useless unnamed columns
        df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False, na=False)]

        # Standardize empty-like values
        df.replace(["", " ", "NA", "N/A", "null", "None"], pd.NA, inplace=True)

        # Strip column names
        df.columns = [str(col).strip() for col in df.columns]

        return df

    def _handle_business_missing_values(
        self,
        df: pd.DataFrame,
        schema_mapping: Optional[Dict[str, str]] = None,
    ) -> tuple[pd.DataFrame, List[str]]:
        """
        Keep business-meaningful missing values untouched.
        Example:
        delivery_date is empty because the order is still pending / not delivered.

        This works generically:
        - uses schema mapping first
        - falls back to common column name variations
        """
        df = df.copy()
        warnings: List[str] = []

        normalized_map = {
            str(col).strip().lower().replace(" ", "_").replace("-", "_"): col
            for col in df.columns
        }

        def resolve_column(role_name: str, fallbacks: List[str]) -> Optional[str]:
            if schema_mapping:
                mapped = schema_mapping.get(role_name)
                if mapped and mapped in df.columns:
                    return mapped

            for key in fallbacks:
                if key in normalized_map:
                    return normalized_map[key]

            return None

        delivery_col = resolve_column(
            "delivery_date",
            ["delivery_date", "invoice_date", "delivered_date", "delivery_dt"],
        )
        status_col = resolve_column(
            "status",
            ["status", "delivery_status", "order_status", "shipment_status"],
        )
        order_col = resolve_column(
            "date",
            ["date", "order_date", "po_date", "transaction_date", "posting_date"],
        )

        if delivery_col and delivery_col in df.columns:
            df[delivery_col] = pd.to_datetime(df[delivery_col], errors="coerce", dayfirst=True)

        if order_col and order_col in df.columns:
            df[order_col] = pd.to_datetime(df[order_col], errors="coerce", dayfirst=True)

        # Case 1: delivery + status both present
        if delivery_col and status_col and status_col in df.columns:
            status_series = (
                df[status_col]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            pending_values = {
                "not delivered",
                "pending",
                "open",
                "in transit",
                "processing",
                "undelivered",
                "not_delivered",
            }

            pending_mask = status_series.isin(pending_values)

            # Keep delivery date null for pending rows
            df.loc[pending_mask & df[delivery_col].isna(), delivery_col] = pd.NaT

            # Helper feature
            df["is_delivered"] = (~(pending_mask & df[delivery_col].isna())).astype(int)

            pending_count = int((pending_mask & df[delivery_col].isna()).sum())
            if pending_count > 0:
                warnings.append(
                    f"{pending_count} rows have empty delivery date with pending/not delivered status. Kept as business-valid missing values."
                )

        # Case 2: only delivery column exists
        elif delivery_col:
            df["is_delivered"] = df[delivery_col].notna().astype(int)
            missing_delivery_count = int(df[delivery_col].isna().sum())

            if missing_delivery_count > 0:
                warnings.append(
                    f"{missing_delivery_count} rows have missing delivery date. No status column found, so values were kept empty and marked using is_delivered."
                )

        # Delivery time feature only for completed deliveries
        if delivery_col and order_col and delivery_col in df.columns and order_col in df.columns:
            df["delivery_time_days"] = (df[delivery_col] - df[order_col]).dt.days

            # Clean impossible negatives
            negative_mask = df["delivery_time_days"] < 0
            if negative_mask.any():
                negative_count = int(negative_mask.sum())
                df.loc[negative_mask, "delivery_time_days"] = np.nan
                warnings.append(
                    f"{negative_count} rows had negative delivery time. Converted to missing for safety."
                )

        return df, warnings

    def _aggregate_by_date(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        categorical_columns: List[str],
    ) -> pd.DataFrame:
        temp_df = df.copy()
        temp_df[date_col] = pd.to_datetime(temp_df[date_col]).dt.floor("D")

        agg_dict = {target_col: "sum"}

        for col in categorical_columns:
            if col in temp_df.columns:
                agg_dict[col] = self._mode_safe

        grouped = (
            temp_df.groupby(date_col, as_index=False)
            .agg(agg_dict)
            .sort_values(date_col)
            .reset_index(drop=True)
        )

        return grouped

    def _add_calendar_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        df = df.copy()
        df["year"] = df[date_col].dt.year
        df["quarter"] = df[date_col].dt.quarter
        df["month"] = df[date_col].dt.month
        df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
        df["day"] = df[date_col].dt.day
        df["dayofweek"] = df[date_col].dt.dayofweek
        df["dayofyear"] = df[date_col].dt.dayofyear
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        return df

    def _add_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        for lag in [1, 2, 3, 7, 14]:
            df[f"lag_{lag}"] = df[target_col].shift(lag)
        return df

    def _add_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        for window in [3, 7, 14]:
            df[f"rolling_mean_{window}"] = df[target_col].shift(1).rolling(window=window).mean()
        for window in [7]:
            df[f"rolling_std_{window}"] = df[target_col].shift(1).rolling(window=window).std()
            df[f"rolling_min_{window}"] = df[target_col].shift(1).rolling(window=window).min()
            df[f"rolling_max_{window}"] = df[target_col].shift(1).rolling(window=window).max()
        return df

    def _add_difference_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = df.copy()
        df["diff_1"] = df[target_col].diff(1)
        df["diff_7"] = df[target_col].diff(7)
        df["pct_change_1"] = df[target_col].pct_change(1).replace([np.inf, -np.inf], np.nan)
        df["pct_change_7"] = df[target_col].pct_change(7).replace([np.inf, -np.inf], np.nan)
        return df

    @staticmethod
    def _mode_safe(series: pd.Series):
        mode = series.mode(dropna=True)
        if not mode.empty:
            return mode.iloc[0]
        return "Unknown"