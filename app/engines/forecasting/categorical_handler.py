from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


EXCLUDED_ROLE_NAMES = {
    "order_id",
    "customer_id",
    "invoice_id",
    "transaction_id",
    "email",
    "phone",
    "mobile",
    "name",
    "address",
    "pincode",
    "zip_code",
    "zip",
}


@dataclass
class CategoricalHandlingResult:
    encoded_df: pd.DataFrame
    used_categorical_columns: List[str]
    dropped_categorical_columns: List[str]
    encoding_summary: Dict[str, str]


class CategoricalHandler:
    def __init__(
        self,
        low_cardinality_threshold: int = 10,
        high_cardinality_ratio: float = 0.40,
    ) -> None:
        self.low_cardinality_threshold = low_cardinality_threshold
        self.high_cardinality_ratio = high_cardinality_ratio

    def select_categorical_columns(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        schema_mapping: Dict[str, str] | None = None,
    ) -> Tuple[List[str], List[str]]:
        schema_mapping = schema_mapping or {}
        selected: List[str] = []
        dropped: List[str] = []

        for col in df.columns:
            if col == target_col or col == date_col:
                continue

            if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
                continue

            col_lower = col.strip().lower()

            if col_lower in EXCLUDED_ROLE_NAMES:
                dropped.append(col)
                continue

            if "id" in col_lower and df[col].nunique(dropna=True) > len(df) * 0.5:
                dropped.append(col)
                continue

            nunique = df[col].nunique(dropna=True)
            if nunique <= 1:
                dropped.append(col)
                continue

            uniqueness_ratio = nunique / max(len(df), 1)
            if uniqueness_ratio > 0.85:
                dropped.append(col)
                continue

            selected.append(col)

        return selected, dropped

    def encode_categorical_columns(
        self,
        df: pd.DataFrame,
        categorical_columns: List[str],
    ) -> CategoricalHandlingResult:
        encoded_df = df.copy()
        used_columns: List[str] = []
        dropped_columns: List[str] = []
        encoding_summary: Dict[str, str] = {}

        for col in categorical_columns:
            if col not in encoded_df.columns:
                continue

            series = encoded_df[col].fillna("Unknown").astype(str).str.strip()
            nunique = series.nunique(dropna=True)
            uniqueness_ratio = nunique / max(len(encoded_df), 1)

            if nunique <= self.low_cardinality_threshold:
                dummies = pd.get_dummies(series, prefix=col, dummy_na=False)
                encoded_df = pd.concat([encoded_df.drop(columns=[col]), dummies], axis=1)
                used_columns.append(col)
                encoding_summary[col] = "one_hot"
                continue

            if uniqueness_ratio <= self.high_cardinality_ratio:
                freq_map = series.value_counts(normalize=True).to_dict()
                encoded_df[f"{col}_freq"] = series.map(freq_map).fillna(0.0)
                encoded_df = encoded_df.drop(columns=[col])
                used_columns.append(col)
                encoding_summary[col] = "frequency_encoding"
                continue

            dropped_columns.append(col)
            encoded_df = encoded_df.drop(columns=[col])
            encoding_summary[col] = "dropped_high_cardinality"

        return CategoricalHandlingResult(
            encoded_df=encoded_df,
            used_categorical_columns=used_columns,
            dropped_categorical_columns=dropped_columns,
            encoding_summary=encoding_summary,
        )