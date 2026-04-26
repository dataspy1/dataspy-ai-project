from typing import Dict, Any
import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    df_copy = df.copy()

    total_rows = int(df_copy.shape[0])
    total_columns = int(df_copy.shape[1])
    duplicate_rows = int(df_copy.duplicated().sum())

    null_counts = df_copy.isnull().sum().to_dict()
    null_percentages = ((df_copy.isnull().sum() / len(df_copy)) * 100).fillna(0).round(2).to_dict()

    numeric_columns = df_copy.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_candidate_columns = []

    for col in df_copy.columns:
        try:
            converted = pd.to_datetime(df_copy[col], errors="coerce")
            valid_ratio = converted.notna().mean()
            if valid_ratio >= 0.6:
                datetime_candidate_columns.append(col)
        except Exception:
            pass

    column_profiles = {}

    for col in df_copy.columns:
        col_data = df_copy[col]

        profile = {
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "null_percentage": float(round((col_data.isnull().mean() * 100), 2)),
            "unique_count": int(col_data.nunique(dropna=True))
        }

        if col in numeric_columns:
            profile["stats"] = {
                "min": None if pd.isna(col_data.min()) else float(col_data.min()),
                "max": None if pd.isna(col_data.max()) else float(col_data.max()),
                "mean": None if pd.isna(col_data.mean()) else float(round(col_data.mean(), 2)),
                "median": None if pd.isna(col_data.median()) else float(round(col_data.median(), 2))
            }
        else:
            top_values = col_data.astype(str).value_counts(dropna=True).head(5).to_dict()
            profile["top_values"] = top_values

        column_profiles[col] = profile

    return {
        "dataset_profile": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "duplicate_rows": duplicate_rows,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "datetime_candidate_columns": datetime_candidate_columns,
            "null_counts": {k: int(v) for k, v in null_counts.items()},
            "null_percentages": {k: float(v) for k, v in null_percentages.items()}
        },
        "column_profiles": column_profiles
    }