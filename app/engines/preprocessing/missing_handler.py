from typing import Dict, List
import pandas as pd


class MissingHandler:
    @staticmethod
    def get_strategy_suggestions(series: pd.Series) -> List[str]:
        suggestions = []
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric:
            suggestions.extend(["mean", "median", "mode", "zero_fill", "drop_rows"])
        else:
            suggestions.extend(["mode", "unknown_label", "drop_rows"])

        if pd.api.types.is_datetime64_any_dtype(series) or "date" in str(series.name).lower():
            suggestions.extend(["forward_fill", "backward_fill", "interpolate"])

        return list(dict.fromkeys(suggestions))

    @staticmethod
    def profile_missing(df: pd.DataFrame) -> List[Dict]:
        profiles = []

        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_percent = round((missing_count / len(df)) * 100, 2) if len(df) > 0 else 0.0

            profiles.append(
                {
                    "column_name": col,
                    "dtype": str(df[col].dtype),
                    "missing_count": missing_count,
                    "missing_percent": missing_percent,
                    "suggested_strategies": MissingHandler.get_strategy_suggestions(df[col]),
                }
            )

        return profiles

    @staticmethod
    def apply_strategies(df: pd.DataFrame, strategies: Dict[str, str]):
        transformed = df.copy()
        logs = []

        for col, strategy in strategies.items():
            if col not in transformed.columns:
                continue

            affected_rows = int(transformed[col].isna().sum())

            if affected_rows == 0 and strategy != "drop_column":
                logs.append(
                    {
                        "column_name": col,
                        "strategy": strategy,
                        "affected_rows": 0,
                        "note": "No missing values found in this column.",
                    }
                )
                continue

            if strategy == "mean":
                transformed[col] = transformed[col].fillna(transformed[col].mean())
            elif strategy == "median":
                transformed[col] = transformed[col].fillna(transformed[col].median())
            elif strategy == "mode":
                mode_val = transformed[col].mode(dropna=True)
                fill_value = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                transformed[col] = transformed[col].fillna(fill_value)
            elif strategy == "forward_fill":
                transformed[col] = transformed[col].ffill()
            elif strategy == "backward_fill":
                transformed[col] = transformed[col].bfill()
            elif strategy == "interpolate":
                try:
                    transformed[col] = transformed[col].interpolate(method="linear", limit_direction="both")
                except Exception:
                    transformed[col] = transformed[col].ffill().bfill()
            elif strategy == "drop_rows":
                transformed = transformed[~transformed[col].isna()]
            elif strategy == "drop_column":
                transformed = transformed.drop(columns=[col])
            elif strategy == "unknown_label":
                transformed[col] = transformed[col].fillna("Unknown")
            elif strategy == "zero_fill":
                transformed[col] = transformed[col].fillna(0)

            logs.append(
                {
                    "column_name": col,
                    "strategy": strategy,
                    "affected_rows": affected_rows,
                    "note": f"Applied {strategy} to column '{col}'.",
                }
            )

        transformed = transformed.reset_index(drop=True)
        return transformed, logs