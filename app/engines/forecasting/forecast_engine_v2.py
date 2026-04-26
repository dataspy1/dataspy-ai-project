from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from app.engines.forecasting.feature_engineering import ForecastFeatureEngineer
from app.engines.forecasting.forecast_models import ForecastModelTrainer


class ForecastEngineV2:
    def __init__(self) -> None:
        self.feature_engineer = ForecastFeatureEngineer()
        self.model_trainer = ForecastModelTrainer()

    def run_forecast(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        horizon: int = 7,
        schema_mapping: Optional[Dict[str, str]] = None,
        segment_column: Optional[str] = None,
    ) -> Dict[str, Any]:

        if horizon <= 0:
            raise ValueError("Forecast horizon must be greater than 0.")

        if df is None or df.empty:
            return {
                "forecast_version": "v2",
                "target_column": target_col,
                "date_column": date_col,
                "forecast_horizon": horizon,
                "model_used": None,
                "metrics": {},
                "historical_points": [],
                "validation_points": [],
                "future_forecast": [],
                "feature_summary": {
                    "feature_count": 0,
                    "used_categorical_columns": [],
                    "dropped_categorical_columns": [],
                    "categorical_encoding": {},
                },
                "warnings": ["Input dataset is empty."],
                "reliability_label": "cautious",
                "summary": "Forecast could not be generated because the dataset is empty.",
            }

        # =========================
        # 🔥 Normalize irregular dates first
        # =========================
        cleaned_df, prep_warnings = self._prepare_irregular_time_series(
            df=df,
            date_col=date_col,
            target_col=target_col,
        )

        if cleaned_df.empty:
            return {
                "forecast_version": "v2",
                "target_column": target_col,
                "date_column": date_col,
                "forecast_horizon": horizon,
                "model_used": None,
                "metrics": {},
                "historical_points": [],
                "validation_points": [],
                "future_forecast": [],
                "feature_summary": {
                    "feature_count": 0,
                    "used_categorical_columns": [],
                    "dropped_categorical_columns": [],
                    "categorical_encoding": {},
                },
                "warnings": prep_warnings or [
                    "Forecast could not be generated after date and target cleanup."
                ],
                "reliability_label": "cautious",
                "summary": "Forecast could not be generated because cleaned time-series data is empty.",
            }

        # 🔥 SEGMENTED MODE
        if segment_column and segment_column in cleaned_df.columns:
            return self._run_segmented_forecast(
                df=cleaned_df,
                date_col=date_col,
                target_col=target_col,
                horizon=horizon,
                segment_column=segment_column,
                schema_mapping=schema_mapping,
            )

        # =========================
        # 🔹 Feature Engineering
        # =========================
        fe_result = self.feature_engineer.prepare_forecasting_frame(
            df=cleaned_df,
            date_col=date_col,
            target_col=target_col,
            schema_mapping=schema_mapping,
        )

        if fe_result.model_df is None or fe_result.model_df.empty:
            return {
                "forecast_version": "v2",
                "target_column": target_col,
                "date_column": date_col,
                "forecast_horizon": horizon,
                "model_used": None,
                "metrics": {},
                "historical_points": [],
                "validation_points": [],
                "future_forecast": [],
                "feature_summary": {
                    "feature_count": 0,
                    "used_categorical_columns": [],
                    "dropped_categorical_columns": [],
                    "categorical_encoding": {},
                },
                "warnings": prep_warnings + list(fe_result.warnings),
                "reliability_label": "cautious",
                "summary": "Forecast preparation failed because the engineered forecasting frame is empty.",
            }

        # =========================
        # 🔹 Model Training
        # =========================
        model_results = self.model_trainer.evaluate_models(
            model_df=fe_result.model_df,
            feature_columns=fe_result.feature_columns,
            target_col=fe_result.target_column,
            date_col=fe_result.date_column,
        )

        best_result = self.model_trainer.pick_best_model(model_results)

        # =========================
        # 🔹 Historical
        # =========================
        historical_points = self._build_historical_points(
            fe_result.model_df,
            fe_result.date_column,
            fe_result.target_column,
        )

        # =========================
        # 🔹 Validation
        # =========================
        validation_points = self._build_validation_points(
            fe_result.model_df,
            fe_result.date_column,
            best_result.actuals,
            best_result.predictions,
        )

        # =========================
        # 🔥 Future Forecast
        # =========================
        future_points = self._generate_future_forecast(
            model=best_result.model,
            model_df=fe_result.model_df,
            feature_columns=fe_result.feature_columns,
            target_col=fe_result.target_column,
            date_col=fe_result.date_column,
            horizon=horizon,
        )

        # =========================
        # 🔹 Warnings
        # =========================
        warnings = []
        warnings.extend(prep_warnings)
        warnings.extend(list(fe_result.warnings))
        warnings.extend(
            self._quality_warnings(
                fe_result.model_df,
                fe_result.used_categorical_columns,
            )
        )

        return {
            "forecast_version": "v2",
            "target_column": target_col,
            "date_column": date_col,
            "forecast_horizon": horizon,
            "model_used": best_result.model_name,
            "metrics": best_result.metrics,
            "historical_points": historical_points,
            "validation_points": validation_points,
            "future_forecast": future_points,
            "feature_summary": {
                "feature_count": len(fe_result.feature_columns),
                "used_categorical_columns": fe_result.used_categorical_columns,
                "dropped_categorical_columns": fe_result.dropped_categorical_columns,
                "categorical_encoding": fe_result.categorical_summary,
            },
            "warnings": warnings,
            "reliability_label": self._get_reliability_label(best_result.metrics),
            "summary": self._build_summary(
                target_col,
                best_result.model_name,
                horizon,
                best_result.metrics,
            ),
        }

    def _prepare_irregular_time_series(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
    ):
        warnings = []
        temp_df = df.copy()

        if date_col not in temp_df.columns:
            return pd.DataFrame(), [f"Date column '{date_col}' was not found in dataset."]

        if target_col not in temp_df.columns:
            return pd.DataFrame(), [f"Target column '{target_col}' was not found in dataset."]

        original_rows = len(temp_df)

        temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
        temp_df[target_col] = pd.to_numeric(temp_df[target_col], errors="coerce")

        invalid_date_count = int(temp_df[date_col].isna().sum())
        invalid_target_count = int(temp_df[target_col].isna().sum())

        temp_df = temp_df.dropna(subset=[date_col, target_col]).copy()

        if invalid_date_count > 0:
            warnings.append(f"{invalid_date_count} rows were dropped due to invalid dates.")
        if invalid_target_count > 0:
            warnings.append(f"{invalid_target_count} rows were dropped due to invalid target values.")

        if temp_df.empty:
            return pd.DataFrame(), warnings + ["No valid rows remained after date and target cleanup."]

        temp_df = temp_df.sort_values(date_col)

        non_numeric_cols = [col for col in temp_df.columns if col not in [date_col, target_col]]

        agg_map = {target_col: "sum"}
        for col in non_numeric_cols:
            agg_map[col] = "last"

        # Combine duplicate same-day rows
        grouped_df = (
            temp_df.groupby(date_col, as_index=False)
            .agg(agg_map)
            .sort_values(date_col)
            .reset_index(drop=True)
        )

        if len(grouped_df) < len(temp_df):
            warnings.append(
                f"{len(temp_df) - len(grouped_df)} duplicate date rows were consolidated at daily level."
            )

        grouped_df = grouped_df.set_index(date_col)

        full_index = pd.date_range(
            start=grouped_df.index.min(),
            end=grouped_df.index.max(),
            freq="D",
        )

        missing_days = len(full_index.difference(grouped_df.index))

        daily_df = grouped_df.reindex(full_index)

        if missing_days > 0:
            warnings.append(
                f"{missing_days} missing calendar dates were inserted using daily resampling."
            )

        # Keep target meaningful for business daily aggregation
        daily_df[target_col] = daily_df[target_col].fillna(0)

        # Carry forward business descriptors where possible
        for col in non_numeric_cols:
            if col in daily_df.columns:
                daily_df[col] = daily_df[col].ffill().bfill()

        daily_df.index.name = date_col
        daily_df = daily_df.reset_index()

        if len(daily_df) < 14:
            warnings.append(
                "Very short cleaned history after preprocessing; forecast may remain unstable."
            )

        if original_rows != len(daily_df):
            warnings.append(
                "Irregular date handling was applied before forecasting."
            )

        return daily_df, warnings

    # =========================
    # 🔥 SEGMENTED FORECAST
    # =========================
    def _run_segmented_forecast(
        self,
        df,
        date_col,
        target_col,
        horizon,
        segment_column,
        schema_mapping,
    ):
        segments = df[segment_column].dropna().unique()

        # 🔥 limit for safety
        segments = segments[:10]

        segment_results = []

        for segment_value in segments:
            segment_df = df[df[segment_column] == segment_value].copy()

            if len(segment_df) < 10:
                continue

            try:
                result = self.run_forecast(
                    df=segment_df,
                    date_col=date_col,
                    target_col=target_col,
                    horizon=horizon,
                    schema_mapping=schema_mapping,
                    segment_column=None,
                )

                segment_results.append({
                    "segment": f"{segment_column} = {segment_value}",
                    "model_used": result.get("model_used"),
                    "future_forecast": result.get("future_forecast"),
                    "metrics": result.get("metrics"),
                    "reliability": result.get("reliability_label"),
                })

            except Exception:
                continue

        return {
            "forecast_version": "v2_segmented",
            "segment_column": segment_column,
            "segments_count": len(segment_results),
            "segments": segment_results,
        }

    # =========================
    # 🔥 FUTURE FORECAST
    # =========================
    def _generate_future_forecast(
        self,
        model,
        model_df,
        feature_columns,
        target_col,
        date_col,
        horizon,
    ):
        # 🔥 Prophet case
        if hasattr(model, "make_future_dataframe"):
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)

            return [
                {
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "predicted": round(float(row["yhat"]), 4),
                }
                for _, row in forecast.tail(horizon).iterrows()
            ]

        # 🔥 ML recursive case
        if model is None:
            return []

        future_rows = []
        last_df = model_df.copy()

        for _ in range(horizon):
            last_row = last_df.iloc[-1:].copy()

            last_row[date_col] = pd.to_datetime(last_row[date_col]) + pd.Timedelta(days=1)

            for lag in [1, 2, 3, 7, 14]:
                lag_col = f"lag_{lag}"
                if lag_col in last_row.columns:
                    if lag == 1:
                        last_row[lag_col] = last_df[target_col].iloc[-1]
                    else:
                        last_row[lag_col] = (
                            last_df[target_col].iloc[-lag]
                            if len(last_df) >= lag
                            else last_df[target_col].iloc[-1]
                        )

            for window in [3, 7, 14]:
                col = f"rolling_mean_{window}"
                if col in last_row.columns:
                    last_row[col] = last_df[target_col].tail(window).mean()

            X = last_row[feature_columns]
            pred = model.predict(X)[0]

            last_row[target_col] = pred

            future_rows.append({
                "date": last_row[date_col].iloc[0].strftime("%Y-%m-%d"),
                "predicted": round(float(pred), 4),
            })

            last_df = pd.concat([last_df, last_row], ignore_index=True)

        return future_rows

    # =========================
    # 🔹 HELPERS
    # =========================
    def _build_historical_points(self, df, date_col, target_col):
        return [
            {
                "date": pd.to_datetime(row[date_col]).strftime("%Y-%m-%d"),
                "value": round(float(row[target_col]), 4),
            }
            for _, row in df[[date_col, target_col]].tail(60).iterrows()
        ]

    def _build_validation_points(self, df, date_col, actuals, predictions):
        tail_dates = df[date_col].tail(len(actuals)).reset_index(drop=True)

        return [
            {
                "date": pd.to_datetime(tail_dates.iloc[i]).strftime("%Y-%m-%d"),
                "actual": round(float(actuals[i]), 4),
                "predicted": round(float(predictions[i]), 4),
            }
            for i in range(len(actuals))
        ]

    def _quality_warnings(self, model_df, categorical_cols):
        warnings = []

        if len(model_df) < 30:
            warnings.append("Forecast is based on a relatively short history window.")

        if len(categorical_cols) == 0:
            warnings.append("No useful categorical business drivers were included in this forecast.")

        return warnings

    def _get_reliability_label(self, metrics):
        mape = metrics.get("mape", 999)

        if mape <= 10:
            return "high"
        if mape <= 20:
            return "moderate"
        return "cautious"

    def _build_summary(self, target_col, model_name, horizon, metrics):
        return (
            f"Forecast generated for '{target_col}' using {model_name} "
            f"with a {horizon}-step horizon. "
            f"Validation metrics: MAE={metrics.get('mae')}, "
            f"RMSE={metrics.get('rmse')}, MAPE={metrics.get('mape')}%."
        )