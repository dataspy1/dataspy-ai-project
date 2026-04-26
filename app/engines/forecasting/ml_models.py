from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from app.engines.forecasting.feature_builder import ForecastFeatureBuilder


class XGBoostForecastModel:
    """
    Recursive XGBoost forecaster using lag, rolling, and calendar features.

    Improvements in this version:
    - stronger XGBoost configuration
    - recency-weighted training
    - recursive forecasting with pattern-aware guardrails
    - seasonal fallback support for daily data
    - trend blending to avoid dead-flat or overly straight forecasts
    - clipping to prevent unrealistic jumps
    """

    def __init__(self) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            ) from e

        self.XGBRegressor = XGBRegressor
        self.feature_builder = ForecastFeatureBuilder()

    def predict(
        self,
        train_series: pd.Series,
        horizon: int,
    ) -> np.ndarray:
        if not isinstance(train_series.index, pd.DatetimeIndex):
            raise ValueError("train_series must use DatetimeIndex.")

        train_series = (
            pd.to_numeric(train_series, errors="coerce")
            .dropna()
            .sort_index()
        )

        if len(train_series) < 20:
            raise ValueError("Not enough history for XGBoost forecasting. At least 20 points required.")

        feature_df, feature_columns = self.feature_builder.build_training_frame(train_series)

        if feature_df.empty:
            raise ValueError("Feature frame is empty after feature engineering.")

        X_train = feature_df[feature_columns]
        y_train = feature_df["target"]

        if len(X_train) < 10:
            raise ValueError("Training feature frame is too small for XGBoost forecasting.")

        sample_weights = self._build_sample_weights(len(X_train))

        model = self.XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            reg_alpha=0.1,
            reg_lambda=1.5,
            gamma=0.0,
            objective="reg:squarederror",
            random_state=42,
        )

        model.fit(X_train, y_train, sample_weight=sample_weights)

        future_dates = self._generate_future_dates(train_series.index, horizon)
        working_series = train_series.copy()
        predictions: List[float] = []

        for next_ts in future_dates:
            next_row = self.feature_builder.build_next_row_features(working_series, next_ts)
            raw_pred = float(model.predict(next_row[feature_columns])[0])

            adjusted_pred = self._apply_prediction_guardrails(
                working_series=working_series,
                next_ts=next_ts,
                raw_pred=raw_pred,
            )

            predictions.append(adjusted_pred)
            working_series.loc[next_ts] = adjusted_pred

        return np.array(predictions, dtype=float)

    def _build_sample_weights(self, n_rows: int) -> np.ndarray:
        """
        Give more importance to recent data points.
        """
        if n_rows <= 1:
            return np.ones(n_rows, dtype=float)

        weights = np.linspace(0.6, 1.4, n_rows)
        return weights.astype(float)

    def _apply_prediction_guardrails(
        self,
        working_series: pd.Series,
        next_ts: pd.Timestamp,
        raw_pred: float,
    ) -> float:
        """
        Blend XGBoost output with recent trend and seasonal hints,
        then clip to a reasonable dynamic range.
        """
        recent_window = min(len(working_series), 14)
        recent_series = working_series.iloc[-recent_window:].astype(float)

        recent_mean = float(recent_series.mean())
        recent_std = float(recent_series.std()) if len(recent_series) > 1 else 0.0

        trend_pred = self._compute_trend_projection(working_series)
        seasonal_pred = self._compute_seasonal_reference(working_series, next_ts)

        blended_pred = raw_pred

        if trend_pred is not None and seasonal_pred is not None:
            blended_pred = (0.60 * raw_pred) + (0.25 * trend_pred) + (0.15 * seasonal_pred)
        elif trend_pred is not None:
            blended_pred = (0.75 * raw_pred) + (0.25 * trend_pred)
        elif seasonal_pred is not None:
            blended_pred = (0.80 * raw_pred) + (0.20 * seasonal_pred)

        # Dynamic clipping so forecast stays believable
        if recent_std > 0:
            lower_bound = recent_mean - (2.5 * recent_std)
            upper_bound = recent_mean + (2.5 * recent_std)
            blended_pred = float(np.clip(blended_pred, lower_bound, upper_bound))

        # Preserve non-negative business metrics if history is non-negative
        if float(working_series.min()) >= 0:
            blended_pred = max(0.0, blended_pred)

        return float(blended_pred)

    def _compute_trend_projection(self, working_series: pd.Series) -> Optional[float]:
        """
        Estimate short-term trend from recent history using linear regression.
        """
        if len(working_series) < 5:
            return None

        recent_window = min(len(working_series), 10)
        recent_y = working_series.iloc[-recent_window:].astype(float).values

        if len(recent_y) < 2:
            return None

        x = np.arange(len(recent_y), dtype=float)

        try:
            slope, intercept = np.polyfit(x, recent_y, 1)
        except Exception:
            return None

        next_x = x[-1] + 1
        trend_pred = intercept + slope * next_x
        return float(trend_pred)

    def _compute_seasonal_reference(
        self,
        working_series: pd.Series,
        next_ts: pd.Timestamp,
    ) -> Optional[float]:
        """
        Use same-weekday historical values as a weak seasonal guide for daily data.
        """
        if not isinstance(working_series.index, pd.DatetimeIndex):
            return None

        inferred_freq = pd.infer_freq(working_series.index)

        # This helper is most useful for daily data
        if inferred_freq not in {"D", "B", None}:
            return None

        same_weekday_series = working_series[working_series.index.weekday == next_ts.weekday()]

        if len(same_weekday_series) < 2:
            return None

        seasonal_window = min(4, len(same_weekday_series))
        seasonal_pred = float(same_weekday_series.iloc[-seasonal_window:].mean())
        return seasonal_pred

    def _generate_future_dates(
        self,
        historical_index: pd.DatetimeIndex,
        horizon: int,
    ) -> List[pd.Timestamp]:
        inferred_freq = pd.infer_freq(historical_index)
        last_date = pd.to_datetime(historical_index[-1])

        if inferred_freq:
            return list(
                pd.date_range(
                    start=last_date,
                    periods=horizon + 1,
                    freq=inferred_freq,
                )[1:]
            )

        return [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]