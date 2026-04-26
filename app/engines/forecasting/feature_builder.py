from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


class ForecastFeatureBuilder:
    """
    Builds time-series forecasting features for ML models.

    Feature groups:
    - lag features
    - rolling statistics
    - momentum / delta features
    - ratio features
    - calendar features
    - cyclical calendar encoding
    """

    def __init__(
        self,
        lag_periods: List[int] | None = None,
        rolling_windows: List[int] | None = None,
    ) -> None:
        self.lag_periods = lag_periods or [1, 2, 3, 5, 7, 14, 21, 28]
        self.rolling_windows = rolling_windows or [3, 7, 14]

    def build_training_frame(
        self,
        series: pd.Series,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Input:
            series: pd.Series with DatetimeIndex and numeric values

        Output:
            feature_df: dataframe with target + engineered features
            feature_columns: list of usable feature columns
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series index must be DatetimeIndex for feature engineering.")

        df = pd.DataFrame({"target": pd.to_numeric(series, errors="coerce")}).copy()
        df = df.sort_index()

        # Lag features
        for lag in self.lag_periods:
            df[f"lag_{lag}"] = df["target"].shift(lag)

        # Rolling features based only on past data
        for window in self.rolling_windows:
            shifted = df["target"].shift(1)
            df[f"rolling_mean_{window}"] = shifted.rolling(window=window).mean()
            df[f"rolling_std_{window}"] = shifted.rolling(window=window).std()
            df[f"rolling_min_{window}"] = shifted.rolling(window=window).min()
            df[f"rolling_max_{window}"] = shifted.rolling(window=window).max()

        # Momentum / change features
        df["diff_1"] = df["target"].shift(1) - df["target"].shift(2)
        df["diff_7"] = df["target"].shift(1) - df["target"].shift(8)

        # Ratio features
        if "rolling_mean_3" in df.columns:
            df["ratio_to_rollmean_3"] = df["target"].shift(1) / (df["rolling_mean_3"] + 1e-6)
        if "rolling_mean_7" in df.columns:
            df["ratio_to_rollmean_7"] = df["target"].shift(1) / (df["rolling_mean_7"] + 1e-6)

        # Basic calendar features
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["day_of_year"] = df.index.dayofyear
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["week_of_year"] = df.index.isocalendar().week.astype(int)
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)

        # Cyclical encodings
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12.0)
        df["day_of_month_sin"] = np.sin(2 * np.pi * df.index.day / 31.0)
        df["day_of_month_cos"] = np.cos(2 * np.pi * df.index.day / 31.0)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().copy()

        feature_columns = [col for col in df.columns if col != "target"]
        return df, feature_columns

    def build_next_row_features(
        self,
        history_series: pd.Series,
        next_timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Build a single feature row for recursive forecasting.
        """
        if not isinstance(history_series.index, pd.DatetimeIndex):
            raise ValueError("History series index must be DatetimeIndex.")

        history_series = pd.to_numeric(history_series, errors="coerce").dropna().sort_index()

        if history_series.empty:
            raise ValueError("History series is empty after cleaning.")

        row = {}

        # Lag features
        for lag in self.lag_periods:
            if len(history_series) >= lag:
                row[f"lag_{lag}"] = float(history_series.iloc[-lag])
            else:
                row[f"lag_{lag}"] = float(history_series.iloc[0])

        # Rolling features
        for window in self.rolling_windows:
            usable_window = min(window, len(history_series))
            recent = history_series.iloc[-usable_window:]

            row[f"rolling_mean_{window}"] = float(recent.mean())
            row[f"rolling_std_{window}"] = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
            row[f"rolling_min_{window}"] = float(recent.min())
            row[f"rolling_max_{window}"] = float(recent.max())

        # Momentum / change features
        if len(history_series) >= 2:
            row["diff_1"] = float(history_series.iloc[-1] - history_series.iloc[-2])
        else:
            row["diff_1"] = 0.0

        if len(history_series) >= 8:
            row["diff_7"] = float(history_series.iloc[-1] - history_series.iloc[-8])
        else:
            row["diff_7"] = row["diff_1"]

        # Ratio features
        roll3 = row.get("rolling_mean_3", 0.0)
        roll7 = row.get("rolling_mean_7", 0.0)
        last_val = float(history_series.iloc[-1])

        row["ratio_to_rollmean_3"] = float(last_val / (roll3 + 1e-6))
        row["ratio_to_rollmean_7"] = float(last_val / (roll7 + 1e-6))

        # Basic calendar features
        row["day_of_week"] = int(next_timestamp.dayofweek)
        row["day_of_month"] = int(next_timestamp.day)
        row["day_of_year"] = int(next_timestamp.dayofyear)
        row["month"] = int(next_timestamp.month)
        row["quarter"] = int(next_timestamp.quarter)
        row["week_of_year"] = int(next_timestamp.isocalendar().week)
        row["is_weekend"] = int(next_timestamp.dayofweek >= 5)
        row["is_month_start"] = int(next_timestamp.is_month_start)
        row["is_month_end"] = int(next_timestamp.is_month_end)

        # Cyclical encodings
        row["dow_sin"] = float(np.sin(2 * np.pi * next_timestamp.dayofweek / 7.0))
        row["dow_cos"] = float(np.cos(2 * np.pi * next_timestamp.dayofweek / 7.0))
        row["month_sin"] = float(np.sin(2 * np.pi * next_timestamp.month / 12.0))
        row["month_cos"] = float(np.cos(2 * np.pi * next_timestamp.month / 12.0))
        row["day_of_month_sin"] = float(np.sin(2 * np.pi * next_timestamp.day / 31.0))
        row["day_of_month_cos"] = float(np.cos(2 * np.pi * next_timestamp.day / 31.0))

        return pd.DataFrame([row])