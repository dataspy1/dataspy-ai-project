from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# 🔹 XGBoost
# =========================
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# =========================
# 🔥 Prophet
# =========================
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# =========================
# 🔹 DATA STRUCTURE
# =========================
@dataclass
class ModelTrainingResult:
    model_name: str
    model: object | None
    metrics: Dict[str, float]
    predictions: List[float]
    actuals: List[float]


# =========================
# 🔹 MAIN TRAINER
# =========================
class ForecastModelTrainer:

    def evaluate_models(
        self,
        model_df: pd.DataFrame,
        feature_columns: List[str],
        target_col: str,
        date_col: str,   # 🔥 NEW PARAM
    ) -> List[ModelTrainingResult]:

        if len(model_df) < 10:
            raise ValueError("Not enough rows to evaluate forecasting models.")

        split_index = int(len(model_df) * 0.8)

        train_df = model_df.iloc[:split_index].copy()
        test_df = model_df.iloc[split_index:].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Train/test split failed due to insufficient rows.")

        X_train = train_df[feature_columns]
        y_train = train_df[target_col]

        X_test = test_df[feature_columns]
        y_test = test_df[target_col]

        results: List[ModelTrainingResult] = []

        # =========================
        # 🔹 BASELINE MODEL
        # =========================
        baseline_preds = self._naive_forecast(train_df[target_col], len(test_df))
        baseline_metrics = self._calculate_metrics(y_test, baseline_preds)

        results.append(
            ModelTrainingResult(
                model_name="baseline_naive",
                model=None,
                metrics=baseline_metrics,
                predictions=baseline_preds.tolist(),
                actuals=y_test.tolist(),
            )
        )

        # =========================
        # 🔹 TREE MODEL (XGB / RF)
        # =========================
        if XGBOOST_AVAILABLE:
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            model_name = "xgboost"
        else:
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            model_name = "random_forest_fallback"

        metrics = self._calculate_metrics(y_test, preds)

        results.append(
            ModelTrainingResult(
                model_name=model_name,
                model=model,
                metrics=metrics,
                predictions=preds.tolist(),
                actuals=y_test.tolist(),
            )
        )

        # =========================
        # 🔥 PROPHET MODEL
        # =========================
        if PROPHET_AVAILABLE:
            try:
                prophet_model = self._train_prophet(
                    df=train_df,
                    date_col=date_col,
                    target_col=target_col,
                )

                future_df = test_df[[date_col]].rename(columns={date_col: "ds"})
                forecast = prophet_model.predict(future_df)

                preds = forecast["yhat"].values
                metrics = self._calculate_metrics(y_test, preds)

                results.append(
                    ModelTrainingResult(
                        model_name="prophet",
                        model=prophet_model,
                        metrics=metrics,
                        predictions=preds.tolist(),
                        actuals=y_test.tolist(),
                    )
                )
            except Exception:
                # Safe fallback — do not crash system
                pass

        return results

    # =========================
    # 🔹 PICK BEST MODEL
    # =========================
    def pick_best_model(self, results: List[ModelTrainingResult]) -> ModelTrainingResult:
        return min(results, key=lambda x: x.metrics.get("rmse", float("inf")))

    # =========================
    # 🔹 BASELINE FORECAST
    # =========================
    def _naive_forecast(self, train_target: pd.Series, steps: int) -> np.ndarray:
        last_value = train_target.iloc[-1]
        return np.array([last_value] * steps)

    # =========================
    # 🔹 PROPHET TRAINER
    # =========================
    def _train_prophet(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
    ):
        prophet_df = df[[date_col, target_col]].rename(
            columns={date_col: "ds", target_col: "y"}
        )

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )

        model.fit(prophet_df)

        return model

    # =========================
    # 🔹 METRICS
    # =========================
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        y_true_np = np.array(y_true, dtype=float)
        y_pred_np = np.array(y_pred, dtype=float)

        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = float(np.sqrt(mean_squared_error(y_true_np, y_pred_np)))

        nonzero_mask = y_true_np != 0
        if nonzero_mask.any():
            mape = float(
                np.mean(
                    np.abs(
                        (y_true_np[nonzero_mask] - y_pred_np[nonzero_mask])
                        / y_true_np[nonzero_mask]
                    )
                )
                * 100
            )
        else:
            mape = 0.0

        return {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "mape": round(float(mape), 4),
        }