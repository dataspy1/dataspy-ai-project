from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

from app.engines.forecasting.schemas import (
    FoldResult,
    ValidationMetrics,
    ValidationResult,
)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    non_zero_mask = y_true != 0
    if non_zero_mask.sum() == 0:
        return None

    return float(
        np.mean(
            np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
        ) * 100
    )


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    valid_mask = denominator != 0
    if valid_mask.sum() == 0:
        return None

    return float(
        np.mean(
            np.abs(y_true[valid_mask] - y_pred[valid_mask]) / denominator[valid_mask]
        ) * 100
    )


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return None
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)


class TimeSeriesValidator:
    """
    Expanding-window validator for time series forecasting.

    Production upgrades:
    - preserves DatetimeIndex for ML models
    - stability scoring
    - consistency scoring
    - validation confidence label
    """

    def __init__(
        self,
        min_train_size: int = 20,
        test_size: int = 5,
        max_folds: int = 3,
    ) -> None:
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.max_folds = max_folds

    def validate(
        self,
        series: pd.Series,
        model_name: str,
        predictor_fn: Callable[[pd.Series, int], np.ndarray],
    ) -> ValidationResult:

        clean_series = pd.to_numeric(series, errors="coerce").dropna()

        if len(clean_series) < self.min_train_size + self.test_size:
            metrics = ValidationMetrics(None, None, None, None, None)
            return ValidationResult(
                model_name=model_name,
                fold_results=[],
                aggregate_metrics=metrics,
                is_stable=False,
                notes=["Insufficient data for validation"],
            )

        folds = self._build_expanding_folds(len(clean_series))
        fold_results: List[FoldResult] = []

        for fold_number, (train_end, test_end) in enumerate(folds, start=1):
            train_series = clean_series.iloc[:train_end]
            test_series = clean_series.iloc[train_end:test_end]

            predictions = predictor_fn(train_series, len(test_series))
            predictions = np.array(predictions, dtype=float)

            y_true = test_series.to_numpy(dtype=float)
            y_pred = predictions

            if len(y_pred) != len(y_true):
                raise ValueError(
                    f"Prediction length mismatch for model '{model_name}': "
                    f"expected {len(y_true)}, got {len(y_pred)}"
                )

            fold_metrics = ValidationMetrics(
                mae=round(mae(y_true, y_pred), 4),
                rmse=round(rmse(y_true, y_pred), 4),
                mape=round(mape(y_true, y_pred), 4) if mape(y_true, y_pred) is not None else None,
                smape=round(smape(y_true, y_pred), 4) if smape(y_true, y_pred) is not None else None,
                wape=round(wape(y_true, y_pred), 4) if wape(y_true, y_pred) is not None else None,
            )

            fold_results.append(
                FoldResult(
                    fold_number=fold_number,
                    train_size=len(train_series),
                    test_size=len(test_series),
                    metrics=fold_metrics,
                )
            )

        aggregate_metrics = self._aggregate_metrics(fold_results)
        stability_score, is_stable = self._calculate_stability(fold_results)
        consistency_score = self._calculate_consistency(fold_results)

        notes = []
        if not is_stable:
            notes.append("High variation across folds detected")

        return ValidationResult(
            model_name=model_name,
            fold_results=fold_results,
            aggregate_metrics=aggregate_metrics,
            is_stable=is_stable,
            notes=notes + [
                f"stability_score={round(stability_score, 3)}",
                f"consistency_score={round(consistency_score, 3)}",
            ],
        )

    def _build_expanding_folds(self, n_obs: int) -> List[Tuple[int, int]]:
        folds = []
        train_end = self.min_train_size

        while train_end + self.test_size <= n_obs and len(folds) < self.max_folds:
            test_end = train_end + self.test_size
            folds.append((train_end, test_end))
            train_end += self.test_size

        return folds

    def _aggregate_metrics(self, fold_results: List[FoldResult]) -> ValidationMetrics:
        def avg(metric: str):
            values = [
                getattr(f.metrics, metric)
                for f in fold_results
                if getattr(f.metrics, metric) is not None
            ]
            return round(float(np.mean(values)), 4) if values else None

        return ValidationMetrics(
            mae=avg("mae"),
            rmse=avg("rmse"),
            mape=avg("mape"),
            smape=avg("smape"),
            wape=avg("wape"),
        )

    def _calculate_stability(self, fold_results: List[FoldResult]) -> Tuple[float, bool]:
        maes = [f.metrics.mae for f in fold_results if f.metrics.mae is not None]

        if len(maes) < 2:
            return 1.0, True

        mean = float(np.mean(maes))
        std = float(np.std(maes))

        if mean == 0:
            return 1.0, True

        cv = std / mean
        stability_score = max(0.0, 1 - cv)

        return stability_score, cv < 0.35

    def _calculate_consistency(self, fold_results: List[FoldResult]) -> float:
        maes = [f.metrics.mae for f in fold_results if f.metrics.mae is not None]

        if len(maes) < 2:
            return 1.0

        diffs = np.diff(maes)
        variance = np.var(diffs)

        return float(1 / (1 + variance))