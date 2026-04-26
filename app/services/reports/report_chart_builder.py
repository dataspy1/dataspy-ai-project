from typing import List, Tuple
import matplotlib.pyplot as plt

from app.services.reports.pdf_report_service import (
    build_chart_filename,
    get_chart_output_path,
)


def extract_xy(series: List[dict], prefix: str) -> Tuple[List[str], List[float]]:
    labels: List[str] = []
    values: List[float] = []

    for idx, row in enumerate(series):
        if not isinstance(row, dict):
            continue

        label = (
            row.get("date")
            or row.get("ds")
            or row.get("timestamp")
            or row.get("period")
            or row.get("label")
            or row.get("x")
            or row.get("date_label")
            or row.get("time")
            or row.get("index")
            or f"{prefix}{idx + 1}"
        )

        raw_value = (
            row.get("value")
            if row.get("value") is not None
            else row.get("forecast")
            if row.get("forecast") is not None
            else row.get("predicted_value")
            if row.get("predicted_value") is not None
            else row.get("yhat")
            if row.get("yhat") is not None
            else row.get("target_value")
            if row.get("target_value") is not None
            else row.get("actual")
            if row.get("actual") is not None
            else row.get("y")
            if row.get("y") is not None
            else row.get("forecast_value")
            if row.get("forecast_value") is not None
            else row.get("predicted")
            if row.get("predicted") is not None
            else row.get("amount")
            if row.get("amount") is not None
            else None
        )

        if raw_value is None:
            continue

        try:
            values.append(float(raw_value))
            labels.append(str(label))
        except Exception:
            continue

    return labels, values


def generate_forecast_chart(
    historical_series: List[dict],
    forecast_series: List[dict],
) -> str:
    filename = build_chart_filename("forecast_chart")
    chart_path = get_chart_output_path(filename)

    historical_labels, historical_values = extract_xy(historical_series, "H")
    forecast_labels, forecast_values = extract_xy(forecast_series, "F")

    plt.figure(figsize=(9, 4.8))

    plotted = False

    if historical_labels and historical_values:
        plt.plot(historical_labels, historical_values, marker="o", label="Historical")
        plotted = True

    if forecast_labels and forecast_values:
        plt.plot(forecast_labels, forecast_values, marker="o", label="Forecast")
        plotted = True

    plt.title("Forecast Trend")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.xticks(rotation=30)

    if plotted:
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No forecast series available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(chart_path)