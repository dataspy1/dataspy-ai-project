import os
import json
import uuid
import pandas as pd

EXPORT_DIR = "app/exports"
os.makedirs(EXPORT_DIR, exist_ok=True)


def create_decisions_csv(decisions: list[dict]) -> str:
    rows = []

    for idx, decision in enumerate(decisions, start=1):
        rows.append({
            "sr_no": idx,
            "priority": decision.get("priority"),
            "action": decision.get("action"),
            "rationale": decision.get("rationale"),
            "decision_type": decision.get("decision_type"),
            "rank_order": decision.get("rank_order"),
            "supporting_evidence": json.dumps(decision.get("supporting_evidence", {}))
        })

    df = pd.DataFrame(rows)

    filename = f"decisions_{uuid.uuid4().hex[:8]}.csv"
    filepath = os.path.join(EXPORT_DIR, filename)
    df.to_csv(filepath, index=False)

    return filepath


def create_forecast_csv(forecast: dict) -> str:
    historical_series = forecast.get("historical_series", [])
    forecast_series = forecast.get("forecast_series", [])

    rows = []

    for row in historical_series:
        rows.append({
            "series_type": "historical",
            "date": row.get("date"),
            "value": row.get("value")
        })

    for row in forecast_series:
        rows.append({
            "series_type": "forecast",
            "date": row.get("date"),
            "value": row.get("value")
        })

    df = pd.DataFrame(rows)

    filename = f"forecast_{uuid.uuid4().hex[:8]}.csv"
    filepath = os.path.join(EXPORT_DIR, filename)
    df.to_csv(filepath, index=False)

    return filepath