import re
import pandas as pd


DATE_NAME_HINTS = [
    "date",
    "dt",
    "day",
    "invoice_date",
    "delivery_date",
    "shipment_date",
    "ship_date",
    "dispatch_date",
    "posting_date",
    "transaction_date",
    "order_date",
    "po_date",
    "bill_date",
    "billing_date",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", case=False, na=False)]

    # Clean column names
    df.columns = [str(col).strip() for col in df.columns]

    # Replace empty strings with NA
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Replace common null-like strings
    df = df.replace(
        {
            "NA": pd.NA,
            "N/A": pd.NA,
            "null": pd.NA,
            "NULL": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
        }
    )

    # Drop fully empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Clean only non-null string values
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    return df


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _looks_like_date_column_name(col_name: str) -> bool:
    normalized = _normalize(col_name)
    return any(hint in normalized for hint in DATE_NAME_HINTS)


def _sample_date_ratio(series: pd.Series) -> float:
    sample = series.dropna().astype(str).head(30)

    if sample.empty:
        return 0.0

    # Remove obvious non-date values
    sample = sample[sample.str.len().between(6, 25)]
    if sample.empty:
        return 0.0

    # First pass: normal parser with dayfirst
    converted = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    ratio = float(converted.notna().mean()) if len(sample) else 0.0

    if ratio >= 0.6:
        return ratio

    # Second pass: a few common strict business formats
    common_formats = [
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d-%m-%y",
        "%d/%m/%y",
    ]

    best_ratio = ratio
    for fmt in common_formats:
        try:
            converted_fmt = pd.to_datetime(sample, errors="coerce", format=fmt)
            fmt_ratio = float(converted_fmt.notna().mean()) if len(sample) else 0.0
            best_ratio = max(best_ratio, fmt_ratio)
        except Exception:
            continue

    return best_ratio


def _convert_series_to_datetime(series: pd.Series) -> pd.Series:
    # First generic pass
    converted = pd.to_datetime(series, errors="coerce", dayfirst=True)

    # If generic parsing is already good enough, keep it
    ratio = float(converted.notna().mean()) if len(series.dropna()) else 0.0
    if ratio >= 0.6:
        return converted

    # Try strict common formats and keep the best
    common_formats = [
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d-%m-%y",
        "%d/%m/%y",
    ]

    best_series = converted
    best_ratio = ratio

    for fmt in common_formats:
        try:
            candidate = pd.to_datetime(series, errors="coerce", format=fmt)
            candidate_ratio = float(candidate.notna().mean()) if len(series.dropna()) else 0.0
            if candidate_ratio > best_ratio:
                best_ratio = candidate_ratio
                best_series = candidate
        except Exception:
            continue

    return best_series


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        try:
            series = df[col]

            # Skip columns already datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                continue

            # Skip strong numeric columns
            if pd.api.types.is_numeric_dtype(series):
                continue

            name_hint = _looks_like_date_column_name(col)
            sample_ratio = _sample_date_ratio(series)

            # Only parse if column name suggests date
            # OR values strongly look like dates
            if not name_hint and sample_ratio < 0.75:
                continue

            converted = _convert_series_to_datetime(series)

            # Only assign if parsing quality is decent
            final_ratio = float(converted.notna().mean()) if len(series.dropna()) else 0.0
            if final_ratio >= 0.6:
                df[col] = converted

        except Exception:
            continue

    return df