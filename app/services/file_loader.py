import os
import uuid
import json
import pandas as pd


SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls", ".json"]


def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()


def validate_file_extension(filename: str):
    ext = get_file_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported types are: {SUPPORTED_EXTENSIONS}"
        )


def save_uploaded_file(upload_file, upload_dir: str = "app/uploads") -> tuple[str, str]:
    os.makedirs(upload_dir, exist_ok=True)

    ext = get_file_extension(upload_file.filename)
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(upload_dir, unique_name)

    with open(file_path, "wb") as f:
        f.write(upload_file.file.read())

    return unique_name, file_path


def load_dataframe(file_path: str) -> pd.DataFrame:
    ext = get_file_extension(file_path)

    if ext == ".csv":
        df = pd.read_csv(file_path)

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return df


def build_upload_response(df: pd.DataFrame, saved_filename: str) -> dict:
    preview = df.head(5).fillna("").to_dict(orient="records")

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    return {
        "saved_filename": saved_filename,
        "preview": preview,
        "columns": list(df.columns),
        "dtypes": dtypes
    }