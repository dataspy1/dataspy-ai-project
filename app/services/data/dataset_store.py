from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[3]

UPLOAD_DIRS = [
    BASE_DIR / "uploads",
    BASE_DIR / "app" / "uploads",
]

PROCESSED_DIRS = [
    BASE_DIR / "processed",
    BASE_DIR / "app" / "processed",
]

TEMPLATE_DIR = BASE_DIR / "templates"

for folder in UPLOAD_DIRS + PROCESSED_DIRS + [TEMPLATE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


class DatasetStore:
    @staticmethod
    def _find_existing_file(filename: str) -> Path | None:
        # direct checks first
        for folder in UPLOAD_DIRS + PROCESSED_DIRS:
            candidate = folder / filename
            if candidate.exists():
                return candidate

        # recursive fallback inside backend/
        for candidate in BASE_DIR.rglob(filename):
            if candidate.is_file():
                return candidate

        return None

    @staticmethod
    def get_upload_path(filename: str) -> Path:
        found = DatasetStore._find_existing_file(filename)
        if found:
            return found
        return UPLOAD_DIRS[0] / filename

    @staticmethod
    def get_processed_path(filename: str) -> Path:
        found = DatasetStore._find_existing_file(filename)
        if found:
            return found
        return PROCESSED_DIRS[0] / filename

    @staticmethod
    def get_template_path(template_name: str) -> Path:
        safe_name = template_name.replace(" ", "_").lower()
        return TEMPLATE_DIR / f"{safe_name}.json"

    @staticmethod
    def load_dataframe(filename: str) -> pd.DataFrame:
        path = DatasetStore._find_existing_file(filename)

        if not path:
            raise FileNotFoundError(f"Dataset not found: {filename}")

        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif suffix == ".json":
            return pd.read_json(path)

        raise ValueError("Unsupported file type. Use CSV, Excel, or JSON.")

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filename: str, processed: bool = True) -> str:
        target_folder = PROCESSED_DIRS[0] if processed else UPLOAD_DIRS[0]
        path = target_folder / filename

        suffix = path.suffix.lower()

        if suffix == ".csv":
            df.to_csv(path, index=False)
        elif suffix in [".xlsx", ".xls"]:
            df.to_excel(path, index=False)
        elif suffix == ".json":
            df.to_json(path, orient="records", indent=2)
        else:
            raise ValueError("Unsupported file type for saving.")

        return path.name