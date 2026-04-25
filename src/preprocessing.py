from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
BUSINESS_CLASS_PRICE_THRESHOLD = 350000.0

USED_SCHEMA = {
    "car_model",
    "month_year",
    "average_price",
    "minimum_price",
    "maximum_price",
}

NEW_SCHEMA = {
    "car_model",
    "old_price",
    "price_change",
    "new_price",
    "date_range",
}


def normalize_column_name(column_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower())
    return normalized.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: normalize_column_name(column) for column in df.columns}
    return df.rename(columns=renamed)


def extract_numbers(value: Any) -> list[float]:
    if value is None or pd.isna(value):
        return []

    text = str(value)
    matches = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    numbers: list[float] = []
    for match in matches:
        cleaned = match.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def parse_price(value: Any) -> float | None:
    numbers = extract_numbers(value)
    if not numbers:
        return None

    if len(numbers) == 1:
        return numbers[0]

    return sum(numbers) / len(numbers)


def parse_price_change(value: Any) -> float | None:
    numbers = extract_numbers(value)
    if not numbers:
        return None

    amount = abs(numbers[0])
    text = str(value).lower()
    if "trending_down" in text or "-" in text:
        return -amount
    return amount


def extract_model_year(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None

    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    if match is None:
        return None

    return int(match.group(0))


def split_car_model(value: Any) -> tuple[str | None, str | None]:
    if value is None or pd.isna(value):
        return None, None

    text = str(value).strip()
    if not text:
        return None, None

    parts = text.split()
    if len(parts) == 1:
        return parts[0], None

    return parts[0], " ".join(parts[1:])


def drop_empty_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    price_columns = [
        column
        for column in ["average_price", "minimum_price", "maximum_price", "old_price", "new_price"]
        if column in df.columns
    ]
    if not price_columns:
        return df
    return df.dropna(subset=price_columns, how="all").copy()


def ensure_columns(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    prepared = df.copy()
    for column in required_columns:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    return prepared


def detect_table_kind(df: pd.DataFrame) -> str:
    columns = set(df.columns)
    if USED_SCHEMA.issubset(columns):
        return "used"
    if NEW_SCHEMA.issubset(columns):
        return "new"
    raise ValueError(f"Unknown dataset schema: {sorted(columns)}")


def preprocess_used_table(df: pd.DataFrame) -> pd.DataFrame:
    prepared = normalize_columns(df).copy()
    prepared = drop_empty_price_rows(prepared)
    prepared = ensure_columns(prepared, ["web_scraper_order"])
    prepared["dataset_part"] = "used"
    brand_and_model = prepared["car_model"].map(split_car_model)
    prepared["brand"] = brand_and_model.map(lambda item: item[0])
    prepared["model"] = brand_and_model.map(lambda item: item[1])
    prepared["average_price_value"] = prepared["average_price"].map(parse_price)
    prepared["minimum_price_value"] = prepared["minimum_price"].map(parse_price)
    prepared["maximum_price_value"] = prepared["maximum_price"].map(parse_price)
    prepared["model_year"] = prepared["car_model"].map(extract_model_year).astype("Int64")
    prepared["snapshot_date"] = pd.to_datetime(prepared["month_year"], format="%Y-%m", errors="coerce")

    return prepared[
        [
            "dataset_part",
            "web_scraper_order",
            "brand",
            "model",
            "car_model",
            "model_year",
            "average_price_value",
            "minimum_price_value",
            "maximum_price_value",
            "snapshot_date",
        ]
    ].rename(
        columns={
            "car_model": "trim_or_variant",
            "average_price_value": "average_price",
            "minimum_price_value": "minimum_price",
            "maximum_price_value": "maximum_price",
        }
    )


def preprocess_new_table(df: pd.DataFrame) -> pd.DataFrame:
    prepared = normalize_columns(df).copy()
    prepared = drop_empty_price_rows(prepared)
    prepared = ensure_columns(prepared, ["web_scraper_order"])
    prepared["dataset_part"] = "new"
    brand_and_model = prepared["car_model"].map(split_car_model)
    prepared["brand"] = brand_and_model.map(lambda item: item[0])
    prepared["model"] = brand_and_model.map(lambda item: item[1])
    prepared["model_year"] = prepared["car_model"].map(extract_model_year).astype("Int64")
    prepared["old_price_value"] = prepared["old_price"].map(parse_price)
    prepared["new_price_value"] = prepared["new_price"].map(parse_price)
    prepared["price_change_value"] = prepared["price_change"].map(parse_price_change)
    prepared["snapshot_date"] = pd.to_datetime(prepared["date_range"], format="%d/%m/%Y", errors="coerce")

    return prepared[
        [
            "dataset_part",
            "web_scraper_order",
            "brand",
            "model",
            "car_model",
            "model_year",
            "old_price_value",
            "price_change_value",
            "new_price_value",
            "snapshot_date",
        ]
    ].rename(
        columns={
            "car_model": "trim_or_variant",
            "old_price_value": "old_price",
            "price_change_value": "price_change",
            "new_price_value": "new_price",
        }
    )


def read_raw_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def find_raw_tables(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    candidates = sorted(path for path in raw_dir.iterdir() if path.is_file())
    return [path for path in candidates if path.suffix.lower() in {".csv", ".xlsx", ".xls"}]


def build_profile(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    target_column = "target_price" if "target_price" in df.columns else None
    has_model_year = "model_year" in df.columns and not df["model_year"].dropna().empty
    return {
        "table_name": table_name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": {column: int(value) for column, value in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_brands": int(df["brand"].nunique(dropna=True)) if "brand" in df.columns else 0,
        "model_year_min": int(df["model_year"].dropna().min()) if has_model_year else None,
        "model_year_max": int(df["model_year"].dropna().max()) if has_model_year else None,
        "target_price_min": float(df[target_column].min()) if target_column else None,
        "target_price_max": float(df[target_column].max()) if target_column else None,
    }


def make_model_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["brand_model"] = prepared["brand"].fillna("") + " " + prepared["model"].fillna("")
    prepared["brand_model"] = prepared["brand_model"].str.strip()
    return prepared


def standardize_for_modeling(df: pd.DataFrame, table_kind: str) -> pd.DataFrame:
    prepared = make_model_features(df)
    if table_kind == "used":
        prepared["target_price"] = prepared["average_price"]
    elif table_kind == "new":
        prepared["target_price"] = prepared["new_price"]
    else:
        raise ValueError(f"Unsupported table kind: {table_kind}")

    return prepared[
        [
            "dataset_part",
            "web_scraper_order",
            "brand",
            "model",
            "brand_model",
            "trim_or_variant",
            "model_year",
            "snapshot_date",
            "target_price",
        ]
    ]


def create_business_class_proxy_dataset(
    df: pd.DataFrame,
    threshold: float = BUSINESS_CLASS_PRICE_THRESHOLD,
) -> pd.DataFrame:
    filtered = df[df["target_price"] >= threshold].copy()
    filtered["business_class_price_threshold"] = threshold
    return filtered


def preprocess_market_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, dict[str, Any]]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_tables = find_raw_tables(raw_dir)
    if not raw_tables:
        raise FileNotFoundError(
            "Raw dataset files were not found. Download the Kaggle dataset and place CSV/XLSX files into data/raw/."
        )

    profiles: dict[str, dict[str, Any]] = {}
    standardized_tables: list[pd.DataFrame] = []
    for raw_path in raw_tables:
        raw_df = normalize_columns(read_raw_table(raw_path))
        table_kind = detect_table_kind(raw_df)
        if table_kind == "used":
            prepared_df = preprocess_used_table(raw_df)
        else:
            prepared_df = preprocess_new_table(raw_df)

        output_path = processed_dir / f"{raw_path.stem}_{table_kind}_processed.csv"
        prepared_df.to_csv(output_path, index=False)
        profiles[raw_path.name] = build_profile(prepared_df, table_kind)
        standardized_tables.append(standardize_for_modeling(prepared_df, table_kind))

    combined_df = pd.concat(standardized_tables, ignore_index=True)
    combined_path = processed_dir / "car_prices_modeling_dataset.csv"
    combined_df.to_csv(combined_path, index=False)
    profiles["combined_modeling_dataset"] = build_profile(combined_df, "combined")

    business_df = create_business_class_proxy_dataset(combined_df)
    business_path = processed_dir / "business_class_and_above_price_proxy.csv"
    business_df.to_csv(business_path, index=False)
    business_profile = build_profile(business_df, "business_class_price_proxy")
    business_profile["price_threshold"] = BUSINESS_CLASS_PRICE_THRESHOLD
    profiles["business_class_and_above_price_proxy"] = business_profile

    profile_path = processed_dir / "data_profile.json"
    profile_path.write_text(json.dumps(profiles, ensure_ascii=True, indent=2), encoding="utf-8")
    return profiles


if __name__ == "__main__":
    summary = preprocess_market_dataset()
    print(json.dumps(summary, ensure_ascii=True, indent=2))
