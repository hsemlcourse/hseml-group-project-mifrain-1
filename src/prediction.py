from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.modeling import engineer_features
from src.preprocessing import make_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_MODEL_INFO_PATH = PROJECT_ROOT / "models" / "best_model_info.json"


@dataclass(frozen=True)
class CarPrediction:
    predicted_price_egp: float
    model_name: str
    primary_metric: str


class CarPricePredictor:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        model_info_path: Path = DEFAULT_MODEL_INFO_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact was not found at {model_path}. "
                "Run `python3 -m src.run_cp1` or add models/best_model.joblib."
            )
        if not model_info_path.exists():
            raise FileNotFoundError(
                f"Model metadata was not found at {model_info_path}. "
                "Run `python3 -m src.run_cp1` first."
            )

        self.model = joblib.load(model_path)
        self.model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
        self.feature_columns = list(self.model_info["feature_columns"])
        self.model_name = str(self.model_info["best_model_name"])
        self.primary_metric = str(self.model_info["primary_metric"])

    def predict_one(self, record: dict[str, Any]) -> CarPrediction:
        return self.predict_many([record])[0]

    def predict_many(self, records: list[dict[str, Any]]) -> list[CarPrediction]:
        features = prepare_features(records, self.feature_columns)
        predictions = self.model.predict(features)
        return [
            CarPrediction(
                predicted_price_egp=float(round(prediction, 2)),
                model_name=self.model_name,
                primary_metric=self.primary_metric,
            )
            for prediction in predictions
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "primary_metric": self.primary_metric,
            "feature_columns": self.feature_columns,
            "split_summary": self.model_info.get("split_summary", {}),
        }


def prepare_features(records: list[dict[str, Any]], feature_columns: list[str]) -> pd.DataFrame:
    if not records:
        raise ValueError("At least one record is required for prediction.")

    df = pd.DataFrame(records).copy()
    required_base_columns = [
        "dataset_part",
        "brand",
        "model",
        "trim_or_variant",
        "model_year",
        "snapshot_date",
    ]
    missing_base_columns = [column for column in required_base_columns if column not in df.columns]
    if missing_base_columns:
        raise ValueError(f"Missing input columns: {missing_base_columns}")

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    if df["snapshot_date"].isna().any():
        raise ValueError("snapshot_date must be a valid date.")

    df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
    if df["model_year"].isna().any():
        raise ValueError("model_year must be numeric.")

    df = make_model_features(df)
    feature_df = engineer_features(df)
    missing_feature_columns = [column for column in feature_columns if column not in feature_df.columns]
    if missing_feature_columns:
        raise ValueError(f"Missing engineered columns: {missing_feature_columns}")

    return feature_df[feature_columns]
