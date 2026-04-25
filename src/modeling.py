from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


SEED = 42
INPUT_DATASET = Path("data/processed/business_class_and_above_price_proxy.csv")
FEATURES_DATASET = Path("data/processed/business_class_features.csv")
RESULTS_PATH = Path("models/experiment_results.csv")
SPLIT_PATH = Path("models/split_summary.json")
BEST_MODEL_PATH = Path("models/best_model.joblib")
BEST_MODEL_INFO_PATH = Path("models/best_model_info.json")

PRIMARY_METRIC = "mae"


def load_dataset(path: Path = INPUT_DATASET) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["snapshot_date"])


def contains_keyword(series: pd.Series, keyword: str) -> pd.Series:
    return series.fillna("").str.contains(rf"\b{re.escape(keyword)}\b", case=False, regex=True).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["snapshot_year"] = prepared["snapshot_date"].dt.year
    prepared["snapshot_month"] = prepared["snapshot_date"].dt.month
    prepared["vehicle_age"] = (prepared["snapshot_year"] - prepared["model_year"]).clip(lower=0)
    prepared["variant_token_count"] = prepared["trim_or_variant"].fillna("").str.split().str.len()
    prepared["variant_char_length"] = prepared["trim_or_variant"].fillna("").str.len()
    prepared["brand_name_length"] = prepared["brand"].fillna("").str.len()
    prepared["model_name_length"] = prepared["model"].fillna("").str.len()
    prepared["is_new_listing"] = (prepared["dataset_part"] == "new").astype(int)
    prepared["has_automatic"] = contains_keyword(prepared["trim_or_variant"], "a/t")
    prepared["has_manual"] = contains_keyword(prepared["trim_or_variant"], "manual")
    prepared["has_turbo"] = contains_keyword(prepared["trim_or_variant"], "turbo")
    prepared["has_coupe"] = contains_keyword(prepared["trim_or_variant"], "coupe")
    prepared["has_suv"] = contains_keyword(prepared["trim_or_variant"], "suv")
    prepared["has_hybrid"] = contains_keyword(prepared["trim_or_variant"], "hybrid")
    prepared["has_electric"] = contains_keyword(prepared["trim_or_variant"], "electric")
    prepared["has_cabrio"] = contains_keyword(prepared["trim_or_variant"], "cabrio")
    prepared["has_hatchback"] = contains_keyword(prepared["trim_or_variant"], "hatchback")
    return prepared


def save_features_dataset(df: pd.DataFrame) -> None:
    df.to_csv(FEATURES_DATASET, index=False)


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    unique_dates = df["snapshot_date"].sort_values().drop_duplicates().reset_index(drop=True)
    train_end = unique_dates.iloc[int(len(unique_dates) * 0.70) - 1]
    val_end = unique_dates.iloc[int(len(unique_dates) * 0.85) - 1]

    train_df = df[df["snapshot_date"] <= train_end].copy()
    val_df = df[(df["snapshot_date"] > train_end) & (df["snapshot_date"] <= val_end)].copy()
    test_df = df[df["snapshot_date"] > val_end].copy()

    summary = {
        "strategy": "chronological split by snapshot_date",
        "train_end": str(train_end.date()),
        "val_end": str(val_end.date()),
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "date_ranges": {
            "train": [str(train_df["snapshot_date"].min().date()), str(train_df["snapshot_date"].max().date())],
            "val": [str(val_df["snapshot_date"].min().date()), str(val_df["snapshot_date"].max().date())],
            "test": [str(test_df["snapshot_date"].min().date()), str(test_df["snapshot_date"].max().date())],
        },
    }
    return train_df, val_df, test_df, summary


def get_feature_columns() -> tuple[list[str], list[str]]:
    categorical = ["dataset_part", "brand", "model", "brand_model"]
    numeric = [
        "model_year",
        "snapshot_year",
        "snapshot_month",
        "vehicle_age",
        "variant_token_count",
        "variant_char_length",
        "brand_name_length",
        "model_name_length",
        "is_new_listing",
        "has_automatic",
        "has_manual",
        "has_turbo",
        "has_coupe",
        "has_suv",
        "has_hybrid",
        "has_electric",
        "has_cabrio",
        "has_hatchback",
    ]
    return categorical, numeric


def build_linear_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def build_tree_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
        ]
    )


def build_models(categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    linear_preprocessor = build_linear_preprocessor(categorical, numeric)
    tree_preprocessor = build_tree_preprocessor(categorical, numeric)

    return {
        "baseline_dummy_median": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", DummyRegressor(strategy="median")),
            ]
        ),
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=18,
                        min_samples_leaf=2,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=8,
                        max_iter=300,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_and_score_models() -> tuple[pd.DataFrame, dict[str, object]]:
    raw_df = load_dataset()
    feature_df = engineer_features(raw_df)
    save_features_dataset(feature_df)

    train_df, val_df, test_df, split_summary = split_by_time(feature_df)
    categorical, numeric = get_feature_columns()
    feature_columns = categorical + numeric
    models = build_models(categorical, numeric)

    X_train = train_df[feature_columns]
    y_train = train_df["target_price"]
    X_val = val_df[feature_columns]
    y_val = val_df["target_price"]
    X_test = test_df[feature_columns]
    y_test = test_df["target_price"]

    results: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline
        val_metrics = evaluate_predictions(y_val, pipeline.predict(X_val))
        test_metrics = evaluate_predictions(y_test, pipeline.predict(X_test))
        results.append(
            {
                "model_name": model_name,
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_r2": val_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )

    results_df = pd.DataFrame(results).sort_values("val_mae").reset_index(drop=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    best_model_name = results_df.iloc[0]["model_name"]
    best_model = fitted_models[str(best_model_name)]
    joblib.dump(best_model, BEST_MODEL_PATH)

    model_info = {
        "primary_metric": PRIMARY_METRIC,
        "best_model_name": str(best_model_name),
        "feature_columns": feature_columns,
        "split_summary": split_summary,
    }
    BEST_MODEL_INFO_PATH.write_text(json.dumps(model_info, ensure_ascii=True, indent=2), encoding="utf-8")
    SPLIT_PATH.write_text(json.dumps(split_summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return results_df, model_info


if __name__ == "__main__":
    results, info = fit_and_score_models()
    print(results.to_string(index=False))
    print(json.dumps(info, ensure_ascii=True, indent=2))
