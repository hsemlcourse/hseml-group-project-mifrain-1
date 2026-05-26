from __future__ import annotations

from src.prediction import prepare_features


def test_prepare_features_builds_model_columns() -> None:
    feature_columns = [
        "dataset_part",
        "brand",
        "model",
        "brand_model",
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
    records = [
        {
            "dataset_part": "used",
            "brand": "BMW",
            "model": "X5",
            "trim_or_variant": "BMW X5 2020 A/T Turbo SUV",
            "model_year": 2020,
            "snapshot_date": "2023-03-27",
        }
    ]

    features = prepare_features(records, feature_columns)

    assert features.shape == (1, len(feature_columns))
    assert features.loc[0, "brand_model"] == "BMW X5"
    assert features.loc[0, "snapshot_year"] == 2023
    assert features.loc[0, "is_new_listing"] == 0
    assert features.loc[0, "has_turbo"] == 1
