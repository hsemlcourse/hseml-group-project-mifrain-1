from __future__ import annotations

import pandas as pd

from src.preprocessing import (
    BUSINESS_CLASS_PRICE_THRESHOLD,
    create_business_class_proxy_dataset,
    detect_table_kind,
    normalize_columns,
    parse_price,
    parse_price_change,
    preprocess_new_table,
    preprocess_used_table,
)


def test_parse_price_handles_plain_values_and_ranges() -> None:
    assert parse_price("1,250,000 EGP") == 1250000.0
    assert parse_price("1,400,000 EGP To 1,420,000 EGP") == 1410000.0
    assert parse_price("error Car not available") is None


def test_parse_price_change_handles_trend_direction() -> None:
    assert parse_price_change("trending_up +50,000 EGP") == 50000.0
    assert parse_price_change("trending_down -20,100 EGP") == -20100.0


def test_detect_table_kind_supports_used_and_new_layouts() -> None:
    used_df = normalize_columns(
        pd.DataFrame(
            columns=[
                "web-scraper-order",
                "Car Model",
                "Month/Year",
                "Average price",
                "Minimum price",
                "Maximum price",
            ]
        )
    )
    new_df = normalize_columns(
        pd.DataFrame(
            columns=[
                "Car Model",
                "Old Price",
                "Price Change",
                "New Price",
                "date_range",
            ]
        )
    )

    assert detect_table_kind(used_df) == "used"
    assert detect_table_kind(new_df) == "new"


def test_preprocess_used_table_builds_numeric_columns() -> None:
    raw_df = pd.DataFrame(
        {
            "web-scraper-order": ["1"],
            "Car Model": ["BMW 520i 2021"],
            "Month/Year": ["2023-03"],
            "Average price": ["2,000,000 EGP"],
            "Minimum price": ["1,900,000 EGP"],
            "Maximum price": ["2,150,000 EGP"],
        }
    )

    prepared = preprocess_used_table(raw_df)

    assert prepared.loc[0, "brand"] == "BMW"
    assert prepared.loc[0, "average_price"] == 2000000.0
    assert prepared.loc[0, "model_year"] == 2021


def test_preprocess_new_table_extracts_brand_year_and_prices() -> None:
    raw_df = pd.DataFrame(
        {
            "Car Model": ["BMW X5 xDrive40i 2024"],
            "Old Price": ["5,000,000 EGP"],
            "Price Change": ["trending_up +150,000 EGP"],
            "New Price": ["5,150,000 EGP"],
            "date_range": ["2024-05-01"],
        }
    )

    prepared = preprocess_new_table(raw_df)

    assert prepared.loc[0, "brand"] == "BMW"
    assert prepared.loc[0, "model_year"] == 2024
    assert prepared.loc[0, "old_price"] == 5000000.0
    assert prepared.loc[0, "price_change"] == 150000.0


def test_create_business_class_proxy_dataset_filters_by_threshold() -> None:
    df = pd.DataFrame(
        {
            "target_price": [BUSINESS_CLASS_PRICE_THRESHOLD - 1, BUSINESS_CLASS_PRICE_THRESHOLD, 700000.0],
            "brand": ["A", "B", "C"],
        }
    )

    filtered = create_business_class_proxy_dataset(df)

    assert len(filtered) == 2
    assert filtered["target_price"].min() >= BUSINESS_CLASS_PRICE_THRESHOLD
