from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


INPUT_DATASET = Path("data/processed/business_class_and_above_price_proxy.csv")
SUMMARY_PATH = Path("data/processed/eda_summary.json")
BRAND_SUMMARY_PATH = Path("data/processed/brand_price_summary.csv")
REPORT_IMAGES_DIR = Path("report/images")


def load_dataset(path: Path = INPUT_DATASET) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["snapshot_date"])


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "unique_brands": int(df["brand"].nunique()),
        "unique_models": int(df["brand_model"].nunique()),
        "date_min": str(df["snapshot_date"].min().date()),
        "date_max": str(df["snapshot_date"].max().date()),
        "model_year_min": int(df["model_year"].min()),
        "model_year_max": int(df["model_year"].max()),
        "dataset_part_distribution": {
            key: int(value) for key, value in df["dataset_part"].value_counts().to_dict().items()
        },
        "price_quantiles": {
            str(key): float(value)
            for key, value in df["target_price"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict().items()
        },
    }


def save_summary(df: pd.DataFrame) -> None:
    summary = build_summary(df)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    brand_summary = (
        df.groupby("brand", as_index=False)
        .agg(
            n_listings=("target_price", "size"),
            median_price=("target_price", "median"),
            mean_price=("target_price", "mean"),
        )
        .sort_values(["n_listings", "median_price"], ascending=[False, False])
    )
    brand_summary.to_csv(BRAND_SUMMARY_PATH, index=False)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11


def save_price_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    sns.histplot(df["target_price"], bins=40, kde=True, ax=ax)
    ax.set_title("Price Distribution for Business Class And Above")
    ax.set_xlabel("Price (EGP)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES_DIR / "price_distribution.png", dpi=150)
    plt.close(fig)


def save_dataset_part_counts(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    counts = df["dataset_part"].value_counts().reset_index()
    counts.columns = ["dataset_part", "count"]
    sns.barplot(data=counts, x="dataset_part", y="count", hue="dataset_part", legend=False, ax=ax)
    ax.set_title("Listings by Dataset Part")
    ax.set_xlabel("Dataset Part")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES_DIR / "dataset_part_counts.png", dpi=150)
    plt.close(fig)


def save_top_brand_counts(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    top_brands = df["brand"].value_counts().head(15).reset_index()
    top_brands.columns = ["brand", "count"]
    sns.barplot(data=top_brands, x="count", y="brand", hue="brand", legend=False, ax=ax)
    ax.set_title("Top 15 Brands by Listing Count")
    ax.set_xlabel("Count")
    ax.set_ylabel("Brand")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES_DIR / "top_brand_counts.png", dpi=150)
    plt.close(fig)


def save_top_brand_median_price(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    top_brands = df["brand"].value_counts().head(15).index
    brand_prices = (
        df[df["brand"].isin(top_brands)]
        .groupby("brand", as_index=False)["target_price"]
        .median()
        .sort_values("target_price", ascending=False)
    )
    sns.barplot(data=brand_prices, x="target_price", y="brand", hue="brand", legend=False, ax=ax)
    ax.set_title("Median Price for Top 15 Brands")
    ax.set_xlabel("Median Price (EGP)")
    ax.set_ylabel("Brand")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES_DIR / "top_brand_median_price.png", dpi=150)
    plt.close(fig)


def save_price_vs_model_year(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df.sample(n=min(len(df), 2500), random_state=42)
    sns.scatterplot(
        data=sample,
        x="model_year",
        y="target_price",
        hue="dataset_part",
        alpha=0.45,
        ax=ax,
    )
    ax.set_title("Price vs Model Year")
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Price (EGP)")
    fig.tight_layout()
    fig.savefig(REPORT_IMAGES_DIR / "price_vs_model_year.png", dpi=150)
    plt.close(fig)


def run_eda() -> dict[str, object]:
    REPORT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    df = load_dataset()
    save_summary(df)
    save_price_distribution(df)
    save_dataset_part_counts(df)
    save_top_brand_counts(df)
    save_top_brand_median_price(df)
    save_price_vs_model_year(df)
    return build_summary(df)


if __name__ == "__main__":
    summary = run_eda()
    print(json.dumps(summary, ensure_ascii=True, indent=2))
