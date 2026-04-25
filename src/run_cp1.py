from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


DATA_PROFILE_PATH = Path("data/processed/data_profile.json")
EDA_SUMMARY_PATH = Path("data/processed/eda_summary.json")
MODEL_INFO_PATH = Path("models/best_model_info.json")


def load_profile() -> dict[str, object]:
    return json.loads(DATA_PROFILE_PATH.read_text(encoding="utf-8"))


def load_eda_summary() -> dict[str, object]:
    return json.loads(EDA_SUMMARY_PATH.read_text(encoding="utf-8"))


def load_model_info() -> dict[str, object]:
    return json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))


def print_summary(results: pd.DataFrame, threshold: float) -> None:
    profile = load_profile()
    eda_summary = load_eda_summary()
    model_info = load_model_info()
    business_profile = profile["business_class_and_above_price_proxy"]
    combined_profile = profile["combined_modeling_dataset"]
    best_test_row = results.sort_values("test_mae").iloc[0]

    print(
        "Этап 1 - Предобработка данных\n"
        "Результат: "
        f"исходный датасет {combined_profile['rows']} строк, "
        f"после price-based фильтра {business_profile['rows']} строк, "
        f"{business_profile['unique_brands']} брендов, "
        f"threshold_egp={int(threshold)}"
    )
    print(
        "Этап 2 - EDA\n"
        "Результат: "
        f"model_year_range={business_profile['model_year_min']}-{business_profile['model_year_max']}, "
        f"price_quantiles_p50={int(float(eda_summary['price_quantiles']['0.5']))}, "
        f"used={eda_summary['dataset_part_distribution']['used']}, "
        f"new={eda_summary['dataset_part_distribution']['new']}"
    )
    print()
    print(
        "Этап 3 - Моделирование и эксперименты\n"
        "Результат: "
        f"основная метрика {model_info['primary_metric']}, "
        f"лучшая по val модель {model_info['best_model_name']}, "
        f"лучшая по test_mae модель {best_test_row['model_name']}"
    )
    print(results.to_string(index=False))


def main() -> None:
    project_root = Path.cwd()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str((project_root / "data/processed/.matplotlib").resolve())
    env["XDG_CACHE_HOME"] = str((project_root / "data/processed/.cache").resolve())

    commands = [
        [sys.executable, "-m", "src.preprocessing"],
        [sys.executable, "-m", "src.eda"],
        [sys.executable, "-m", "src.modeling"],
    ]
    for command in commands:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )

    results = pd.read_csv("models/experiment_results.csv")
    profile = load_profile()
    threshold = profile["business_class_and_above_price_proxy"]["price_threshold"]
    print_summary(results, threshold)


if __name__ == "__main__":
    main()
