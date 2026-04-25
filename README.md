# ML Project — Предсказание цен автомобилей бизнес-класса и выше

**Студент:** Гилядов Борис Равашиевич

**Группа:** БИВ232


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Регрессия

**Датасет:** [Car Prices Market](https://www.kaggle.com/datasets/muhammedzidan/car-prices-market)

**Целевая метрика:** `MAE`. Основная метрика выбрана из-за тяжёлого правого хвоста распределения цен и наличия дорогих выбросов, для которых `RMSE` слишком чувствительна. Дополнительно считаются `RMSE` и `R2`.

**Определение сегмента:** в рамках CP1 сегмент `business class and above` задаётся ценовым порогом `target_price >= 350000 EGP`.

**Текущее состояние данных после первичной обработки:** `78,612` строк по рынку подержанных автомобилей и `3,433` строк по рынку новых автомобилей. После price-based фильтра остаётся `11,107` строк.


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные данные, summary и датасеты для моделирования
│   └── raw                     # Исходные CSV из Kaggle
├── models                      # Результаты экспериментов и служебные model-артефакты
├── notebooks                   # Не используются на этапе CP1, пайплайн реализован кодом в src/
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Графики EDA
│   └── report.md               # Отчёт по проекту
├── src
│   ├── preprocessing.py        # Предобработка и сбор итогового датасета
│   ├── eda.py                  # EDA и сохранение визуализаций
│   ├── modeling.py             # Обучение baseline и экспериментальных моделей
│   └── run_cp1.py              # Единый запуск полного пайплайна CP1
├── tests
│   ├── conftest.py             # Конфигурация путей для тестов
│   └── test_preprocessing.py   # Тесты пайплайна предобработки
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```

Скачать raw-данные с Kaggle и положить исходные CSV/XLSX-файлы в [`data/raw/`](data/raw).

После этого можно прогнать предобработку:

```bash
source .venv/bin/activate
python3 -m src.run_cp1
```

Минимальный запуск через Docker:

```bash
docker compose up --build
```

Для Docker-запуска raw CSV-файлы также должны лежать в `data/raw/`.

Что делает `python3 -m src.run_cp1`:
- `Этап 1 — preprocessing` -> очищает raw-данные, приводит цены и даты к нормальному виду, собирает итоговые таблицы в `data/processed/`
- `Этап 2 — eda` -> считает summary по итоговому датасету и сохраняет графики в `report/images/`
- `Этап 3 — modeling` -> обучает baseline и экспериментальные модели, сохраняет результаты в `models/`
- `Этап 4 — console summary` -> печатает краткую сводку по датасету и итоговую таблицу метрик моделей

Проверка качества кода:

```bash
source .venv/bin/activate
pytest -q
flake8 src tests --max-line-length=120
```

## Данные
- `data/raw/` — исходные файлы из Kaggle
- `data/processed/` — предобработанные таблицы и профиль качества данных
- `data/processed/data_profile.json` — сводка по строкам, колонкам, пропускам и диапазону лет
- `data/processed/business_class_and_above_price_proxy.csv` — итоговый срез для CP1
- `data/processed/business_class_features.csv` — модельный датасет после feature engineering

**Фактические размеры:**
- исходный объединённый датасет: `82,045` строк
- итоговый price-based срез: `11,107` строк
- брендов в итоговом срезе: `54`
- диапазон модельных лет: `2003–2023`

**Фичи в модельном датасете:**
- категориальные: `dataset_part`, `brand`, `model`, `brand_model`
- числовые и бинарные: `model_year`, `snapshot_year`, `snapshot_month`, `vehicle_age`, `variant_token_count`, `variant_char_length`, `brand_name_length`, `model_name_length`, `is_new_listing`, `has_automatic`, `has_manual`, `has_turbo`, `has_coupe`, `has_suv`, `has_hybrid`, `has_electric`, `has_cabrio`, `has_hatchback`

Итого модель использует `22` признака, что закрывает формальный критерий по количеству колонок.

## EDA
- summary: [`data/processed/eda_summary.json`](data/processed/eda_summary.json)
- графики:
  - [`report/images/price_distribution.png`](report/images/price_distribution.png)
  - [`report/images/dataset_part_counts.png`](report/images/dataset_part_counts.png)
  - [`report/images/top_brand_counts.png`](report/images/top_brand_counts.png)
  - [`report/images/top_brand_median_price.png`](report/images/top_brand_median_price.png)
  - [`report/images/price_vs_model_year.png`](report/images/price_vs_model_year.png)

**Ключевые наблюдения:**
- в итоговом срезе `8301` объявление из `used` части и `2806` из `new`
- медианная цена в итоговом срезе: `514000 EGP`
- 90-й перцентиль цены: `990000 EGP`
- датасет остаётся смешанным по брендам, поэтому сегмент трактуется как ценовой, а не брендовый

## Моделирование
Для защиты от leakage используется **хронологический split по `snapshot_date`**:
- `train`: до `2022-09-06`, `7160` строк
- `val`: `2022-09-08` — `2022-12-18`, `1915` строк
- `test`: `2022-12-19` — `2023-03-27`, `2032` строки

Артефакты:
- [`models/experiment_results.csv`](models/experiment_results.csv)
- [`models/split_summary.json`](models/split_summary.json)
- [`models/best_model.joblib`](models/best_model.joblib)
- [`models/best_model_info.json`](models/best_model_info.json)

После запуска `python3 -m src.run_cp1` в консоль выводится краткая сводка по датасету и таблица метрик моделей. Для сверки с результатами в `README` нужно смотреть:
- `business_rows`
- `brands`
- `threshold_egp`
- `model_year_range`
- таблицу моделей с `val_mae`, `test_mae`, `test_rmse`, `test_r2`

Формат консольного вывода:
- `Этап 1 - Предобработка данных`
- `Результат: исходный датасет ..., после price-based фильтра ..., threshold_egp=...`
- `Этап 2 - EDA`
- `Результат: model_year_range=..., price_quantiles_p50=..., used=..., new=...`
- `Этап 3 - Моделирование и эксперименты`
- `Результат: основная метрика ..., лучшая по val модель ..., лучшая по test_mae модель ...`
- затем таблица моделей


## Результаты
| Модель | Val MAE | Test MAE | Test RMSE | Test R2 |
|--------|---------|----------|-----------|---------|
| DummyRegressor | 311676.60 | 337904.99 | 695838.28 | -0.2108 |
| LinearRegression | 204111.93 | 289671.40 | 449685.54 | 0.4943 |
| Ridge | 222610.47 | 299673.30 | 520521.87 | 0.3224 |
| RandomForestRegressor | 177057.95 | 291184.35 | 495961.01 | 0.3849 |
| HistGradientBoostingRegressor | 172821.99 | 354368.96 | 583415.68 | 0.1488 |

По валидационной `MAE` лучшей стала `HistGradientBoostingRegressor`, но на тесте наиболее устойчивый результат показала `LinearRegression`. Для CP1 это фиксируется как важный вывод: простая линейная модель уже даёт сильный baseline, а более сложные ансамбли требуют дополнительной настройки на CP2.


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
