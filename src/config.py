"""Общие настройки проекта: пути, горизонт прогноза, параметры пополнения."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
SALES_FILE = DATA_RAW / "sales_daily.csv.gz"
CATALOG_FILE = DATA_RAW / "sku_catalog.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42
START_DATE = "2024-01-01"
END_DATE = "2026-06-30"

# Горизонт прогноза в днях. Все лаги в модели >= HORIZON, поэтому
# прогноз на весь горизонт строится в момент начала окна без утечки данных.
HORIZON = 35
N_FOLDS = 5  # первое окно служит прогревом: по его ошибкам считается страховой запас

# Политика пополнения: периодический пересмотр раз в неделю, поставка через 2 дня
REVIEW_DAYS = 7
LEAD_DAYS = 2
SIM_DAYS = 28
SERVICE_Z = 1.65  # целевой уровень сервиса около 95%

MODEL_NAMES = {
    "ma28": "Скользящее среднее 28 дн",
    "snaive": "Сезонный наивный",
    "holt_winters": "Holt-Winters",
    "lgbm": "LightGBM",
}
