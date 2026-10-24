"""Загрузка данных, сегментация спроса и признаки для LightGBM.

Все лаги и скользящие окна сдвинуты минимум на HORIZON дней, поэтому
признаки для любого дня окна прогноза известны в момент его начала.
Цена и флаг промо входят в признаки: акции планируются заранее.
"""
import numpy as np
import pandas as pd

from src.calendar_ru import holiday_flags
from src.config import CATALOG_FILE, HORIZON, SALES_FILE

KEYS = ["store_id", "sku_id"]
LAGS = [HORIZON, HORIZON + 7, HORIZON + 14, HORIZON + 21, 364]
WINDOWS = [7, 28, 91]
CATEGORICAL = ["store_id", "sku_id", "category"]


def load_sales() -> pd.DataFrame:
    df = pd.read_csv(SALES_FILE, parse_dates=["date"])
    catalog = pd.read_csv(CATALOG_FILE)
    df = df.merge(catalog[["sku_id", "cost_ratio"]], on="sku_id", how="left")
    df["cost_price"] = df["regular_price"] * df["cost_ratio"]
    df["series_id"] = df["store_id"] + "_" + df["sku_id"]
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def demand_segments(df: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Классификация Syntetos-Boylan по истории до даты before.

    ADI: средний интервал между ненулевыми продажами.
    CV2: квадрат коэффициента вариации ненулевых продаж.
    """
    hist = df[df["date"] < before]

    def _stats(q: pd.Series) -> pd.Series:
        nz = q[q > 0]
        adi = len(q) / max(len(nz), 1)
        cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
        return pd.Series({"adi": adi, "cv2": cv2, "mean_daily": q.mean()})

    stats = hist.groupby("series_id")["qty"].apply(_stats).unstack()
    conditions = [
        (stats["adi"] < 1.32) & (stats["cv2"] < 0.49),
        (stats["adi"] < 1.32),
        (stats["cv2"] < 0.49),
    ]
    stats["segment"] = np.select(conditions, ["Стабильный", "Нерегулярный", "Прерывистый"], default="Комковатый")
    return stats.reset_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки на всей панели; строки окна прогноза используют только прошлое."""
    out = df.copy()
    g = out.groupby("series_id")["qty"]
    for lag in LAGS:
        out[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(HORIZON)
    sg = shifted.groupby(out["series_id"])
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = sg.transform(lambda s, w=w: s.rolling(w, min_periods=max(3, w // 4)).mean())
    out["roll_std_28"] = sg.transform(lambda s: s.rolling(28, min_periods=7).std())
    # Средняя доля продаж по дню недели за последние 8 недель до горизонта
    out["dow"] = out["date"].dt.dayofweek
    out["dow_mean_8w"] = (
        out.groupby(["series_id", "dow"])["qty"]
        .transform(lambda s: s.shift(HORIZON // 7 + 1).rolling(8, min_periods=2).mean())
    )

    out["month"] = out["date"].dt.month
    out["dayofyear"] = out["date"].dt.dayofyear
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    hol = holiday_flags(pd.DatetimeIndex(out["date"])).reset_index(drop=True)
    for col in hol.columns:
        out[col] = hol[col].to_numpy()

    out["discount"] = 1 - out["price"] / out["regular_price"]
    hist_price = out.groupby("series_id")["regular_price"].shift(HORIZON)
    out["price_change"] = out["regular_price"] / hist_price - 1
    out["log_price"] = np.log(out["price"])

    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    return out


FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + ["roll_std_28", "dow_mean_8w", "dow", "month", "dayofyear", "weekofyear",
       "pre_new_year", "new_year_holidays", "march_8", "may_holidays", "feb_23",
       "promo_flag", "discount", "price_change", "log_price"]
    + CATEGORICAL
)
