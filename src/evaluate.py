"""Бэктест со скользящим началом окна и метрики точности."""
import numpy as np
import pandas as pd

from src.config import END_DATE, HORIZON, MODEL_NAMES, N_FOLDS
from src.features import build_features, demand_segments
from src.models import forecast_holt_winters, forecast_lgbm, forecast_ma28, forecast_snaive


def fold_origins(n_folds: int = N_FOLDS) -> list[pd.Timestamp]:
    """Начала окон: последнее окно заканчивается датой END_DATE, окна не пересекаются."""
    last_origin = pd.Timestamp(END_DATE) - pd.Timedelta(days=HORIZON - 1)
    return [last_origin - pd.Timedelta(days=HORIZON * k) for k in reversed(range(n_folds))]


def run_backtest(df: pd.DataFrame, verbose: bool = True):
    """Прогнозы всех моделей по всем окнам и модель LightGBM последнего окна."""
    feat = build_features(df)
    parts, last_model = [], None
    for fold, origin in enumerate(fold_origins()):
        if verbose:
            print(f"Окно {fold}: прогноз с {origin.date()} на {HORIZON} дней")
        lgbm_pred, last_model = forecast_lgbm(feat, origin)
        part = df.loc[lgbm_pred.index, ["series_id", "store_id", "sku_id", "category", "date",
                                         "qty", "price", "cost_price", "promo_flag"]].copy()
        part["fold"] = fold
        part["origin"] = origin
        part["step"] = (part["date"] - origin).dt.days
        part["ma28"] = forecast_ma28(df, origin)
        part["snaive"] = forecast_snaive(df, origin)
        part["holt_winters"] = forecast_holt_winters(df, origin)
        part = part.join(lgbm_pred)
        seg = demand_segments(df, origin)[["series_id", "segment"]]
        part = part.merge(seg, on="series_id", how="left").set_index(part.index)
        parts.append(part)
    return pd.concat(parts), last_model


def wape(y, f) -> float:
    return float(np.abs(np.asarray(y) - np.asarray(f)).sum() / np.asarray(y).sum())


def bias(y, f) -> float:
    return float((np.asarray(f) - np.asarray(y)).sum() / np.asarray(y).sum())


def accuracy_table(fc: pd.DataFrame, models=tuple(MODEL_NAMES)) -> pd.DataFrame:
    """Точность по уровням агрегации: день x SKU x магазин, неделя x SKU x магазин, выручка магазина за неделю."""
    fc = fc.copy()
    fc["week"] = fc["fold"].astype(str) + "_" + (fc["step"] // 7).astype(str)
    rows = []
    for m in models:
        weekly = fc.groupby(["series_id", "week"])[["qty", m]].sum()
        fc["rev_fact"] = fc["qty"] * fc["price"]
        fc["rev_fc"] = fc[m] * fc["price"]
        rev = fc.groupby(["store_id", "week"])[["rev_fact", "rev_fc"]].sum()
        rows.append({
            "Модель": MODEL_NAMES[m],
            "WAPE, день x SKU x магазин": wape(fc["qty"], fc[m]),
            "WAPE, неделя x SKU x магазин": wape(weekly["qty"], weekly[m]),
            "WAPE выручки, магазин x неделя": wape(rev["rev_fact"], rev["rev_fc"]),
            "Смещение (bias)": bias(fc["qty"], fc[m]),
            "MAE, шт в день": float(np.abs(fc["qty"] - fc[m]).mean()),
        })
    return pd.DataFrame(rows)


def wape_by_segment(fc: pd.DataFrame, models=tuple(MODEL_NAMES)) -> pd.DataFrame:
    """WAPE на недельном уровне SKU x магазин по сегментам спроса."""
    fc = fc.copy()
    fc["week"] = fc["fold"].astype(str) + "_" + (fc["step"] // 7).astype(str)
    weekly = fc.groupby(["series_id", "segment", "week"])[["qty", *models]].sum().reset_index()
    out = {MODEL_NAMES[m]: weekly.groupby("segment").apply(lambda g, m=m: wape(g["qty"], g[m]), include_groups=False)
           for m in models}
    table = pd.DataFrame(out)
    # Сегмент пересчитывается в начале каждого окна, поэтому ряд может переходить между сегментами
    table["Пар ряд x окно"] = fc[["segment", "series_id", "fold"]].drop_duplicates().groupby("segment").size()
    return table


def scenario_coverage(fc: pd.DataFrame) -> dict:
    """Доля фактических продаж внутри интервала p10-p90 и доля ниже p10."""
    inside = (fc["qty"] >= fc["lgbm_p10"]) & (fc["qty"] <= fc["lgbm_p90"])
    return {"Внутри p10-p90": float(inside.mean()), "Ниже p10": float((fc["qty"] < fc["lgbm_p10"]).mean()),
            "Выше p90": float((fc["qty"] > fc["lgbm_p90"]).mean())}
