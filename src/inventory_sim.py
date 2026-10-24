"""Перевод точности прогноза в деньги: симуляция пополнения запасов.

Политика одинакова для всех моделей: раз в REVIEW_DAYS дней заказ
«до уровня» S = прогноз на (LEAD_DAYS + REVIEW_DAYS) дней + страховой запас.
Страховой запас = z * RMSE модели в предыдущем окне * sqrt(L + R), то есть
более точная модель держит меньше страховки при том же целевом сервисе.
Различается только прогноз, поэтому разница в запасах объясняется точностью.
"""
import numpy as np
import pandas as pd

from src.config import HORIZON, LEAD_DAYS, MODEL_NAMES, REVIEW_DAYS, SERVICE_Z, SIM_DAYS


def _matrix(fc: pd.DataFrame, col: str) -> np.ndarray:
    """Таблица (ряд x шаг горизонта) для одного окна."""
    return fc.pivot(index="series_id", columns="step", values=col).sort_index().to_numpy(dtype=float)


def simulate_fold(fc_fold: pd.DataFrame, model: str, sigma: np.ndarray) -> dict:
    demand = _matrix(fc_fold, "qty")
    forecast = _matrix(fc_fold, model)
    cost = _matrix(fc_fold, "cost_price")
    price = _matrix(fc_fold, "price")
    level = _matrix(fc_fold, "ma28")[:, 0]
    n = demand.shape[0]
    cover = LEAD_DAYS + REVIEW_DAYS

    safety = SERVICE_Z * sigma * np.sqrt(cover)
    on_hand = np.ceil(level * (LEAD_DAYS + REVIEW_DAYS / 2))  # одинаковый стартовый остаток
    arrivals = np.zeros((n, HORIZON + LEAD_DAYS + 1))
    stock_rub, excess_rub, stock_units = [], [], []
    sold_total = lost_units = lost_rub = 0.0
    short_days = 0

    for d in range(SIM_DAYS):
        on_hand += arrivals[:, d]
        if d % REVIEW_DAYS == 0:
            position = on_hand + arrivals[:, d + 1:].sum(axis=1)
            target = forecast[:, d:d + cover].sum(axis=1) + safety
            arrivals[:, d + LEAD_DAYS] += np.maximum(0, np.ceil(target - position))
        sales = np.minimum(on_hand, demand[:, d])
        lost = demand[:, d] - sales
        on_hand -= sales

        need_next = demand[:, d + 1:d + 1 + cover].sum(axis=1)
        stock_units.append(on_hand.sum())
        stock_rub.append((on_hand * cost[:, d]).sum())
        excess_rub.append((np.maximum(0, on_hand - need_next) * cost[:, d]).sum())
        sold_total += sales.sum()
        lost_units += lost.sum()
        lost_rub += (lost * price[:, d]).sum()
        short_days += int((lost > 0).sum())

    demand_total = demand[:, :SIM_DAYS].sum()
    return {
        "avg_stock_units": np.mean(stock_units),
        "avg_stock_rub": np.mean(stock_rub),
        "avg_excess_rub": np.mean(excess_rub),
        "fill_rate": sold_total / demand_total,
        "lost_sales_rub": lost_rub,
        "stockout_day_share": short_days / (n * SIM_DAYS),
        "turnover_days": np.mean(stock_units) / (sold_total / SIM_DAYS),
    }


def _rmse_by_series(fc_fold: pd.DataFrame, model: str) -> np.ndarray:
    err = (fc_fold[model] - fc_fold["qty"]) ** 2
    return np.sqrt(err.groupby(fc_fold["series_id"]).mean().sort_index().to_numpy())


def simulate_all(fc: pd.DataFrame, models=tuple(MODEL_NAMES)) -> pd.DataFrame:
    """Окно 0 только калибрует страховой запас, результаты по окнам 1..N усредняются."""
    folds = sorted(fc["fold"].unique())
    rows = []
    for m in models:
        for prev, cur in zip(folds[:-1], folds[1:]):
            sigma = _rmse_by_series(fc[fc["fold"] == prev], m)
            res = simulate_fold(fc[fc["fold"] == cur], m, sigma)
            rows.append({"model": m, "fold": cur, **res})
    per_fold = pd.DataFrame(rows)
    summary = per_fold.groupby("model").agg(
        avg_stock_rub=("avg_stock_rub", "mean"),
        avg_excess_rub=("avg_excess_rub", "mean"),
        fill_rate=("fill_rate", "mean"),
        stockout_day_share=("stockout_day_share", "mean"),
        lost_sales_rub=("lost_sales_rub", "sum"),
        turnover_days=("turnover_days", "mean"),
    ).reindex(list(models))
    summary.index = [MODEL_NAMES[m] for m in summary.index]
    return summary
