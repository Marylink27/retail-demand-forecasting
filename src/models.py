"""Модели прогноза: два простых бейзлайна, Holt-Winters и глобальный LightGBM.

Каждая функция получает историю до даты origin и возвращает прогноз
на HORIZON дней вперёд для всех рядов (магазин x SKU).
"""
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import HORIZON, SEED
from src.features import FEATURES


def _future_index(df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    end = origin + pd.Timedelta(days=HORIZON - 1)
    return df.loc[(df["date"] >= origin) & (df["date"] <= end), ["series_id", "date"]].copy()


def forecast_ma28(df: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Скользящее среднее за 28 дней: типичный ручной заказ «как в прошлом месяце»."""
    hist = df[(df["date"] < origin) & (df["date"] >= origin - pd.Timedelta(days=28))]
    level = hist.groupby("series_id")["qty"].mean()
    fut = _future_index(df, origin)
    return pd.Series(fut["series_id"].map(level).to_numpy(), index=fut.index)


def forecast_snaive(df: pd.DataFrame, origin: pd.Timestamp) -> pd.Series:
    """Сезонный наивный прогноз: среднее того же дня недели за последние 4 недели."""
    hist = df[(df["date"] < origin) & (df["date"] >= origin - pd.Timedelta(days=28))]
    level = hist.groupby(["series_id", hist["date"].dt.dayofweek])["qty"].mean()
    fut = _future_index(df, origin)
    keys = pd.MultiIndex.from_arrays([fut["series_id"], fut["date"].dt.dayofweek])
    return pd.Series(level.reindex(keys).to_numpy(), index=fut.index)


def _hw_one(y: np.ndarray) -> np.ndarray:
    """Holt-Winters с аддитивной недельной сезонностью по последнему году истории."""
    if (y > 0).sum() < 30:
        return np.repeat(y[-28:].mean(), HORIZON)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                initialization_method="estimated",
            ).fit(optimized=True)
            return np.clip(fit.forecast(HORIZON), 0, None)
        except Exception:
            return np.repeat(y[-28:].mean(), HORIZON)


def forecast_holt_winters(df: pd.DataFrame, origin: pd.Timestamp, n_jobs: int = -1) -> pd.Series:
    hist = df[(df["date"] < origin) & (df["date"] >= origin - pd.Timedelta(days=364))]
    series = {sid: g["qty"].to_numpy(dtype=float) for sid, g in hist.groupby("series_id")}
    preds = Parallel(n_jobs=n_jobs)(delayed(_hw_one)(y) for y in series.values())
    pred_map = dict(zip(series.keys(), preds))

    fut = _future_index(df, origin)
    step = (fut["date"] - origin).dt.days.to_numpy()
    values = np.array([pred_map[s][h] for s, h in zip(fut["series_id"], step)])
    return pd.Series(values, index=fut.index)


LGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)


def _train_rows(feat: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    train = feat[(feat["date"] < origin) & feat["lag_364"].notna()]
    # Дни с нулевым остатком исключаются: продажи там занижены дефицитом, а не спросом
    return train[train["stock_end"] > 0]


def fit_lgbm(feat: pd.DataFrame, origin: pd.Timestamp, objective: str = "tweedie", alpha: float | None = None):
    train = _train_rows(feat, origin)
    params = dict(LGB_PARAMS, objective=objective)
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.2
    if objective == "quantile":
        params["alpha"] = alpha
    model = lgb.LGBMRegressor(**params)
    model.fit(train[FEATURES], train["qty"])
    return model


def forecast_lgbm(feat: pd.DataFrame, origin: pd.Timestamp, quantiles=(0.1, 0.5, 0.9)):
    """Точечный прогноз (tweedie) и квантили для трёх сценариев спроса."""
    fut_idx = _future_index(feat, origin).index
    X = feat.loc[fut_idx, FEATURES]
    point_model = fit_lgbm(feat, origin)
    result = pd.DataFrame(index=fut_idx)
    result["lgbm"] = np.clip(point_model.predict(X), 0, None)
    for q in quantiles:
        m = fit_lgbm(feat, origin, objective="quantile", alpha=q)
        result[f"lgbm_p{int(q * 100)}"] = np.clip(m.predict(X), 0, None)
    # Квантили не должны пересекаться
    qcols = [f"lgbm_p{int(q * 100)}" for q in quantiles]
    result[qcols] = np.sort(result[qcols].to_numpy(), axis=1)
    return result, point_model
