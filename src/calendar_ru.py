"""Российский торговый календарь: праздники, которые заметно двигают спрос."""
import numpy as np
import pandas as pd


def holiday_flags(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Флаги праздничных периодов для каждой даты."""
    d = pd.DatetimeIndex(dates)
    month, day = d.month, d.day
    flags = pd.DataFrame(index=d)
    flags["pre_new_year"] = ((month == 12) & (day >= 24)).astype(int)
    flags["new_year_holidays"] = ((month == 1) & (day <= 8)).astype(int)
    flags["march_8"] = ((month == 3) & (day >= 5) & (day <= 8)).astype(int)
    flags["may_holidays"] = ((month == 5) & (day <= 10)).astype(int)
    flags["feb_23"] = ((month == 2) & (day >= 20) & (day <= 23)).astype(int)
    return flags


def holiday_multiplier(dates: pd.DatetimeIndex) -> np.ndarray:
    """Множитель спроса от праздников, используется генератором данных."""
    f = holiday_flags(dates)
    mult = np.ones(len(f))
    mult *= np.where(f["pre_new_year"] == 1, 1.6, 1.0)
    mult *= np.where(f["new_year_holidays"] == 1, 0.75, 1.0)
    mult *= np.where(f["march_8"] == 1, 1.3, 1.0)
    mult *= np.where(f["may_holidays"] == 1, 1.2, 1.0)
    mult *= np.where(f["feb_23"] == 1, 1.1, 1.0)
    return mult
