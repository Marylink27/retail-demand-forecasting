"""Генератор синтетической выгрузки продаж в структуре отчёта 1С.

Данные повторяют устройство реальной розницы: недельная и годовая
сезонность, праздники, рост цен, промоакции, тренд, разная скорость
продаж и дефициты на полке (продажи в такие дни занижены, спрос цензурирован).
Коммерческие данные заказчиков конфиденциальны, поэтому в репозитории
используется генератор с фиксированным seed.
"""
import numpy as np
import pandas as pd

from src.calendar_ru import holiday_multiplier
from src.config import CATALOG_FILE, DATA_RAW, END_DATE, SALES_FILE, SEED, START_DATE

CATEGORIES = {
    # категория: (пик сезона по дню года, амплитуда годовой сезонности, рост в выходные)
    "Напитки": (200, 0.35, 1.25),
    "Снеки": (355, 0.20, 1.30),
    "Бакалея": (30, 0.10, 1.10),
    "Молочная продукция": (120, 0.08, 1.15),
    "Бытовая химия": (90, 0.05, 1.20),
}

SKU_NAMES = {
    "Напитки": ["Лимонад 0,5 л", "Холодный чай 1 л", "Минеральная вода 1,5 л", "Сок яблочный 1 л",
                "Квас 1,5 л", "Морс клюквенный 1 л", "Энергетик 0,45 л", "Газировка кола 0,5 л"],
    "Снеки": ["Чипсы 150 г", "Орехи ассорти 100 г", "Сухарики 80 г", "Попкорн 90 г",
              "Шоколадный батончик", "Печенье овсяное 300 г", "Мармелад 200 г", "Вафли 220 г"],
    "Бакалея": ["Макароны 450 г", "Рис круглый 900 г", "Гречка 800 г", "Масло подсолнечное 1 л",
                "Сахар 1 кг", "Мука 2 кг", "Кофе молотый 250 г", "Чай чёрный 100 пак"],
    "Молочная продукция": ["Молоко 2,5% 1 л", "Кефир 1% 900 г", "Йогурт питьевой 270 г", "Творог 5% 300 г",
                           "Сыр полутвёрдый 200 г", "Сметана 15% 300 г", "Масло сливочное 180 г", "Ряженка 500 г"],
    "Бытовая химия": ["Средство для посуды 500 мл", "Стиральный порошок 3 кг", "Гель для стирки 1,3 л",
                      "Чистящее средство 500 мл", "Губки для посуды 5 шт", "Мешки для мусора 30 шт",
                      "Освежитель воздуха", "Капсулы для стирки 15 шт"],
}

N_STORES = 8


def _round_to_tens(x):
    return np.maximum(10, np.round(np.asarray(x) / 10) * 10)


def build_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """Справочник SKU: категория, базовый спрос, цена, себестоимость, эластичность."""
    rows = []
    for cat, names in SKU_NAMES.items():
        for name in names:
            rows.append({"category": cat, "sku_name": name})
    cat_df = pd.DataFrame(rows)
    n = len(cat_df)
    cat_df.insert(0, "sku_id", [f"SKU{1000 + i}" for i in range(n)])

    # Три скорости продаж: быстрые, средние и медленные (прерывистый спрос)
    speed = rng.choice(["fast", "medium", "slow"], size=n, p=[0.3, 0.4, 0.3])
    base = np.where(speed == "fast", rng.uniform(12, 35, n),
                    np.where(speed == "medium", rng.uniform(3, 9, n), rng.uniform(0.25, 1.4, n)))
    cat_df["speed_class"] = speed
    cat_df["base_demand"] = base.round(3)
    cat_df["base_price"] = _round_to_tens(rng.lognormal(np.log(160), 0.6, n))
    cat_df["cost_ratio"] = rng.uniform(0.62, 0.74, n).round(3)
    cat_df["elasticity"] = rng.uniform(1.2, 2.6, n).round(2)
    cat_df["trend_per_year"] = rng.normal(0.04, 0.10, n).round(3)
    return cat_df


def _price_paths(catalog, dates, rng):
    """Регулярная цена с плановыми повышениями и промо-периоды по сети."""
    n_sku, n_days = len(catalog), len(dates)
    regular = np.repeat(catalog["base_price"].to_numpy()[:, None], n_days, axis=1).astype(float)
    for k in range(n_sku):
        for _ in range(rng.integers(1, 4)):
            start = rng.integers(60, n_days - 30)
            regular[k, start:] *= rng.uniform(1.04, 1.10)
    regular = _round_to_tens(regular)

    promo = np.zeros((n_sku, n_days), dtype=int)
    discount = np.zeros((n_sku, n_days))
    for k in range(n_sku):
        n_promos = rng.poisson(5 * n_days / 365)
        for _ in range(n_promos):
            start = rng.integers(0, n_days - 14)
            length = rng.choice([7, 14])
            promo[k, start:start + length] = 1
            discount[k, start:start + length] = rng.uniform(0.15, 0.30)
    price = _round_to_tens(regular * (1 - discount))
    return regular, price, promo


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n_days = len(dates)
    catalog = build_catalog(rng)
    n_sku = len(catalog)

    store_mult = rng.uniform(0.6, 1.6, N_STORES)
    regular, price, promo = _price_paths(catalog, dates, rng)

    # Детерминированная часть спроса, форма (sku, день)
    doy = np.asarray(dates.dayofyear)
    weekend = np.asarray(dates.dayofweek >= 4)  # пятница, суббота, воскресенье
    t_years = np.arange(n_days) / 365.0
    hol = holiday_multiplier(dates)

    lam_sku = np.empty((n_sku, n_days))
    for k, row in catalog.iterrows():
        peak, amp, wk = CATEGORIES[row["category"]]
        yearly = 1 + amp * np.cos(2 * np.pi * (doy - peak) / 365.25)
        weekly = np.where(weekend, wk, 1.0)
        trend = (1 + row["trend_per_year"]) ** t_years
        price_eff = (price[k] / regular[k]) ** (-row["elasticity"])
        promo_eff = np.where(promo[k] == 1, 1.25, 1.0)  # выкладка и ценники акции
        lam_sku[k] = row["base_demand"] * yearly * weekly * trend * hol * price_eff * promo_eff

    # Истинный спрос с гамма-пуассоновским шумом, форма (sku, магазин, день)
    lam = lam_sku[:, None, :] * store_mult[None, :, None]
    shape = 4.0
    demand = rng.poisson(lam * rng.gamma(shape, 1 / shape, size=lam.shape))

    # История остатков: еженедельный заказ «на глаз» по среднему, бывают срывы поставок
    sales = np.zeros_like(demand)
    stock_end = np.zeros_like(demand)
    on_hand = np.ceil(lam[:, :, 0] * 10).astype(int)
    pipeline = np.zeros((n_sku, N_STORES, n_days + 10), dtype=int)
    order_dow = rng.integers(0, 7, N_STORES)
    for d in range(n_days):
        on_hand += pipeline[:, :, d]
        sold = np.minimum(on_hand, demand[:, :, d])
        sales[:, :, d] = sold
        on_hand -= sold
        stock_end[:, :, d] = on_hand
        is_order_day = dates[d].dayofweek == order_dow
        if is_order_day.any():
            lo = max(0, d - 27)
            ma = sales[:, :, lo:d + 1].mean(axis=2)
            target = (ma + 0.3) * 13 * rng.uniform(0.8, 1.35, size=ma.shape)
            qty = np.maximum(0, np.ceil(target - on_hand)).astype(int)
            failed = rng.random(ma.shape) < 0.05  # поставщик не привёз заказ
            qty = np.where(is_order_day[None, :] & ~failed, qty, 0)
            pipeline[:, :, d + 2] += qty

    sku_idx, store_idx, day_idx = np.meshgrid(np.arange(n_sku), np.arange(N_STORES), np.arange(n_days), indexing="ij")
    df = pd.DataFrame({
        "date": dates[day_idx.ravel()],
        "store_id": "M" + pd.Series(store_idx.ravel() + 1).astype(str),
        "sku_id": catalog["sku_id"].to_numpy()[sku_idx.ravel()],
        "category": catalog["category"].to_numpy()[sku_idx.ravel()],
        "qty": sales.ravel(),
        "price": price[sku_idx.ravel(), day_idx.ravel()].astype(int),
        "regular_price": regular[sku_idx.ravel(), day_idx.ravel()].astype(int),
        "promo_flag": promo[sku_idx.ravel(), day_idx.ravel()],
        "stock_end": stock_end.ravel(),
    })
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    return df, catalog


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    sales, catalog = generate()
    sales.to_csv(SALES_FILE, index=False, compression="gzip")
    catalog.to_csv(CATALOG_FILE, index=False)
    print(f"Сохранено {len(sales):,} строк продаж и {len(catalog)} SKU в {DATA_RAW}")


if __name__ == "__main__":
    main()
