"""Графики для README и ноутбука в едином стиле."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.config import FIGURES, MODEL_NAMES

INK, ACCENT, WARM, GREY, LIGHT = "#141A2E", "#2C52C4", "#B5631A", "#8A93A8", "#D8DEEA"
MODEL_COLORS = {
    MODEL_NAMES["ma28"]: GREY,
    MODEL_NAMES["snaive"]: "#B9C1D3",
    MODEL_NAMES["holt_winters"]: WARM,
    MODEL_NAMES["lgbm"]: ACCENT,
}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": GREY, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": LIGHT, "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)


def _pct(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}%}".replace(".", ",")


def _thousands(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def plot_sales_overview(df: pd.DataFrame):
    daily = df.groupby("date")["qty"].sum()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(daily.index, daily.values, color=LIGHT, lw=0.8, label="Продажи за день")
    ax.plot(daily.index, daily.rolling(28, center=True).mean(), color=ACCENT, lw=2, label="Среднее за 28 дней")
    ax.set_title("Продажи сети: недельный ритм, годовая сезонность и пики перед Новым годом")
    ax.set_ylabel("шт в день")
    _thousands(ax)
    ax.legend(loc="upper left")
    _save(fig, "01_sales_overview.png")


def plot_forecast_example(fc: pd.DataFrame, series_id: str):
    s = fc[fc["series_id"] == series_id].sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for _, w in s.groupby("fold"):
        ax.fill_between(w["date"], w["lgbm_p10"], w["lgbm_p90"], color=ACCENT, alpha=0.12, lw=0)
    ax.fill_between([], [], [], color=ACCENT, alpha=0.12, label="Сценарии p10-p90")
    ax.plot(s["date"], s["qty"], color=INK, lw=1.3, label="Факт")
    for i, (_, w) in enumerate(s.groupby("fold")):
        ax.plot(w["date"], w["ma28"], color=GREY, lw=1.4, ls="--", label=MODEL_NAMES["ma28"] if i == 0 else None)
        ax.plot(w["date"], w["lgbm"], color=ACCENT, lw=1.8, label=MODEL_NAMES["lgbm"] if i == 0 else None)
    for start in s.groupby("fold")["date"].min().iloc[1:]:
        ax.axvline(start, color=GREY, lw=0.8, ls=":")
    promo = s[s["promo_flag"] == 1]
    ymax = s[["qty", "lgbm_p90", "ma28"]].to_numpy().max() * 1.12
    if len(promo):
        ax.scatter(promo["date"], np.full(len(promo), ymax * 0.95), marker="s", s=10, color=WARM, label="Дни промо")
    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    ax.set_title(f"{series_id}: среднее за месяц не видит акцию, а после неё завышает прогноз")
    ax.set_ylabel("шт в день")
    ax.legend(ncol=5, loc="upper left", fontsize=8, bbox_to_anchor=(0, -0.08))
    _save(fig, "02_forecast_example.png")


def plot_wape(seg_table: pd.DataFrame, overall: pd.DataFrame):
    models = [m for m in MODEL_COLORS if m in seg_table.columns]
    table = seg_table[models].copy()
    table.loc["Все ряды"] = overall.set_index("Модель").loc[models, "WAPE, неделя x SKU x магазин"]
    order = ["Все ряды", "Стабильный", "Нерегулярный", "Прерывистый", "Комковатый"]
    table = table.loc[[i for i in order if i in table.index]]
    x = np.arange(len(table))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, m in enumerate(models):
        bars = ax.bar(x + i * width - 0.4 + width / 2, table[m], width, color=MODEL_COLORS[m], label=m)
        if m == MODEL_NAMES["lgbm"]:
            ax.bar_label(bars, labels=[_pct(v, 0) for v in table[m]], fontsize=8, padding=2, color=ACCENT)
    ax.set_xticks(x, table.index)
    ax.set_ylim(0, table.to_numpy().max() * 1.28)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("WAPE, неделя x SKU x магазин")
    ax.set_title("Ошибка прогноза: LightGBM точнее во всех сегментах спроса")
    ax.legend(ncol=4, loc="upper left", fontsize=8)
    _save(fig, "03_wape_by_model.png")


def plot_inventory(summary: pd.DataFrame):
    models = list(summary.index)
    colors = [MODEL_COLORS[m] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    panels = [
        (axes[0], "avg_stock_rub", "Средний запас, тыс руб по себестоимости", None),
        (axes[1], "avg_excess_rub", "Излишек сверх нужд цикла, тыс руб", "fill_rate"),
    ]
    for ax, col, title, extra in panels:
        vals = summary[col] / 1000
        bars = ax.barh(models, vals, color=colors)
        labels = [f"{v:,.0f}".replace(",", " ") for v in vals]
        if extra:
            labels = [f"{lab}   сервис {_pct(r)}" for lab, r in zip(labels, summary[extra])]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
        ax.set_xlim(0, vals.max() * (1.55 if extra else 1.2))
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("При одинаковой политике заказа точный прогноз держит меньше лишнего товара",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    _save(fig, "04_inventory_effect.png")


def feature_importance(model) -> pd.Series:
    imp = pd.Series(model.booster_.feature_importance("gain"), index=model.booster_.feature_name())
    return (imp / imp.sum()).sort_values(ascending=False)


FEATURE_LABELS = {
    "sku_id": "SKU", "store_id": "Магазин", "category": "Категория",
    "roll_mean_7": "Среднее за 7 дн", "roll_mean_28": "Среднее за 28 дн", "roll_mean_91": "Среднее за 91 дн",
    "roll_std_28": "Разброс за 28 дн", "dow_mean_8w": "Тот же день недели, 8 нед",
    "dow": "День недели", "month": "Месяц", "dayofyear": "День года", "weekofyear": "Неделя года",
    "promo_flag": "Промо", "discount": "Скидка", "log_price": "Цена", "price_change": "Рост регулярной цены",
    "pre_new_year": "Перед Новым годом", "new_year_holidays": "Новогодние каникулы",
    "march_8": "8 марта", "may_holidays": "Майские", "feb_23": "23 февраля",
}


def plot_importance(imp: pd.Series, top: int = 15):
    imp = imp.sort_values().tail(top)
    imp.index = [FEATURE_LABELS.get(i, i.replace("lag_", "Лаг ") + (" дн" if i.startswith("lag_") else "")) for i in imp.index]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(imp.index, imp.values, color=ACCENT)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_title("Главные признаки: уровень продаж SKU, затем промо и скидка")
    ax.grid(axis="y", visible=False)
    _save(fig, "05_feature_importance.png")


def plot_scenarios(coverage: dict):
    """Калибровка сценариев: как часто факт попадает в коридор p10-p90."""
    labels = ["Ниже p10\n(пессимистичного)", "Внутри p10-p90", "Выше p90\n(оптимистичного)"]
    fact = [coverage["Ниже p10"], coverage["Внутри p10-p90"], coverage["Выше p90"]]
    expected = [0.10, 0.80, 0.10]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(labels, fact, color=[WARM, ACCENT, WARM], width=0.55, label="Факт")
    ax.scatter(labels, expected, marker="_", s=900, color=INK, lw=2, zorder=3, label="Ожидается при идеальной калибровке")
    ax.bar_label(bars, labels=[_pct(v) for v in fact], label_type="center", fontsize=10, color="white", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, 1)
    ax.set_title(f"Коридор сценариев: факт внутри p10-p90 в {_pct(fact[1], 0)} дней при ожидаемых 80%")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    _save(fig, "06_scenarios.png")
