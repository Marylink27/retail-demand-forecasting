"""Полный прогон: данные, бэктест, метрики, симуляция запасов, графики.

Запуск:  python -m src.run_pipeline
Быстрый пересчёт отчётов по сохранённым прогнозам:  python -m src.run_pipeline --reuse
"""
import argparse

import pandas as pd

from src import generate_data, plots
from src.config import FIGURES, REPORTS, SALES_FILE
from src.evaluate import accuracy_table, run_backtest, scenario_coverage, wape_by_segment
from src.features import load_sales
from src.inventory_sim import simulate_all

FORECASTS_FILE = REPORTS / "backtest_forecasts.csv.gz"
IMPORTANCE_FILE = REPORTS / "feature_importance.csv"


def load_backtest() -> tuple[pd.DataFrame, pd.Series]:
    fc = pd.read_csv(FORECASTS_FILE, parse_dates=["date", "origin"])
    imp = pd.read_csv(IMPORTANCE_FILE, index_col=0)["gain_share"]
    return fc, imp


def pick_example_series(fc: pd.DataFrame) -> str:
    """Самый продаваемый регулярный ряд, у которого в последних двух окнах была акция."""
    last = fc[fc["fold"] >= fc["fold"].max() - 1]
    regular = last[last["segment"].isin(["Стабильный", "Нерегулярный"])]
    stats = regular.groupby("series_id").agg(qty=("qty", "sum"), promo=("promo_flag", "sum"))
    return stats[stats["promo"] > 0]["qty"].idxmax()


def build_reports(df: pd.DataFrame, fc: pd.DataFrame, imp: pd.Series) -> dict:
    fc_eval = fc[fc["fold"] > 0]  # окно 0 только калибрует страховой запас
    res = {
        "accuracy": accuracy_table(fc_eval),
        "segments": wape_by_segment(fc_eval),
        "inventory": simulate_all(fc),
        "coverage": scenario_coverage(fc_eval),
    }
    res["accuracy"].to_csv(REPORTS / "metrics_accuracy.csv", index=False)
    res["segments"].to_csv(REPORTS / "metrics_by_segment.csv")
    res["inventory"].to_csv(REPORTS / "inventory_effect.csv")
    pd.Series(res["coverage"]).to_csv(REPORTS / "scenario_coverage.csv", header=["share"])

    plots.set_style()
    plots.plot_sales_overview(df)
    plots.plot_forecast_example(fc[fc["fold"] >= fc["fold"].max() - 1], pick_example_series(fc))
    plots.plot_wape(res["segments"], res["accuracy"])
    plots.plot_inventory(res["inventory"])
    plots.plot_importance(imp)
    plots.plot_scenarios(res["coverage"])
    return res


def print_summary(res: dict):
    pd.set_option("display.width", 160)
    print("\nТочность прогноза (окна 1-4):\n", res["accuracy"].round(3).to_string(index=False))
    print("\nWAPE по сегментам:\n", res["segments"].round(3).to_string())
    print("\nСценарии:", {k: round(v, 3) for k, v in res["coverage"].items()})
    inv = res["inventory"]
    print("\nЗапасы:\n", inv.round(3).to_string())
    base, best = inv.iloc[0], inv.loc["LightGBM"]
    print(f"\nИзлишек LightGBM против «{inv.index[0]}»: {best['avg_excess_rub'] / base['avg_excess_rub'] - 1:+.1%}")
    print(f"Средний запас: {best['avg_stock_rub'] / base['avg_stock_rub'] - 1:+.1%}")
    print(f"Сервис: {base['fill_rate']:.1%} -> {best['fill_rate']:.1%}")


def main(reuse: bool = False):
    if not SALES_FILE.exists():
        generate_data.main()
    df = load_sales()
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if reuse and FORECASTS_FILE.exists():
        fc, imp = load_backtest()
    else:
        fc, last_model = run_backtest(df)
        imp = plots.feature_importance(last_model)
        fc.to_csv(FORECASTS_FILE, index=False, compression="gzip")
        imp.to_csv(IMPORTANCE_FILE, header=["gain_share"])
    print_summary(build_reports(df, fc, imp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="не переобучать модели, взять сохранённые прогнозы")
    main(reuse=parser.parse_args().reuse)
