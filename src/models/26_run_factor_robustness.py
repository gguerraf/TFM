"""Run factor-model robustness checks"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from model_extension_utils import (
    EVENT_H,
    MODEL_TERMS,
    OUTPUT_DIR,
    batch_factor_car_for_dates,
    download_or_load_factors,
    fit_absorbed,
    format_result_block,
    load_panel,
    load_returns,
    recompute_interaction_terms,
    result_row,
)

FACTOR_PANEL_PATH = OUTPUT_DIR / "factor_robustness_panel.csv"
SUMMARY_PATH = OUTPUT_DIR / "factor_robustness_summary.csv"
DETAILS_PATH = OUTPUT_DIR / "factor_robustness_details.txt"

def compute_factor_cars(panel, returns, factors):
    needed = {"etf", "Shock_i_factor", "threshold_factor", "survives_factor", "AR_j_factor"}
    if FACTOR_PANEL_PATH.exists():
        cached_cols = set(pd.read_csv(FACTOR_PANEL_PATH, nrows=0).columns)
        if needed.issubset(cached_cols):
            print(f"Loading cached factor robustness panel: {FACTOR_PANEL_PATH}")
            extra = pd.read_csv(FACTOR_PANEL_PATH, parse_dates=["t0"])
            return panel.merge(extra, on=["etf", "event_id", "stock_i", "stock_j", "t0"], how="left")

    events = panel[["etf", "event_id", "stock_i", "t0"]].drop_duplicates().copy()
    receiver_dates = panel[["etf", "stock_j", "t0"]].drop_duplicates().copy()

    shock_parts = []
    for stock_i, stock_events in events.groupby("stock_i", sort=False):
        if stock_i not in returns.columns:
            continue
        cars = batch_factor_car_for_dates(returns[stock_i], factors, stock_events["t0"].unique(), EVENT_H)
        if cars.empty:
            continue
        cars = cars.rename(columns={
            "car": "Shock_i_factor",
            "sigma_eps": "sigma_eps_factor",
            "threshold": "threshold_factor",
        })
        shock_parts.append(stock_events.merge(cars, on="t0", how="left"))
    shocks = pd.concat(shock_parts, ignore_index=True)
    shocks["survives_factor"] = shocks["Shock_i_factor"].abs() > shocks["threshold_factor"]

    receiver_parts = []
    for (etf, stock_j), stock_dates in receiver_dates.groupby(["etf", "stock_j"], sort=False):
        if stock_j not in returns.columns:
            continue
        cars = batch_factor_car_for_dates(returns[stock_j], factors, stock_dates["t0"].unique(), EVENT_H)
        if cars.empty:
            continue
        cars = cars.rename(columns={"car": "AR_j_factor"})[["t0", "AR_j_factor"]]
        merged = stock_dates.merge(cars, on="t0", how="left")
        receiver_parts.append(merged)
    receivers = pd.concat(receiver_parts, ignore_index=True)

    extra = panel[["etf", "event_id", "stock_i", "stock_j", "t0"]].copy()
    extra = extra.merge(
        shocks[["etf", "event_id", "stock_i", "t0", "Shock_i_factor", "sigma_eps_factor", "threshold_factor", "survives_factor"]],
        on=["etf", "event_id", "stock_i", "t0"],
        how="left",
    )
    extra = extra.merge(receivers, on=["etf", "stock_j", "t0"], how="left")
    extra.to_csv(FACTOR_PANEL_PATH, index=False)
    return panel.merge(extra, on=["etf", "event_id", "stock_i", "stock_j", "t0"], how="left")

def run_variant(panel, label, survive_only):
    df = panel.copy()
    if survive_only:
        df = df[df["survives_factor"] == True].copy()
    df = recompute_interaction_terms(df, shock_col="Shock_i_factor")
    df["AR_j"] = df["AR_j_factor"]
    res, used = fit_absorbed(df, terms=MODEL_TERMS)
    row = result_row(label, res)
    row["rows_used"] = len(used)
    return row, format_result_block(label, res)

def main():
    print("Loading panel, returns and factors...")
    panel = load_panel().reset_index(drop=True)
    returns = load_returns()
    factors = download_or_load_factors()
    print(f"  Factor rows: {len(factors):,}")
    robust = compute_factor_cars(panel, returns, factors)

    variants = [
        ("FF5_Carhart_same_external_events", False),
        ("FF5_Carhart_surviving_factor_events", True),
    ]
    rows = []
    details = ["FAMA-FRENCH / CARHART ROBUSTNESS", "=" * 80]
    details.append(f"Base panel rows: {len(panel):,}")
    details.append(f"Rows with factor receiver AR: {robust['AR_j_factor'].notna().sum():,}")
    details.append(f"Rows with surviving factor shocks: {robust['survives_factor'].sum():,}")
    details.append("")

    for label, survive_only in variants:
        print(f"Estimating {label}...")
        try:
            row, block = run_variant(robust, label, survive_only)
            rows.append(row)
            details.append(block)
            details.append("")
        except Exception as exc:
            details.append(f"{label} failed: {exc}")
            print(f"  [WARN] {label} failed: {exc}")

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    DETAILS_PATH.write_text("\n".join(details), encoding="utf-8")
    if not summary.empty:
        print(summary.to_string(index=False, float_format="%.6f"))

if __name__ == "__main__":
    main()
