"""Run LOO and LTO benchmark checks"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from model_extension_utils import (
    EVENT_H,
    MAIN_TERMS,
    MODEL_TERMS,
    OUTPUT_DIR,
    batch_market_car_for_dates,
    fit_absorbed,
    format_result_block,
    load_holdings,
    load_panel,
    load_returns,
    make_weight_panel,
    recompute_interaction_terms,
    result_row,
    synthetic_basket_return,
)

ROBUST_PANEL_PATH = OUTPUT_DIR / "benchmark_robustness_panel.csv"
SUMMARY_PATH = OUTPUT_DIR / "benchmark_robustness_summary.csv"
DETAILS_PATH = OUTPUT_DIR / "benchmark_robustness_details.txt"

def compute_origin_shocks(panel, returns):
    events = panel[["etf", "event_id", "stock_i", "t0"]].drop_duplicates().copy()
    out_parts = []
    for etf, ev in events.groupby("etf", sort=False):
        print(f"Computing ETF(-i) origin shocks for {etf}...")
        holdings = load_holdings(etf)
        etf_symbols = holdings["symbol"].dropna().unique().tolist()
        cols = [c for c in etf_symbols if c in returns.columns]
        ret_etf = returns[cols]
        weights = make_weight_panel(holdings, returns.index, cols)
        etf_parts = []
        for stock_i, stock_events in ev.groupby("stock_i", sort=False):
            if stock_i not in ret_etf.columns:
                continue
            bench_i = synthetic_basket_return(weights, ret_etf, [stock_i])
            cars = batch_market_car_for_dates(ret_etf[stock_i], bench_i, stock_events["t0"], EVENT_H)
            if cars.empty:
                continue
            cars = cars.rename(columns={
                "car": "Shock_i_loo",
                "sigma_eps": "sigma_eps_loo",
                "threshold": "threshold_loo",
            })
            merged = stock_events.merge(cars, on="t0", how="left")
            etf_parts.append(merged)
        if etf_parts:
            out_parts.append(pd.concat(etf_parts, ignore_index=True))
    shocks = pd.concat(out_parts, ignore_index=True)
    shocks["survives_loo"] = shocks["Shock_i_loo"].abs() > shocks["threshold_loo"]
    return shocks

def compute_receiver_outcomes(panel, returns):
    result = panel[["etf", "event_id", "stock_i", "stock_j", "t0"]].copy()
    result["AR_j_lto"] = np.nan
    result["AR_j_loo_receiver"] = np.nan

    for etf, etf_panel in panel.groupby("etf", sort=False):
        print(f"Computing receiver outcomes for {etf}...")
        holdings = load_holdings(etf)
        etf_symbols = holdings["symbol"].dropna().unique().tolist()
        cols = [c for c in etf_symbols if c in returns.columns]
        ret_etf = returns[cols]
        weights = make_weight_panel(holdings, returns.index, cols)
        etf_idx = etf_panel.index

        for stock_j, stock_panel in etf_panel.groupby("stock_j", sort=False):
            if stock_j not in ret_etf.columns:
                continue
            bench_j = synthetic_basket_return(weights, ret_etf, [stock_j])
            cars = batch_market_car_for_dates(ret_etf[stock_j], bench_j, stock_panel["t0"], EVENT_H)
            if not cars.empty:
                mapping = cars.set_index("t0")["car"]
                result.loc[stock_panel.index, "AR_j_loo_receiver"] = stock_panel["t0"].map(mapping).to_numpy()

        pair_groups = etf_panel.groupby(["stock_i", "stock_j"], sort=False)
        total_pairs = pair_groups.ngroups
        for n, ((stock_i, stock_j), pair_panel) in enumerate(pair_groups, start=1):
            if n % 500 == 0:
                print(f"  {etf}: pair {n:,}/{total_pairs:,}")
            if stock_j not in ret_etf.columns:
                continue
            exclude = [stock_i, stock_j]
            bench_ij = synthetic_basket_return(weights, ret_etf, exclude)
            cars = batch_market_car_for_dates(ret_etf[stock_j], bench_ij, pair_panel["t0"], EVENT_H)
            if cars.empty:
                continue
            mapping = cars.set_index("t0")["car"]
            result.loc[pair_panel.index, "AR_j_lto"] = pair_panel["t0"].map(mapping).to_numpy()

    return result

def build_or_load_robust_panel(panel, returns):
    needed = {"Shock_i_loo", "threshold_loo", "survives_loo", "AR_j_lto", "AR_j_loo_receiver"}
    if ROBUST_PANEL_PATH.exists():
        cached_cols = set(pd.read_csv(ROBUST_PANEL_PATH, nrows=0).columns)
        if needed.issubset(cached_cols):
            print(f"Loading cached benchmark robustness panel: {ROBUST_PANEL_PATH}")
            extra = pd.read_csv(ROBUST_PANEL_PATH, parse_dates=["t0"])
            return panel.merge(extra, on=["etf", "event_id", "stock_i", "stock_j", "t0"], how="left")

    shocks = compute_origin_shocks(panel, returns)
    receivers = compute_receiver_outcomes(panel, returns)
    extra = receivers.merge(
        shocks[["etf", "event_id", "Shock_i_loo", "sigma_eps_loo", "threshold_loo", "survives_loo"]],
        on=["etf", "event_id"],
        how="left",
    )
    extra.to_csv(ROBUST_PANEL_PATH, index=False)
    return panel.merge(extra, on=["etf", "event_id", "stock_i", "stock_j", "t0"], how="left")

def run_variant(panel, label, y_col, survive_only):
    df = panel.copy()
    if survive_only:
        df = df[df["survives_loo"] == True].copy()
    df = df.rename(columns={y_col: "AR_j_alt"})
    df = recompute_interaction_terms(df, shock_col="Shock_i_loo")
    df["AR_j"] = df["AR_j_alt"]
    res, used = fit_absorbed(df, terms=MODEL_TERMS)
    return result_row(label, res), format_result_block(label, res), len(used)

def main():
    print("Loading panel and returns...")
    panel = load_panel().reset_index(drop=True)
    returns = load_returns()
    robust = build_or_load_robust_panel(panel, returns)

    variants = [
        ("LOO_origin_LTO_receiver_all_external_events", "AR_j_lto", False),
        ("LOO_origin_LTO_receiver_surviving_events", "AR_j_lto", True),
        ("LOO_origin_LOO_receiver_surviving_events", "AR_j_loo_receiver", True),
    ]
    rows = []
    details = ["BENCHMARK ROBUSTNESS: LOO/LTO", "=" * 80]
    details.append(f"Base panel rows: {len(panel):,}")
    details.append(f"Rows with LTO receiver AR: {robust['AR_j_lto'].notna().sum():,}")
    details.append(f"Rows with surviving LOO origin shocks: {robust['survives_loo'].sum():,}")
    details.append("")

    for label, y_col, survive_only in variants:
        print(f"Estimating {label}...")
        try:
            row, block, n_used = run_variant(robust, label, y_col, survive_only)
            row["rows_used"] = n_used
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
