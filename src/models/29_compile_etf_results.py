"""Compile ETF-level result summaries"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from model_extension_utils import (
    MAIN_TERMS,
    MODEL_TERMS,
    OUTPUT_DIR,
    fit_absorbed,
    load_panel,
    recompute_interaction_terms,
    result_row,
)

ETF_LIST = ["XME", "XLE", "IHE", "XLV"]
LP_HORIZONS = list(range(11))
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
QUANTILE_MAX_ROWS = 50_000
RANDOM_STATE = 42

BASE_TERMS = [
    "b1_term", "b2_term", "b4_term", "b5_term",
    "receiver_weight_term", "corr_term", "hhi_term", "asym_term",
    "Illiq_j", "Mispricing_k", "Similarity_ij",
    "w_j", "Corr_ij_60d", "HHI_etf_t", "Neg_e",
]

def safe_fit(df, label, etf, y_col="AR_j", terms=None):
    try:
        res, used = fit_absorbed(df, terms=terms or MODEL_TERMS, y_col=y_col)
        row = result_row(label, res)
        row["etf"] = etf
        row["rows_used"] = len(used)
        return row
    except Exception as exc:
        row = {"specification": label, "etf": etf, "error": str(exc)}
        return row

def baseline_by_etf(panel):
    rows = []
    for etf in ETF_LIST:
        sub = panel[panel["etf"] == etf].copy()
        rows.append(safe_fit(sub, "baseline_expanded", etf))
    return pd.DataFrame(rows)

def benchmark_by_etf(panel):
    path = OUTPUT_DIR / "benchmark_robustness_panel.csv"
    extra = pd.read_csv(path, parse_dates=["t0"])
    robust = panel.merge(extra, on=["etf", "event_id", "stock_i", "stock_j", "t0"], how="left")
    variants = [
        ("LOO_LTO_all_events", "AR_j_lto", False),
        ("LOO_LTO_surviving_events", "AR_j_lto", True),
        ("LOO_receiver_surviving_events", "AR_j_loo_receiver", True),
    ]
    rows = []
    for etf in ETF_LIST:
        etf_panel = robust[robust["etf"] == etf].copy()
        for label, y_col, survive_only in variants:
            df = etf_panel.copy()
            if survive_only:
                df = df[df["survives_loo"] == True].copy()
            df = recompute_interaction_terms(df, shock_col="Shock_i_loo")
            df["AR_j"] = df[y_col]
            rows.append(safe_fit(df, label, etf))
    return pd.DataFrame(rows)

def factor_by_etf(panel):
    path = OUTPUT_DIR / "factor_robustness_panel.csv"
    extra = pd.read_csv(path, parse_dates=["t0"])
    robust = panel.merge(extra, on=["etf", "event_id", "stock_i", "stock_j", "t0"], how="left")
    variants = [
        ("FF5_Carhart_same_events", False),
        ("FF5_Carhart_surviving_events", True),
    ]
    rows = []
    for etf in ETF_LIST:
        etf_panel = robust[robust["etf"] == etf].copy()
        for label, survive_only in variants:
            df = etf_panel.copy()
            if survive_only:
                df = df[df["survives_factor"] == True].copy()
            df = recompute_interaction_terms(df, shock_col="Shock_i_factor")
            df["AR_j"] = df["AR_j_factor"]
            rows.append(safe_fit(df, label, etf))
    return pd.DataFrame(rows)

def local_projections_by_etf(panel):
    outcomes = pd.read_csv(OUTPUT_DIR / "local_projection_outcomes.csv")
    df = panel.reset_index(drop=True).copy()
    df["row_id"] = np.arange(len(df))
    df = df.merge(outcomes, on="row_id", how="left")
    rows = []
    for etf in ETF_LIST:
        etf_panel = df[df["etf"] == etf].copy()
        for h in LP_HORIZONS:
            label = f"LP_h{h}"
            row = safe_fit(etf_panel, label, etf, y_col=f"AR_j_h{h}")
            row["horizon"] = h
            rows.append(row)
    return pd.DataFrame(rows)

def quantile_by_etf(panel):
    rows = []
    required = ["AR_j", *BASE_TERMS, "etf", "year_quarter"]
    df = panel.dropna(subset=required).copy()
    for etf in ETF_LIST:
        sub = df[df["etf"] == etf].copy()
        if len(sub) > QUANTILE_MAX_ROWS:
            sub = sub.sample(QUANTILE_MAX_ROWS, random_state=RANDOM_STATE)
        x = sub[BASE_TERMS].astype(float).copy()
        dummies = pd.get_dummies(sub[["year_quarter"]].astype(str), drop_first=True, dtype=float)
        x = pd.concat([x, dummies], axis=1)
        x = sm.add_constant(x, has_constant="add")
        y = sub["AR_j"].astype(float)
        model = QuantReg(y, x)
        for q in QUANTILES:
            try:
                res = model.fit(q=q, max_iter=1000, p_tol=1e-5)
                row = {"etf": etf, "quantile": q, "N": res.nobs}
                for term in MAIN_TERMS:
                    row[f"{term}_coef"] = res.params.get(term, np.nan)
                    row[f"{term}_pval"] = res.pvalues.get(term, np.nan)
                rows.append(row)
            except Exception as exc:
                rows.append({"etf": etf, "quantile": q, "error": str(exc)})
    return pd.DataFrame(rows)

def save(df, name):
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)

def main():
    panel = load_panel().reset_index(drop=True)
    save(baseline_by_etf(panel), "etf_baseline_summary.csv")
    save(benchmark_by_etf(panel), "etf_benchmark_robustness_summary.csv")
    save(factor_by_etf(panel), "etf_factor_robustness_summary.csv")
    save(local_projections_by_etf(panel), "etf_local_projections_summary.csv")
    save(quantile_by_etf(panel), "etf_quantile_regression_summary.csv")

if __name__ == "__main__":
    main()
