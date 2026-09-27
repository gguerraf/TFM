"""Compute shock effects at the means and average marginal effects"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from model_extension_utils import (
    ETF_LIST,
    MAIN_TERMS,
    MODEL_TERMS,
    OUTPUT_DIR,
    centre_interaction_terms,
    fit_absorbed,
    load_panel,
    recompute_interaction_terms,
    shock_effects,
)

SUMMARY_PATH = OUTPUT_DIR / "marginal_effects_summary.csv"
CENTRED_PATH = OUTPUT_DIR / "centred_specifications.csv"
DETAILS_PATH = OUTPUT_DIR / "marginal_effects_details.txt"
MERGE_KEYS = ["etf", "event_id", "stock_i", "stock_j", "t0"]
LP_HORIZONS = range(11)


def summarise(label, group, df, y_col="AR_j", terms=None):
    res, used = fit_absorbed(df, terms=terms or MODEL_TERMS, y_col=y_col)
    row = {
        "specification": label,
        "group": group,
        "N": int(res.nobs),
        "adj_R2": res.rsquared_adj,
        "b1_raw": res.params.get("b1_term", np.nan),
        "b1_raw_se": res.std_errors.get("b1_term", np.nan),
        "b1_raw_pval": res.pvalues.get("b1_term", np.nan),
        "corr_term": res.params.get("corr_term", np.nan),
        "corr_term_se": res.std_errors.get("corr_term", np.nan),
        "corr_term_pval": res.pvalues.get("corr_term", np.nan),
    }
    row.update(shock_effects(res, used))
    return row


def main_and_etf_rows(panel):
    print("Estimating the main model and the ETF models", flush=True)
    rows = [summarise("main_expanded", "main", panel)]
    for etf in ETF_LIST:
        rows.append(summarise(f"etf_{etf}", "etf", panel[panel["etf"] == etf]))
    return rows


def benchmark_rows(panel):
    path = OUTPUT_DIR / "benchmark_robustness_panel.csv"
    if not path.exists():
        print(f"[WARN] {path.name} not found, run 24_run_benchmark_robustness.py first")
        return []
    print("Estimating the synthetic benchmark models", flush=True)
    extra = pd.read_csv(path, parse_dates=["t0"])
    robust = panel.merge(extra, on=MERGE_KEYS, how="left")
    variants = [
        ("LOO_origin_LTO_receiver_all_external_events", "AR_j_lto", False),
        ("LOO_origin_LTO_receiver_surviving_events", "AR_j_lto", True),
        ("LOO_origin_LOO_receiver_surviving_events", "AR_j_loo_receiver", True),
    ]
    rows = []
    for label, y_col, survive_only in variants:
        df = robust[robust["survives_loo"] == True] if survive_only else robust
        df = recompute_interaction_terms(df, shock_col="Shock_i_loo")
        df["AR_j"] = df[y_col]
        rows.append(summarise(label, "benchmark", df))
        if label == "LOO_origin_LTO_receiver_surviving_events":
            for etf in ETF_LIST:
                rows.append(summarise(f"{label}_{etf}", "benchmark_etf", df[df["etf"] == etf]))
    return rows


def factor_rows(panel):
    path = OUTPUT_DIR / "factor_robustness_panel.csv"
    if not path.exists():
        print(f"[WARN] {path.name} not found, run 26_run_factor_robustness.py first")
        return []
    print("Estimating the factor models", flush=True)
    extra = pd.read_csv(path, parse_dates=["t0"])
    robust = panel.merge(extra, on=MERGE_KEYS, how="left")
    rows = []
    for label, survive_only in [
        ("FF5_Carhart_same_external_events", False),
        ("FF5_Carhart_surviving_factor_events", True),
    ]:
        df = robust[robust["survives_factor"] == True] if survive_only else robust
        df = recompute_interaction_terms(df, shock_col="Shock_i_factor")
        df["AR_j"] = df["AR_j_factor"]
        rows.append(summarise(label, "factor", df))
    return rows


def local_projection_rows(panel):
    path = OUTPUT_DIR / "local_projection_outcomes.csv"
    if not path.exists():
        print(f"[WARN] {path.name} not found, run 25_run_local_projections.py first")
        return []
    print("Estimating the local projections", flush=True)
    outcomes = pd.read_csv(path)
    df = panel.reset_index(drop=True).copy()
    df["row_id"] = np.arange(len(df))
    df = df.merge(outcomes, on="row_id", how="left")
    rows = []
    for h in LP_HORIZONS:
        row = summarise(f"LP_h{h}", "local_projection", df, y_col=f"AR_j_h{h}")
        row["horizon"] = h
        rows.append(row)
    return rows


def centred_rows(panel):
    print("Estimating the centred specifications", flush=True)
    required = ["AR_j", *MODEL_TERMS, "stock_j", "year_quarter", "event_id"]
    sample = centre_interaction_terms(panel.dropna(subset=required).copy())
    rows = []
    specs = [
        ("full_expanded_centred", MODEL_TERMS),
        ("full_without_correlation_centred",
         [t for t in MODEL_TERMS if t not in {"corr_term", "Corr_ij_60d"}]),
    ]
    for label, terms in specs:
        res, used = fit_absorbed(sample, terms=terms)
        row = {"specification": label, "N": int(res.nobs), "adj_R2": res.rsquared_adj}
        for term in MAIN_TERMS:
            row[f"{term}_coef"] = res.params.get(term, np.nan)
            row[f"{term}_se"] = res.std_errors.get(term, np.nan)
            row[f"{term}_pval"] = res.pvalues.get(term, np.nan)
        rows.append(row)
    return rows


def write_details(summary, centred):
    lines = [
        "SHOCK EFFECTS AT THE MEANS AND AVERAGE MARGINAL EFFECTS",
        "=" * 80,
        "Standard errors are two-way clustered by event_id and stock_j",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['specification']:<52s} N={row['N']:>8,d}  "
            f"b1_raw={row['b1_raw']:+.4f} (p={row['b1_raw_pval']:.4f})  "
            f"at_means={row['direct_at_means']:+.4f} (t={row['direct_at_means_t']:.2f})  "
            f"AME={row['ame_shock']:+.5f} (t={row['ame_shock_t']:.2f})  "
            f"corr={row['corr_term']:+.4f}"
        )
    lines += ["", "CENTRED SPECIFICATIONS", "=" * 80]
    for _, row in centred.iterrows():
        lines.append(
            f"{row['specification']:<40s} N={row['N']:>8,d}  "
            f"b1={row['b1_term_coef']:+.4f} (p={row['b1_term_pval']:.4f})"
        )
    DETAILS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    panel = load_panel()
    rows = main_and_etf_rows(panel)
    rows += benchmark_rows(panel)
    rows += factor_rows(panel)
    rows += local_projection_rows(panel)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    centred = pd.DataFrame(centred_rows(panel))
    centred.to_csv(CENTRED_PATH, index=False)

    write_details(summary, centred)


if __name__ == "__main__":
    main()
