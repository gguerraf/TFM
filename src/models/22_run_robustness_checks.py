"""Run robustness checks"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv.absorbing import AbsorbingLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from model_extension_utils import first_day_of_episode_events, recompute_interaction_terms

BASE_DIR   = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR   = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR    = BASE_DIR / "figures"

MAIN_TERMS = ["b1_term", "b2_term", "b4_term", "b5_term",
              "receiver_weight_term", "corr_term", "hhi_term", "asym_term"]

BASE_FORMULA = ("AR_j ~ b1_term + b2_term + b4_term + b5_term"
                " + receiver_weight_term + corr_term + hhi_term + asym_term"
                " + Illiq_j + Mispricing_k + Similarity_ij"
                " + w_j + Corr_ij_60d + HHI_etf_t + Neg_e"
                " + C(stock_j) + C(year_quarter)")

def run_spec(df: pd.DataFrame, label: str, formula: str = None,
             log_lines: list = None,
             cluster_cols: tuple = ("event_id",)) -> dict:
    """Runs one specification using absorbed fixed effects"""
    if formula is None:
        formula = BASE_FORMULA

    terms = ["b1_term", "b2_term", "b4_term", "b5_term",
             "receiver_weight_term", "corr_term", "hhi_term"]
    if "asym_term" in formula:
        terms.append("asym_term")
    terms.extend(["Illiq_j", "Mispricing_k", "Similarity_ij",
                  "w_j", "Corr_ij_60d", "HHI_etf_t"])
    if "Neg_e" in formula:
        terms.append("Neg_e")

    absorb_cols = ["stock_j"]
    if "C(year_quarter)" in formula:
        absorb_cols.append("year_quarter")

    cluster_cols = list(cluster_cols)
    required = ["AR_j", *terms, *absorb_cols, *cluster_cols]
    sub = df.dropna(subset=[c for c in required if c in df.columns]).copy()

    if len(sub) < 100:
        msg = f"  [{label}] Too few obs ({len(sub)}). Skipping."
        print(msg)
        if log_lines is not None:
            log_lines.append(msg)
        return None

    try:
        y = sub["AR_j"]
        x = sub[terms]
        absorb = sub[absorb_cols].astype("category")
        clusters = pd.DataFrame(
            {col: pd.Categorical(sub[col]).codes for col in cluster_cols},
            index=sub.index,
        )
        res = AbsorbingLS(y, x, absorb=absorb).fit(
            cov_type="clustered",
            clusters=clusters,
        )
    except Exception as e:
        msg = f"  [{label}] Regression failed: {e}"
        print(msg)
        if log_lines is not None:
            log_lines.append(msg)
        return None

    row = {"check": label, "N": res.nobs, "adj_R2": res.rsquared_adj,
           "clusters": "+".join(cluster_cols)}
    for term in MAIN_TERMS:
        if term in res.params.index:
            row[f"{term}_coef"] = res.params[term]
            row[f"{term}_pval"] = res.pvalues[term]
        else:
            row[f"{term}_coef"] = np.nan
            row[f"{term}_pval"] = np.nan

    if log_lines is not None:
        log_lines.append(f"\n{'=' * 60}")
        log_lines.append(f"CHECK: {label}  (N = {res.nobs:.0f}, "
                         f"clustered by {' and '.join(cluster_cols)})")
        log_lines.append(f"{'=' * 60}")
        for term in MAIN_TERMS:
            if term in res.params.index:
                stars = ("***" if res.pvalues[term] < 0.01 else
                         "**"  if res.pvalues[term] < 0.05 else
                         "*"   if res.pvalues[term] < 0.10 else "  ")
                log_lines.append(
                    f"  {term:<15s}: {res.params[term]:+.6f}  "
                    f"p={res.pvalues[term]:.4f} {stars}")
        log_lines.append(f"  Adj R2: {res.rsquared_adj:.4f}")

    return row

def outside_etf_placebo(panel: pd.DataFrame, seed: int = 123):
    """Replaces each event's shock with a same-date shock from outside the ETF"""
    rng = np.random.default_rng(seed)
    events = (panel[["etf", "event_id", "t0", "stock_i", "Shock_i", "w_i"]]
              .drop_duplicates(["etf", "event_id"]))
    members = pd.concat([
        panel[["etf", "t0", "stock_j"]].rename(columns={"stock_j": "stock_i_donor"}),
        panel[["etf", "t0", "stock_i"]].rename(columns={"stock_i": "stock_i_donor"}),
    ]).drop_duplicates()
    members["_held"] = True

    pairs = events.merge(events, on="t0", suffixes=("", "_donor"))
    pairs = pairs[pairs["etf"] != pairs["etf_donor"]]
    pairs = pairs.merge(members, on=["etf", "t0", "stock_i_donor"], how="left")
    pairs = pairs[pairs["_held"].isna()].copy()
    pairs["_draw"] = rng.random(len(pairs))
    donors = (pairs.sort_values("_draw")
              .drop_duplicates(["etf", "event_id"])
              [["etf", "event_id", "Shock_i_donor", "w_i_donor"]])

    out = panel.merge(donors, on=["etf", "event_id"], how="inner")
    out["Shock_i"] = out["Shock_i_donor"]
    out["w_i"] = out["w_i_donor"]
    out = recompute_interaction_terms(out.drop(columns=["Shock_i_donor", "w_i_donor"]))
    return out, len(events), len(donors)

def run_all_checks(panel: pd.DataFrame) -> pd.DataFrame:
    """Runs the full robustness battery"""

    results = []
    log_lines = []
    log_lines.append("ROBUSTNESS CHECK BATTERY")
    log_lines.append("=" * 70)
    log_lines.append(f"Baseline panel: {len(panel):,} observations")
    log_lines.append(f"Date range: {panel['t0'].min()} to {panel['t0'].max()}")

    print("\nR0: Baseline specification")
    r = run_spec(panel, "R0_Baseline", log_lines=log_lines)
    if r: results.append(r)

    print("R5a: Positive shocks only")
    pos = panel[panel["Neg_e"] == 0].copy()

    formula_nosign = ("AR_j ~ b1_term + b2_term + b4_term + b5_term"
                      " + receiver_weight_term + corr_term + hhi_term"
                      " + Illiq_j + Mispricing_k + Similarity_ij"
                      " + w_j + Corr_ij_60d + HHI_etf_t"
                      " + C(stock_j) + C(year_quarter)")
    r = run_spec(pos, "R5a_Positive_shocks", formula=formula_nosign,
                 log_lines=log_lines)
    if r: results.append(r)

    print("R5b: Negative shocks only")
    neg = panel[panel["Neg_e"] == 1].copy()
    r = run_spec(neg, "R5b_Negative_shocks", formula=formula_nosign,
                 log_lines=log_lines)
    if r: results.append(r)

    print("R6a: Pre-COVID (before 2020-03-01)")
    pre_covid = panel[panel["t0"] < "2020-03-01"].copy()
    r = run_spec(pre_covid, "R6a_Pre_COVID", log_lines=log_lines)
    if r: results.append(r)

    print("R6b: Post-COVID (2020-03-01 onward)")
    post_covid = panel[panel["t0"] >= "2020-03-01"].copy()
    r = run_spec(post_covid, "R6b_Post_COVID", log_lines=log_lines)
    if r: results.append(r)

    for etf in panel["etf"].unique():
        print(f"R9: ETF = {etf}")
        sub = panel[panel["etf"] == etf].copy()
        r = run_spec(sub, f"R9_{etf}", log_lines=log_lines)
        if r: results.append(r)

    print("R10: Excluding the top-3 weight origin stocks of each ETF")
    mean_w = panel.groupby(["etf", "stock_i"], as_index=False)["w_i"].mean()
    top3 = (mean_w.sort_values("w_i", ascending=False)
            .groupby("etf").head(3)[["etf", "stock_i"]])
    for etf, group in top3.groupby("etf"):
        log_lines.append(f"  R10 excluded origin stocks in {etf}: "
                         f"{', '.join(group['stock_i'])}")
    excl = panel.merge(top3.assign(_top3=True), on=["etf", "stock_i"], how="left")
    excl = excl[excl["_top3"].isna()].drop(columns="_top3")
    r = run_spec(excl, "R10_Excl_top3_weight_origins", log_lines=log_lines)
    if r: results.append(r)

    print("R11: Trimmed AR_j and Shock_i (1%-99%)")
    trim = panel.copy()
    keep = pd.Series(True, index=trim.index)
    for col in ["AR_j", "Shock_i"]:
        q01, q99 = trim[col].quantile([0.01, 0.99])
        keep &= trim[col].between(q01, q99)
    trim = trim.loc[keep].copy()
    r = run_spec(trim, "R11_Trimmed_1_99", log_lines=log_lines)
    if r: results.append(r)

    print("R16: Placebo event assignment within ETF")
    rng = np.random.default_rng(42)
    placebo = panel.copy()
    event_cols = ["event_id", "etf", "Shock_i", "w_i", "Neg_e"]
    events = placebo[event_cols].drop_duplicates("event_id").copy()
    shuffled_parts = []
    for _, group in events.groupby("etf", sort=False):
        shuffled = group.copy()
        source = group[["Shock_i", "w_i", "Neg_e"]].to_numpy()
        shuffled[["Shock_i_p", "w_i_p", "Neg_e_p"]] = source[
            rng.permutation(len(group))
        ]
        shuffled_parts.append(shuffled[["event_id", "Shock_i_p", "w_i_p", "Neg_e_p"]])
    shuffled_events = pd.concat(shuffled_parts, ignore_index=True)
    placebo = placebo.merge(shuffled_events, on="event_id", how="left")
    placebo["Shock_i"] = placebo["Shock_i_p"]
    placebo["w_i"] = placebo["w_i_p"]
    placebo["Neg_e"] = placebo["Neg_e_p"].astype(int)
    placebo["b1_term"] = placebo["Shock_i"] * placebo["w_i"]
    placebo["b2_term"] = placebo["Shock_i"] * placebo["w_i"] * placebo["Illiq_j"]
    placebo["b4_term"] = placebo["Shock_i"] * placebo["w_i"] * placebo["Mispricing_k"]
    placebo["b5_term"] = placebo["Shock_i"] * placebo["Similarity_ij"]
    placebo["receiver_weight_term"] = placebo["Shock_i"] * placebo["w_i"] * placebo["w_j"]
    placebo["corr_term"] = placebo["Shock_i"] * placebo["w_i"] * placebo["Corr_ij_60d"]
    placebo["hhi_term"] = placebo["Shock_i"] * placebo["w_i"] * placebo["HHI_etf_t"]
    placebo["asym_term"] = placebo["Neg_e"] * placebo["Shock_i"] * placebo["w_i"]
    placebo = placebo.drop(columns=["Shock_i_p", "w_i_p", "Neg_e_p"])
    r = run_spec(placebo, "R16_Placebo_event_assignment", log_lines=log_lines)
    if r: results.append(r)

    print("R17: Placebo with shocks from outside the ETF")
    outside, n_events, n_matched = outside_etf_placebo(panel, seed=123)
    log_lines.append(f"\nR17: {n_matched:,} of {n_events:,} events matched "
                     f"to a same-date shock from outside the ETF")
    r = run_spec(outside, "R17_Placebo_outside_ETF_shocks", log_lines=log_lines)
    if r: results.append(r)

    print("R18: Pre-event placebo (note: would need recomputed AR_j)")

    np.random.seed(456)
    pre_event = panel.copy()
    pre_event["AR_j"] = np.random.permutation(pre_event["AR_j"].values)
    r = run_spec(pre_event, "R18_Shuffled_AR_j", log_lines=log_lines)
    if r: results.append(r)

    print("R21: No year-quarter fixed effects")
    formula_no_yq = ("AR_j ~ b1_term + b2_term + b4_term + b5_term"
                     " + receiver_weight_term + corr_term + hhi_term"
                     " + asym_term + Illiq_j + Mispricing_k + Similarity_ij"
                     " + w_j + Corr_ij_60d + HHI_etf_t + Neg_e + C(stock_j)")
    r = run_spec(panel, "R21_No_YQ_FE", formula=formula_no_yq,
                 log_lines=log_lines)
    if r: results.append(r)

    print("R22: Non-overlapping events")
    first_days = first_day_of_episode_events(panel)
    non_overlap = panel.merge(first_days, on=["etf", "event_id"], how="inner")
    log_lines.append(f"\nR22: {len(first_days):,} first-day events, "
                     f"{len(non_overlap):,} receiver observations")
    r = run_spec(non_overlap, "R22_Non_overlapping_events", log_lines=log_lines,
                 cluster_cols=("event_id", "stock_j"))
    if r: results.append(r)

    results_df = pd.DataFrame(results)

    summary_path = OUTPUT_DIR / "robustness_summary.csv"
    results_df.to_csv(summary_path, index=False)

    details_path = OUTPUT_DIR / "robustness_details.txt"
    with open(details_path, "w") as f:
        f.write("\n".join(log_lines))

    print("\n" + "=" * 90)
    print("ROBUSTNESS SUMMARY")
    print("=" * 90)
    display_cols = ["check", "N", "adj_R2", "clusters"]
    for t in MAIN_TERMS:
        display_cols.extend([f"{t}_coef", f"{t}_pval"])
    print(results_df[display_cols].to_string(index=False, float_format="%.4f"))

    _plot_robustness_heatmap(results_df)

    return results_df

def _plot_robustness_heatmap(df: pd.DataFrame):
    """Creates a heatmap showing coefficient signs and significance"""
    coef_cols = [f"{t}_coef" for t in MAIN_TERMS]
    pval_cols = [f"{t}_pval" for t in MAIN_TERMS]

    coefs = df[coef_cols].copy()
    coefs.columns = [c.replace("_coef", "") for c in coef_cols]
    coefs.index = df["check"]

    pvals = df[pval_cols].copy()
    pvals.columns = [c.replace("_pval", "") for c in pval_cols]
    pvals.index = df["check"]

    annot = coefs.copy().astype(str)
    for col in annot.columns:
        for idx in annot.index:
            c = coefs.loc[idx, col]
            p = pvals.loc[idx, col]
            if pd.isna(c):
                annot.loc[idx, col] = ""
            else:
                stars = ("***" if p < 0.01 else
                         "**" if p < 0.05 else
                         "*" if p < 0.1 else "")
                annot.loc[idx, col] = f"{c:.3f}{stars}"

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.5)))
    sns.heatmap(coefs.astype(float), annot=annot, fmt="",
                center=0, cmap="RdBu_r",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Coefficient"})
    ax.set_title("Robustness Checks: Coefficient Stability\n"
                 "(*** p<0.01, ** p<0.05, * p<0.10)", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    out = FIG_DIR / "robustness_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()

if __name__ == "__main__":
    panel_path = OUTPUT_DIR / "panel_improved.csv"

    if not panel_path.exists():
        print(f"[ERROR] {panel_path} not found.")
        print("Run 21_estimate_improved_model.py first to generate the panel.")
        exit(1)

    print(f"Loading panel from {panel_path}")
    panel = pd.read_csv(panel_path, parse_dates=["t0"])
    print(f"  Panel shape: {panel.shape}")

    results = run_all_checks(panel)

