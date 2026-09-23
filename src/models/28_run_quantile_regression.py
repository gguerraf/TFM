"""Run quantile regression diagnostics"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from model_extension_utils import OUTPUT_DIR, load_panel

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
MAX_ROWS = 200_000
RANDOM_STATE = 42
SUMMARY_PATH = OUTPUT_DIR / "quantile_regression_summary.csv"
DETAILS_PATH = OUTPUT_DIR / "quantile_regression_details.txt"

BASE_TERMS = [
    "b1_term", "b2_term", "b4_term", "b5_term",
    "receiver_weight_term", "corr_term", "hhi_term", "asym_term",
    "Illiq_j", "Mispricing_k", "Similarity_ij",
    "w_j", "Corr_ij_60d", "HHI_etf_t", "Neg_e",
]
MAIN_TERMS = [
    "b1_term", "b2_term", "b4_term", "b5_term",
    "receiver_weight_term", "corr_term", "hhi_term", "asym_term",
]

def prepare_sample(panel):
    required = ["AR_j", *BASE_TERMS, "etf", "year_quarter"]
    df = panel.dropna(subset=required).copy()
    if len(df) > MAX_ROWS:
        target = max(1, int(MAX_ROWS))
        df = df.sample(target, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)

def build_design(df):
    x = df[BASE_TERMS].astype(float).copy()
    dummies = pd.get_dummies(df[["etf", "year_quarter"]].astype(str), drop_first=True, dtype=float)
    x = pd.concat([x, dummies], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = df["AR_j"].astype(float)
    return y, x

def main():
    print("Loading improved panel...")
    panel = load_panel()
    df = prepare_sample(panel)
    print(f"Quantile regression sample: {len(df):,} rows")

    y, x = build_design(df)
    model = QuantReg(y, x)
    rows = []
    details = [
        "QUANTILE REGRESSION DIAGNOSTIC",
        "This is a secondary tail analysis, not the main fixed-effects model.",
        f"Sample rows: {len(df):,}",
        "",
    ]

    for q in QUANTILES:
        print(f"Estimating quantile q={q:.2f}...")
        res = model.fit(q=q, max_iter=1000, p_tol=1e-5)
        row = {"quantile": q, "N": res.nobs}
        details.append("=" * 80)
        details.append(f"Quantile {q:.2f}")
        details.append("=" * 80)
        for term in MAIN_TERMS:
            row[f"{term}_coef"] = res.params.get(term, np.nan)
            row[f"{term}_se"] = res.bse.get(term, np.nan)
            row[f"{term}_pval"] = res.pvalues.get(term, np.nan)
            details.append(
                f"{term:<24s} coef={row[f'{term}_coef']:+.6f} "
                f"se={row[f'{term}_se']:.6f} p={row[f'{term}_pval']:.4f}"
            )
        rows.append(row)
        details.append("")

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    DETAILS_PATH.write_text("\n".join(details), encoding="utf-8")
    print(summary.to_string(index=False, float_format="%.6f"))

if __name__ == "__main__":
    main()
