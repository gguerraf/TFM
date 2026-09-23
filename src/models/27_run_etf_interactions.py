"""Run pooled ETF interaction tests"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from model_extension_utils import (
    OUTPUT_DIR,
    MODEL_TERMS,
    MAIN_TERMS,
    fit_absorbed,
    format_result_block,
    load_panel,
    result_row,
)

REFERENCE_ETF = "XME"
INTERACTION_BASE_TERMS = [
    "b1_term", "b2_term", "b4_term", "b5_term",
    "receiver_weight_term", "corr_term", "hhi_term", "asym_term",
]

def add_etf_interactions(panel):
    df = panel.copy()
    etfs = sorted([e for e in df["etf"].dropna().unique() if e != REFERENCE_ETF])
    interaction_terms = []
    for etf in etfs:
        flag = (df["etf"] == etf).astype(float)
        for term in INTERACTION_BASE_TERMS:
            name = f"{term}_x_{etf}"
            df[name] = df[term] * flag
            interaction_terms.append(name)
    return df, interaction_terms

def main():
    print("Loading improved panel...")
    panel = load_panel()
    print(f"  Panel shape: {panel.shape}")

    df, interaction_terms = add_etf_interactions(panel)
    terms = MODEL_TERMS + interaction_terms

    print("Estimating pooled ETF-interaction model...")
    res, used = fit_absorbed(df, terms=terms)
    rows = [result_row("pooled_etf_interactions", res, terms=MAIN_TERMS + interaction_terms)]
    summary = pd.DataFrame(rows)

    out_csv = OUTPUT_DIR / "etf_interactions_summary.csv"
    out_txt = OUTPUT_DIR / "etf_interactions_details.txt"
    summary.to_csv(out_csv, index=False)

    lines = [
        "POOLED ETF-SPECIFIC INTERACTION MODEL",
        f"Reference ETF: {REFERENCE_ETF}",
        f"Regression rows used: {len(used):,}",
        "",
        format_result_block("Pooled ETF interactions", res, terms=MAIN_TERMS + interaction_terms),
    ]
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    print(summary.to_string(index=False, float_format="%.6f"))

if __name__ == "__main__":
    main()
