"""Run sequential model specification comparisons"""

import time

import numpy as np
import pandas as pd

from model_extension_utils import (
    CONTROL_TERMS,
    MAIN_TERMS,
    MODEL_TERMS,
    OUTPUT_DIR,
    fit_absorbed,
)

PANEL_PATH = OUTPUT_DIR / "panel_improved.csv"
SUMMARY_PATH = OUTPUT_DIR / "model_specification_comparison.csv"
DETAILS_PATH = OUTPUT_DIR / "model_specification_comparison_details.txt"
CLUSTER_COLS = ["event_id", "stock_j"]
REFERENCE_FULL = "full_expanded"
REFERENCE_PREVIOUS = "previous_improved"

SPECIFICATIONS = [
    {
        "specification": "core_b1",
        "description": "Core direct transmission only",
        "terms": ["b1_term"],
    },
    {
        "specification": "core_plus_liquidity",
        "description": "Core direct transmission and liquidity channel",
        "terms": ["b1_term", "b2_term"],
    },
    {
        "specification": "core_liquidity_similarity",
        "description": "Core, liquidity and similarity channels",
        "terms": ["b1_term", "b2_term", "b5_term"],
    },
    {
        "specification": REFERENCE_PREVIOUS,
        "description": "Previous improved specification before receiver weight, correlation and concentration",
        "terms": [
            "b1_term", "b2_term", "b4_term", "b5_term", "asym_term",
            "Illiq_j", "Mispricing_k", "Similarity_ij", "Neg_e",
        ],
    },
    {
        "specification": REFERENCE_FULL,
        "description": "Full expanded specification",
        "terms": MODEL_TERMS,
    },
    {
        "specification": "full_without_receiver_weight",
        "description": "Full model without receiver weight channel",
        "terms": [t for t in MODEL_TERMS if t not in {"receiver_weight_term", "w_j"}],
    },
    {
        "specification": "full_without_correlation",
        "description": "Full model without pre-event correlation channel",
        "terms": [t for t in MODEL_TERMS if t not in {"corr_term", "Corr_ij_60d"}],
    },
    {
        "specification": "full_without_hhi",
        "description": "Full model without ETF concentration channel",
        "terms": [t for t in MODEL_TERMS if t not in {"hhi_term", "HHI_etf_t"}],
    },
    {
        "specification": "full_without_new_variables",
        "description": "Full model without receiver weight, correlation and concentration variables",
        "terms": [
            t for t in MODEL_TERMS
            if t not in {
                "receiver_weight_term", "corr_term", "hhi_term",
                "w_j", "Corr_ij_60d", "HHI_etf_t",
            }
        ],
    },
    {
        "specification": "full_without_main_controls",
        "description": "Full interaction channels without lower-order controls",
        "terms": MAIN_TERMS,
    },
    {
        "specification": "full_without_asymmetry",
        "description": "Full model without negative-shock asymmetry",
        "terms": [t for t in MODEL_TERMS if t not in {"asym_term", "Neg_e"}],
    },
]


def term_value(series, term):
    return series.get(term, np.nan)


def build_row(spec, panel):
    start = time.perf_counter()
    terms = spec["terms"]
    row = {
        "specification": spec["specification"],
        "description": spec["description"],
        "included_variables": ";".join(terms),
        "included_main_channels": ";".join([t for t in terms if t in MAIN_TERMS]),
        "included_controls": ";".join([t for t in terms if t in CONTROL_TERMS]),
        "cluster_cols": ";".join(CLUSTER_COLS),
        "two_way_clustered": False,
        "error": "",
    }
    try:
        res, used = fit_absorbed(panel, terms=terms, cluster_cols=CLUSTER_COLS)
        runtime = time.perf_counter() - start
        significant_terms = [
            term for term in MAIN_TERMS
            if term in terms and term_value(res.pvalues, term) < 0.05
        ]
        row.update({
            "N": res.nobs,
            "unique_events": used["event_id"].nunique(),
            "unique_receivers": used["stock_j"].nunique(),
            "adj_R2": res.rsquared_adj,
            "R2": getattr(res, "rsquared", np.nan),
            "absorbed_R2": getattr(res, "absorbed_rsquared", np.nan),
            "AIC": getattr(res, "aic", np.nan),
            "BIC": getattr(res, "bic", np.nan),
            "significant_channel_count_5pct": len(significant_terms),
            "significant_channels_5pct": ";".join(significant_terms),
            "runtime_seconds": runtime,
            "two_way_clustered": True,
        })
        for term in MAIN_TERMS:
            row[f"{term}_coef"] = term_value(res.params, term)
            row[f"{term}_se"] = term_value(res.std_errors, term)
            row[f"{term}_pval"] = term_value(res.pvalues, term)
    except Exception as exc:
        row["runtime_seconds"] = time.perf_counter() - start
        row["error"] = str(exc)
    return row


def add_reference_changes(summary):
    out = summary.copy()
    full = out.loc[out["specification"] == REFERENCE_FULL, "b1_term_coef"]
    previous = out.loc[out["specification"] == REFERENCE_PREVIOUS, "b1_term_coef"]
    full_b1 = full.iloc[0] if len(full) else np.nan
    previous_b1 = previous.iloc[0] if len(previous) else np.nan
    out["b1_change_vs_full"] = out["b1_term_coef"] - full_b1
    out["b1_change_vs_previous_improved"] = out["b1_term_coef"] - previous_b1
    return out


def write_details(summary):
    lines = ["SEQUENTIAL SPECIFICATION COMPARISON", "=" * 80, ""]
    for _, row in summary.iterrows():
        lines.append(row["specification"])
        lines.append(row["description"])
        lines.append(f"Terms: {row['included_variables']}")
        if row.get("error"):
            lines.append(f"Error: {row['error']}")
        lines.append("")
    DETAILS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["t0"])
    rows = []
    for spec in SPECIFICATIONS:
        print(f"Estimating {spec['specification']}...")
        rows.append(build_row(spec, panel))
    summary = add_reference_changes(pd.DataFrame(rows))
    summary.to_csv(SUMMARY_PATH, index=False)
    write_details(summary)
    display_cols = [
        "specification", "N", "adj_R2", "b1_term_coef", "b1_term_pval",
        "significant_channel_count_5pct", "runtime_seconds", "error",
    ]
    print(summary[display_cols].to_string(index=False, float_format="%.6f"))


if __name__ == "__main__":
    main()