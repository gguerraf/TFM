"""Run local projection regressions"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_extension_utils import (
    ESTIMATION_WINDOW,
    FIG_DIR,
    GAP,
    MAIN_TERMS,
    MODEL_TERMS,
    OUTPUT_DIR,
    fit_absorbed,
    load_benchmark_returns,
    load_panel,
    load_returns,
    result_row,
    select_benchmark,
)

MAX_HORIZON = 10
OUTCOME_PATH = OUTPUT_DIR / "local_projection_outcomes.csv"
SUMMARY_PATH = OUTPUT_DIR / "local_projections_summary.csv"
DETAILS_PATH = OUTPUT_DIR / "local_projections_details.txt"
FIG_PATH = FIG_DIR / "local_projection_irf.png"

def batch_car_paths(ret_stock, ret_bench, event_dates, max_horizon):
    event_dates = pd.to_datetime(pd.Series(event_dates).dropna().unique())
    idx = ret_stock.index.intersection(ret_bench.index).sort_values()
    y_full = ret_stock.reindex(idx).astype(float).fillna(0.0).to_numpy()
    b_full = ret_bench.reindex(idx).astype(float).fillna(0.0).to_numpy()
    pos = idx.get_indexer(event_dates)
    valid = (pos >= ESTIMATION_WINDOW + GAP) & (pos >= 0) & (pos + max_horizon < len(idx))
    dates = event_dates[valid]
    pos = pos[valid]
    cols = [f"AR_j_h{h}" for h in range(max_horizon + 1)]
    if len(pos) == 0:
        return pd.DataFrame(columns=["t0", *cols])
    starts = pos - ESTIMATION_WINDOW - GAP
    win_idx = starts[:, None] + np.arange(ESTIMATION_WINDOW)[None, :]
    x_batch = np.stack([np.ones_like(win_idx, dtype=float), b_full[win_idx]], axis=2)
    y_batch = y_full[win_idx]
    xtx = np.einsum("nwp,nwq->npq", x_batch, x_batch)
    xty = np.einsum("nwp,nw->np", x_batch, y_batch)
    coefs = np.full((len(pos), 2), np.nan)
    try:
        coefs = np.linalg.solve(xtx, xty[..., None]).squeeze(-1)
    except Exception:
        for k in range(len(pos)):
            try:
                coefs[k], *_ = np.linalg.lstsq(x_batch[k], y_batch[k], rcond=None)
            except Exception:
                pass
    ev_idx = pos[:, None] + np.arange(max_horizon + 1)[None, :]
    ar = y_full[ev_idx] - coefs[:, 0, None] - coefs[:, 1, None] * b_full[ev_idx]
    car = np.cumsum(ar, axis=1)
    out = pd.DataFrame(car, columns=cols)
    out.insert(0, "t0", dates)
    return out

def compute_outcomes(panel, returns, bench_returns):
    cached_cols = ["row_id", *[f"AR_j_h{h}" for h in range(MAX_HORIZON + 1)]]
    if OUTCOME_PATH.exists():
        cached = pd.read_csv(OUTCOME_PATH)
        if set(cached_cols).issubset(cached.columns) and len(cached) == len(panel):
            print(f"Loading cached local projection outcomes: {OUTCOME_PATH}")
            return cached[cached_cols]

    outcomes = pd.DataFrame({"row_id": panel["row_id"].to_numpy()})
    for h in range(MAX_HORIZON + 1):
        outcomes[f"AR_j_h{h}"] = np.nan

    for etf, etf_panel in panel.groupby("etf", sort=False):
        print(f"Computing dynamic outcomes for {etf}...")
        _, ret_bench = select_benchmark(etf, returns, bench_returns)
        for stock_j, stock_panel in etf_panel.groupby("stock_j", sort=False):
            if stock_j not in returns.columns:
                continue
            paths = batch_car_paths(returns[stock_j], ret_bench, stock_panel["t0"], MAX_HORIZON)
            if paths.empty:
                continue
            mapping = paths.set_index("t0")
            idx = stock_panel.index
            for h in range(MAX_HORIZON + 1):
                outcomes.loc[idx, f"AR_j_h{h}"] = stock_panel["t0"].map(mapping[f"AR_j_h{h}"]).to_numpy()

    outcomes.to_csv(OUTCOME_PATH, index=False)
    return outcomes

def plot_irf(summary):
    terms = ["b1_term", "b2_term", "b5_term", "receiver_weight_term", "corr_term", "hhi_term"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    axes = axes.ravel()
    for ax, term in zip(axes, terms):
        coef = summary[f"{term}_coef"]
        se = summary[f"{term}_se"]
        h = summary["horizon"]
        ax.plot(h, coef, marker="o", linewidth=1.5)
        ax.fill_between(h, coef - 1.96 * se, coef + 1.96 * se, alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(term)
        ax.set_xlabel("Horizon")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150)
    plt.close()

def main():
    print("Loading panel and returns...")
    panel = load_panel().reset_index(drop=True)
    panel["row_id"] = np.arange(len(panel))
    returns = load_returns()
    bench_returns = load_benchmark_returns()
    outcomes = compute_outcomes(panel, returns, bench_returns)
    panel = panel.merge(outcomes, on="row_id", how="left")

    rows = []
    detail_blocks = []
    for h in range(MAX_HORIZON + 1):
        y_col = f"AR_j_h{h}"
        print(f"Estimating local projection horizon h={h}...")
        res, used = fit_absorbed(panel, y_col=y_col, terms=MODEL_TERMS)
        row = result_row(f"LP_h{h}", res)
        row["horizon"] = h
        row["rows_used"] = len(used)
        rows.append(row)
        detail_blocks.append(f"HORIZON h={h}\n" + "-" * 80)
        for term in MAIN_TERMS:
            coef = res.params.get(term, float("nan"))
            se   = res.std_errors.get(term, float("nan"))
            pval = res.pvalues.get(term, float("nan"))
            detail_blocks.append(
                f"{term:<24s} coef={coef:+.6f} "
                f"se={se:.6f} p={pval:.4f}"
            )
        detail_blocks.append("")

    summary = pd.DataFrame(rows).sort_values("horizon")
    summary.to_csv(SUMMARY_PATH, index=False)
    DETAILS_PATH.write_text("\n".join(detail_blocks), encoding="utf-8")
    plot_irf(summary)

if __name__ == "__main__":
    main()
