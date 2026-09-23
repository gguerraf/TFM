"""Estimate the baseline spillover model"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR   = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR   = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

VALIDATION_MODE = False

if VALIDATION_MODE:
    ETF_LIST = ["XME", "XLE"]
    print("Running in VALIDATION MODE: XME and XLE only.")
else:
    ETF_LIST = ["SPY", "XME", "XLE", "IHE", "XLV"]
    print("Running in FULL MODE: all 5 ETFs.")

ETF_BENCHMARK_CANDIDATES = {
    "SPY": ["^GSPC"],
    "XME": ["XLB", "^SP500-15", "^GSPC"],
    "XLE": ["^GSPC"],
    "IHE": ["XLV", "^SP500-35", "^GSPC"],
    "XLV": ["^SP500-35", "VHT", "^GSPC"],
}

ESTIMATION_WINDOW = 120
GAP               = 5
EVENT_H           = 3

SHOCK_THRESHOLD = 1.5

ILLIQ_WINDOW      = 20
MISPRICING_WINDOW = 5

print("\nLoading returns_clean.csv...")
_peek      = pd.read_csv(PROC_DIR / "returns_clean.csv", nrows=0)
_idx_col   = _peek.columns[0]
returns    = pd.read_csv(PROC_DIR / "returns_clean.csv",
                         index_col=_idx_col, parse_dates=True)
returns.index = pd.to_datetime(returns.index)
print(f"  Returns shape: {returns.shape}")

print("Loading benchmarks.csv...")
_bpeak   = pd.read_csv(PROC_DIR / "benchmarks.csv", nrows=0)
_bidx    = _bpeak.columns[0]
benchmarks_df = pd.read_csv(PROC_DIR / "benchmarks.csv",
                             index_col=_bidx, parse_dates=True)
benchmarks_df.index = pd.to_datetime(benchmarks_df.index)
bench_returns = np.log(benchmarks_df / benchmarks_df.shift(1)).iloc[1:]
print(f"  Benchmark returns shape: {bench_returns.shape}")

mapping_path = PROC_DIR / "benchmark_mapping.csv"
if mapping_path.exists():
    _map_df       = pd.read_csv(mapping_path)
    ETF_BENCHMARK = dict(zip(_map_df["etf"], _map_df["benchmark"]))
    print(f"  Benchmark mapping: {ETF_BENCHMARK}")
else:
    ETF_BENCHMARK = {
        "SPY": "^GSPC", "XME": "XLB",
        "XLE": "^GSPC", "IHE": "XLV", "XLV": "^SP500-35",
    }
    print("  [WARN] benchmark_mapping.csv not found. Using hardcoded mapping.")

gics_path = PROC_DIR / "gics_data.csv"
if gics_path.exists():
    gics_df = pd.read_csv(gics_path, index_col="ticker")
    print(f"  GICS data loaded: {len(gics_df)} tickers classified")
    print(f"  Unique sectors  : {gics_df['sector'].nunique()}")
    print(f"  Unique industries: {gics_df['industry'].nunique()}")
else:
    gics_df = pd.DataFrame(columns=["sector", "industry"])
    print("  [WARN] gics_data.csv not found. Similarity_ij will be binary (0/1).")

amihud_path = PROC_DIR / "amihud.csv"
if amihud_path.exists():
    _apeak   = pd.read_csv(amihud_path, nrows=0)
    _aidx    = _apeak.columns[0]
    amihud_df = pd.read_csv(amihud_path, index_col=_aidx, parse_dates=True)
    amihud_df.index = pd.to_datetime(amihud_df.index)
    print(f"  Amihud data loaded: {amihud_df.shape}")
else:
    amihud_df = pd.DataFrame()
    print("  [WARN] amihud.csv not found. Falling back to volatility proxy.")

def load_holdings(etf: str) -> pd.DataFrame:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv",
                     parse_dates=["atDate"])
    return df

def select_benchmark(etf: str) -> tuple:
    """Selects the benchmark for a given ETF"""
    ticker = ETF_BENCHMARK.get(etf)
    if ticker and ticker in bench_returns.columns:
        print(f"  Benchmark for {etf}: {ticker} (benchmarks.csv)")
        return ticker, bench_returns[ticker].dropna()
    candidates = ETF_BENCHMARK_CANDIDATES.get(etf, ["^GSPC"])
    for cand in candidates:
        if cand in returns.columns:
            print(f"  Benchmark for {etf}: {cand} (returns_clean.csv fallback)")
            return cand, returns[cand].dropna()
    if etf in returns.columns:
        print(f"  [WARN] {etf}: using ETF itself as benchmark (endogeneity risk)")
        return etf, returns[etf].dropna()
    raise ValueError(f"No benchmark found for {etf}")

def identify_shocks_vectorized(ret_stocks: pd.DataFrame,
                                ret_bench:  pd.Series) -> pd.DataFrame:
    """Identifies shock events for all constituent stocks using vectorized"""

    bench = ret_bench.reindex(ret_stocks.index).fillna(0).values
    dates = ret_stocks.index
    T     = len(dates)
    records = []

    X_full = np.column_stack([np.ones(T), bench])

    min_event_idx = ESTIMATION_WINDOW + GAP
    max_event_idx = T - EVENT_H - 1

    if min_event_idx >= max_event_idx:
        print("  [WARN] Not enough data for shock identification.")
        return pd.DataFrame()

    n_events = max_event_idx - min_event_idx
    print(f"  Eligible event dates: {n_events:,}")

    event_indices = np.arange(min_event_idx, max_event_idx)

    est_starts = event_indices - ESTIMATION_WINDOW - GAP
    est_ends   = event_indices - GAP

    valid_mask  = est_starts >= 0
    event_indices = event_indices[valid_mask]
    est_starts    = est_starts[valid_mask]
    est_ends      = est_ends[valid_mask]
    n_valid       = len(event_indices)

    print(f"  Valid event dates (full estimation window): {n_valid:,}")

    n_stocks = ret_stocks.shape[1]

    for s_idx, ticker in enumerate(ret_stocks.columns):
        if s_idx % 10 == 0:
            print(f"    Processing stock {s_idx+1}/{n_stocks}: {ticker}")

        y_full = ret_stocks[ticker].fillna(0).values

        alphas = np.full(n_valid, np.nan)
        betas  = np.full(n_valid, np.nan)

        idx_matrix = (est_starts[:, None] +
                      np.arange(ESTIMATION_WINDOW)[None, :])

        Y_batch = y_full[idx_matrix]
        X_batch = X_full[idx_matrix]

        XtX = np.einsum('kij,kil->kjl', X_batch, X_batch)
        Xty = np.einsum('kij,ki->kj',   X_batch, Y_batch)

        det  = XtX[:, 0, 0] * XtX[:, 1, 1] - XtX[:, 0, 1] * XtX[:, 1, 0]
        safe = np.abs(det) > 1e-12

        alphas[safe] = (XtX[safe, 1, 1] * Xty[safe, 0]
                        - XtX[safe, 0, 1] * Xty[safe, 1]) / det[safe]
        betas[safe]  = (XtX[safe, 0, 0] * Xty[safe, 1]
                        - XtX[safe, 1, 0] * Xty[safe, 0]) / det[safe]

        ev_idx_matrix = (event_indices[:, None] +
                         np.arange(EVENT_H + 1)[None, :])

        ev_idx_matrix = np.clip(ev_idx_matrix, 0, T - 1)

        R_event  = y_full[ev_idx_matrix]
        Rm_event = bench[ev_idx_matrix]

        AR = (R_event
              - alphas[:, None]
              - betas[:, None] * Rm_event)

        CARs = AR.sum(axis=1)

        valid_cars = CARs[~np.isnan(alphas)]
        if len(valid_cars) < 10:
            continue

        car_mean = np.nanmean(valid_cars)
        car_std  = np.nanstd(valid_cars)
        if car_std < 1e-10:
            continue

        for k in range(n_valid):
            if np.isnan(alphas[k]) or np.isnan(CARs[k]):
                continue
            if abs(CARs[k] - car_mean) > SHOCK_THRESHOLD * car_std:
                records.append({
                    "t0":          dates[event_indices[k]],
                    "stock_i":     ticker,
                    "car_i":       CARs[k],
                    "is_negative": int(CARs[k] < 0),
                })

    return pd.DataFrame(records)

def illiquidity_proxy(ticker: str, t0: pd.Timestamp,
                       ret: pd.Series) -> float:
    """Amihud (2002) illiquidity ratio, computed as the rolling mean of"""
    if not amihud_df.empty and ticker in amihud_df.columns:
        past = amihud_df.index[amihud_df.index < t0]
        if len(past) > 0:
            val = amihud_df.loc[past[-1], ticker]
            if not np.isnan(val):
                return float(val)

    past = ret.index[ret.index < t0]
    if len(past) < ILLIQ_WINDOW:
        return np.nan
    return float(ret.reindex(past[-ILLIQ_WINDOW:]).std())

def similarity(ticker_i: str, ticker_j: str) -> float:
    """GICS-based economic similarity between two stocks"""
    if gics_df.empty:
        return 1.0

    row_i = gics_df.loc[ticker_i] if ticker_i in gics_df.index else None
    row_j = gics_df.loc[ticker_j] if ticker_j in gics_df.index else None

    if row_i is None or row_j is None:
        return 0.5

    sector_i   = row_i.get("sector",   None)
    sector_j   = row_j.get("sector",   None)
    industry_i = row_i.get("industry", None)
    industry_j = row_j.get("industry", None)

    if pd.isna(sector_i) or pd.isna(sector_j):
        return 0.5

    if sector_i != sector_j:
        return 0.0

    if pd.isna(industry_i) or pd.isna(industry_j):
        return 0.5

    if industry_i == industry_j:
        return 1.0

    return 0.5

def mispricing_proxy(etf_ret: pd.Series,
                     bench_ret: pd.Series,
                     t0: pd.Timestamp) -> float:
    """ETF premium/discount proxy: cumulative ETF return minus benchmark"""
    past = etf_ret.index[etf_ret.index < t0]
    if len(past) < MISPRICING_WINDOW:
        return 0.0
    w = past[-MISPRICING_WINDOW:]
    return float(etf_ret.reindex(w).sum() - bench_ret.reindex(w).sum())

def compute_car_single(ret_stock: pd.Series,
                        ret_bench: pd.Series,
                        t0: pd.Timestamp) -> float:
    """Computes CAR for a single stock at a single event date"""
    pre = ret_stock.index[ret_stock.index < t0]
    if len(pre) < ESTIMATION_WINDOW + GAP:
        return np.nan

    est_end   = pre[-(GAP + 1)]
    est_start = pre[-(ESTIMATION_WINDOW + GAP)]

    mask = (ret_stock.index >= est_start) & (ret_stock.index <= est_end)
    y    = ret_stock[mask].fillna(0).values
    x    = ret_bench.reindex(ret_stock[mask].index).fillna(0).values

    if len(y) < 30:
        return np.nan

    X = np.column_stack([np.ones(len(x)), x])
    try:
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return np.nan

    alpha, beta = coefs[0], coefs[1]

    future = ret_stock.index[ret_stock.index >= t0]
    if len(future) < EVENT_H + 1:
        return np.nan

    w_dates = future[:EVENT_H + 1]
    ar = (ret_stock.reindex(w_dates).fillna(0).values
          - alpha
          - beta * ret_bench.reindex(w_dates).fillna(0).values)
    return float(ar.sum())

def build_panel(shocks:      pd.DataFrame,
                holdings_df: pd.DataFrame,
                ret_stocks:  pd.DataFrame,
                ret_bench:   pd.Series,
                etf_ret:     pd.Series,
                etf_symbol:  str) -> pd.DataFrame:
    """Builds the observation panel (j, e) for the spillover regression"""
    rows = []
    hold_dates = sorted(holdings_df["atDate"].unique())
    n_shocks   = len(shocks)

    for evt_idx, (_, shock) in enumerate(shocks.iterrows()):
        if evt_idx % 500 == 0:
            print(f"    Building panel: event {evt_idx}/{n_shocks}...")

        t0      = shock["t0"]
        stock_i = shock["stock_i"]
        car_i   = shock["car_i"]
        is_neg  = shock["is_negative"]

        past_hold = [d for d in hold_dates if d <= t0]
        if not past_hold:
            continue
        t_hold = past_hold[-1]
        hold_t = holdings_df[holdings_df["atDate"] == t_hold]

        row_i = hold_t[hold_t["symbol"] == stock_i]
        if row_i.empty:
            continue
        w_i = float(row_i["weight"].iloc[0])
        if np.isnan(w_i) or w_i <= 0:
            continue

        mispricing = mispricing_proxy(etf_ret, ret_bench, t0)

        peers = hold_t[hold_t["symbol"] != stock_i]["symbol"].tolist()

        for stock_j in peers:
            if stock_j not in ret_stocks.columns:
                continue

            r_j  = ret_stocks[stock_j]
            ar_j = compute_car_single(r_j, ret_bench, t0)
            if np.isnan(ar_j):
                continue

            illiq_j    = illiquidity_proxy(stock_j, t0, r_j)
            sim_ij     = similarity(stock_i, stock_j)
            illiq_safe = illiq_j if not np.isnan(illiq_j) else 0.0

            rows.append({
                "event_id":      f"{stock_i}_{t0.date()}",
                "etf":           etf_symbol,
                "stock_j":       stock_j,
                "stock_i":       stock_i,
                "t0":            t0,
                "AR_j":          ar_j,
                "Shock_i":       car_i,
                "w_i":           w_i,
                "Illiq_j":       illiq_j,
                "Mispricing_k":  mispricing,
                "Similarity_ij": sim_ij,
                "Neg_e":         is_neg,

                "b1_term":   car_i * w_i,
                "b2_term":   car_i * w_i * illiq_safe,
                "b4_term":   car_i * w_i * mispricing,
                "b5_term":   car_i * sim_ij,
                "asym_term": is_neg * car_i * w_i,
            })

    return pd.DataFrame(rows)

def run_regression(panel: pd.DataFrame) -> None:
    """Estimates the core intra-ETF spillover regression with stock fixed effects"""
    required = ["AR_j", "b1_term", "b2_term", "b4_term", "b5_term", "asym_term"]
    df = panel.dropna(subset=required)

    if len(df) < 100:
        print(f"[WARN] Panel too small ({len(df)} obs). "
              f"Try lowering SHOCK_THRESHOLD.")
        return

    print(f"\nEstimating regression on {len(df):,} observations...")

    formula = ("AR_j ~ b1_term + b2_term + b4_term + b5_term + asym_term"
               " + C(stock_j)")

    result = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["event_id"]}
    )

    main_terms = ["b1_term", "b2_term", "b4_term", "b5_term", "asym_term"]

    print("\n" + "=" * 70)
    print("INTRA-ETF SPILLOVER REGRESSION RESULTS")
    print("=" * 70)
    tbl = result.summary2().tables[1]

    available_cols = [c for c in ["Coef.", "Std.Err.", "t", "P>|t|",
                                   "z", "P>|z|", "[0.025", "0.975]"]
                      if c in tbl.columns]
    print(tbl.loc[main_terms, available_cols])
    print(f"\nN observations : {result.nobs:.0f}")
    print(f"Adjusted R-sq  : {result.rsquared_adj:.4f}")

    coefs  = result.params
    pvals  = result.pvalues
    labels = {
        "b1_term":   "beta1  Baseline propagation      (Shock x w_i)",
        "b2_term":   "beta2  Liquidity channel          (Shock x w_i x Illiq_j)",
        "b4_term":   "beta4  Arbitrage channel          (Shock x w_i x Mispricing)",
        "b5_term":   "beta5  Informational spillover    (Shock x Similarity)",
        "asym_term": "delta  Negative asymmetry         (Neg x Shock x w_i)",
    }
    txt_path = OUTPUT_DIR / "regression_results.txt"
    with open(txt_path, "w") as f:
        f.write(result.summary().as_text())

    _plot_coefs(coefs, pvals, main_terms, labels)

def _plot_coefs(coefs, pvals, terms, labels):
    """Saves a coefficient plot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    vals   = [coefs[t] for t in terms]
    short  = [labels[t].split()[0] for t in terms]
    colors = ["#E24B4A" if pvals[t] < 0.05 else "#888780" for t in terms]
    ax.barh(short, vals, color=colors, height=0.5)
    ax.axvline(0, color="#2C2C2A", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient estimate")
    ax.set_title("Spillover regression coefficients  (red = p<0.05)",
                 fontsize=11)
    plt.tight_layout()
    out = OUTPUT_DIR / "coef_plot.png"
    plt.savefig(out, dpi=150)
    plt.close()

def asymmetry_analysis(panel: pd.DataFrame) -> None:
    """Estimates the model separately for positive and negative shocks (H5)"""
    for sign, label in [(0, "POSITIVE"), (1, "NEGATIVE")]:
        sub = panel[panel["Neg_e"] == sign].dropna(
            subset=["AR_j", "b1_term", "b2_term", "b4_term", "b5_term"])
        if len(sub) < 50:
            print(f"[WARN] Too few {label} shocks ({len(sub)}). Skipping.")
            continue
        formula = "AR_j ~ b1_term + b2_term + b4_term + b5_term + C(stock_j)"
        res = smf.ols(formula, data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["event_id"]})
        print(f"\n{'=' * 60}")
        print(f"SUBSAMPLE: {label} SHOCKS  (N={res.nobs:.0f})")
        print(f"{'=' * 60}")
        tbl = res.summary2().tables[1]
        available_cols = [c for c in ["Coef.", "Std.Err.", "t", "P>|t|",
                                       "z", "P>|z|"]
                          if c in tbl.columns]
        print(tbl.loc[["b1_term","b2_term","b4_term","b5_term"],
                       available_cols])

if __name__ == "__main__":

    panel_path = OUTPUT_DIR / "panel_full.csv"
    if panel_path.exists():
        print("")
        print(f"Loading existing panel from {panel_path}...")
        full_panel = pd.read_csv(panel_path, parse_dates=["t0"])
        print(f"  Panel shape: {full_panel.shape}")
        run_regression(full_panel)
        asymmetry_analysis(full_panel)
        import sys; sys.exit(0)

    panels_all = []

    for etf in ETF_LIST:
        print(f"\n{'=' * 60}")
        print(f"ETF: {etf}")
        print(f"{'=' * 60}")

        hold_df = load_holdings(etf)

        try:
            bench_sym, ret_bench = select_benchmark(etf)
        except ValueError as e:
            print(f"  [ERROR] {e}. Skipping {etf}.")
            continue

        etf_ret = returns[etf].dropna() if etf in returns.columns else ret_bench

        etf_stocks   = hold_df["symbol"].unique().tolist()
        avail_stocks = [t for t in etf_stocks if t in returns.columns
                        and t != bench_sym]
        ret_stocks   = returns[avail_stocks]

        print(f"  Holdings symbols    : {len(etf_stocks):,}")
        print(f"  With return data    : {len(avail_stocks):,}")

        if len(avail_stocks) < 5:
            print(f"  [WARN] Too few stocks. Skipping {etf}.")
            continue

        print(f"\n  Identifying shocks "
              f"(threshold={SHOCK_THRESHOLD}sigma, vectorized)...")
        shocks = identify_shocks_vectorized(ret_stocks, ret_bench)
        print(f"  Shocks identified: {len(shocks):,}")

        if shocks.empty:
            print(f"  [WARN] No shocks found. Try lowering SHOCK_THRESHOLD.")
            continue

        n_neg = shocks["is_negative"].sum()
        n_pos = len(shocks) - n_neg
        print(f"  Positive shocks: {n_pos:,}  |  Negative shocks: {n_neg:,}")

        print(f"\n  Building observation panel (j, e)...")
        panel = build_panel(shocks, hold_df, ret_stocks,
                            ret_bench, etf_ret, etf)
        print(f"  Panel observations: {len(panel):,}")

        if not panel.empty:
            panels_all.append(panel)

    if not panels_all:
        print("\n[ERROR] No panels built. Check data and configuration.")
    else:
        full_panel = pd.concat(panels_all, ignore_index=True)
        print(f"\nFull panel: {len(full_panel):,} observations")

        panel_path = OUTPUT_DIR / "panel_full.csv"
        full_panel.to_csv(panel_path, index=False)

        run_regression(full_panel)
        asymmetry_analysis(full_panel)

