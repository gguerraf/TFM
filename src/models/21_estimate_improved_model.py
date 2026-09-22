"""
21_estimate_improved_model.py
====================
Improved Intra-ETF spillover model.

Improvements over 20_estimate_baseline_model.py (the baseline):
  C1 - Fixed lookahead bias: shock threshold is now event-specific,
       using only the estimation-window residual std (Patell 1976).
  C2 - XLE benchmark changed from ^GSPC to IXC (iShares Global Energy).
  C3 - SPY excluded from the main analysis (self-benchmarking problem).
  C4 - Lower-order main effects added (Illiq_j, Mispricing_k, etc.).
  C5 - Two-way clustered SEs (event_id x stock_j).
  C6 - Overlapping/contaminated events filtered.
  C7 - Illiquidity: log-transformed, NaN-preserving (no 0.0 imputation).
  D2 - Year-quarter time fixed effects added.

Inputs:
    processed/{etf}_holdings.csv   (from 01_load_holdings.py)
    processed/returns_clean.csv    (from 11_compute_returns.py)
    processed/benchmarks.csv       (from 06_download_benchmarks.py)
    processed/gics_data.csv        (from 08_download_gics.py)
    processed/amihud.csv           (from 10_download_volume.py)

Outputs:
    results/panel_improved.csv
    results/regression_improved.txt
    results/coef_plot_improved.png
    results/shock_diagnostics.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv.absorbing import AbsorbingLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from bisect import bisect_left, bisect_right

# ================================================================================
# CONFIGURATION
# ================================================================================

BASE_DIR   = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR   = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR    = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# -- Key change C3: SPY excluded from main analysis --
ETF_LIST = ["XME", "XLE", "IHE", "XLV"]

# -- Key change C2: XLE now uses IXC instead of ^GSPC --
ETF_BENCHMARK = {
    "XME": "XLB",
    "XLE": "IXC",         # Was ^GSPC - now iShares Global Energy ETF
    "IHE": "XLV",
    "XLV": "^SP500-35",
}

ETF_BENCHMARK_CANDIDATES = {
    "XME": ["XLB", "^SP500-15", "^GSPC"],
    "XLE": ["IXC", "^GSPC"],        # IXC first
    "IHE": ["XLV", "^SP500-35", "^GSPC"],
    "XLV": ["^SP500-35", "VHT", "^GSPC"],
}

# Market model parameters
ESTIMATION_WINDOW = 120   # trading days for OLS estimation
GAP               = 5     # days between estimation window end and event date
EVENT_H           = 3     # CAR window: [t0, t0 + EVENT_H]

# -- Key change C1: event-specific threshold --
# Shock flagged when |CAR| > SHOCK_THRESHOLD * sqrt(H+1) * sigma_epsilon
# where sigma_epsilon is from the estimation window residuals.
SHOCK_THRESHOLD = 1.5

# Mechanism variable windows
ILLIQ_WINDOW      = 20
MISPRICING_WINDOW = 5

# ================================================================================
# STEP 0: LOAD DATA
# ================================================================================

print("\nLoading returns_clean.csv...")
_peek      = pd.read_csv(PROC_DIR / "returns_clean.csv", nrows=0)
_idx_col   = _peek.columns[0]
returns    = pd.read_csv(PROC_DIR / "returns_clean.csv",
                         index_col=_idx_col, parse_dates=True)
returns.index = pd.to_datetime(returns.index)
print(f"  Returns shape: {returns.shape}")

# Load benchmark prices
print("Loading benchmarks.csv...")
_bpeak   = pd.read_csv(PROC_DIR / "benchmarks.csv", nrows=0)
_bidx    = _bpeak.columns[0]
benchmarks_df = pd.read_csv(PROC_DIR / "benchmarks.csv",
                             index_col=_bidx, parse_dates=True)
benchmarks_df.index = pd.to_datetime(benchmarks_df.index)
bench_returns = np.log(benchmarks_df / benchmarks_df.shift(1)).iloc[1:]
print(f"  Benchmark returns shape: {bench_returns.shape}")

# Load GICS sector/industry classifications
gics_path = PROC_DIR / "gics_data.csv"
if gics_path.exists():
    gics_df = pd.read_csv(gics_path, index_col="ticker")
    print(f"  GICS data loaded: {len(gics_df)} tickers classified")
else:
    gics_df = pd.DataFrame(columns=["sector", "industry"])
    print("  [WARN] gics_data.csv not found.")

# Load Amihud illiquidity
amihud_path = PROC_DIR / "amihud.csv"
if amihud_path.exists():
    _apeak   = pd.read_csv(amihud_path, nrows=0)
    _aidx    = _apeak.columns[0]
    amihud_df = pd.read_csv(amihud_path, index_col=_aidx, parse_dates=True)
    amihud_df.index = pd.to_datetime(amihud_df.index)
    # -- Key change C7/D6: log-transform Amihud --
    # Scale factor 1e10 brings values into a reasonable range before log
    amihud_log = np.log1p(amihud_df * 1e10)
    print(f"  Amihud data loaded and log-transformed: {amihud_df.shape}")
else:
    amihud_df = pd.DataFrame()
    amihud_log = pd.DataFrame()
    print("  [WARN] amihud.csv not found.")


def load_holdings(etf: str) -> pd.DataFrame:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv",
                     parse_dates=["atDate"])
    return df


def select_benchmark(etf: str) -> tuple:
    """
    Selects the benchmark for a given ETF.
    Priority: ETF_BENCHMARK mapping -> bench_returns -> returns_clean fallback.
    """
    ticker = ETF_BENCHMARK.get(etf)
    if ticker and ticker in bench_returns.columns:
        print(f"  Benchmark for {etf}: {ticker} (benchmarks.csv)")
        return ticker, bench_returns[ticker].dropna()
    # Also check if IXC is in the main returns matrix
    if ticker and ticker in returns.columns:
        print(f"  Benchmark for {etf}: {ticker} (returns_clean.csv)")
        return ticker, returns[ticker].dropna()
    candidates = ETF_BENCHMARK_CANDIDATES.get(etf, ["^GSPC"])
    for cand in candidates:
        if cand in bench_returns.columns:
            print(f"  Benchmark for {etf}: {cand} (benchmarks.csv fallback)")
            return cand, bench_returns[cand].dropna()
        if cand in returns.columns:
            print(f"  Benchmark for {etf}: {cand} (returns_clean.csv fallback)")
            return cand, returns[cand].dropna()
    raise ValueError(f"No benchmark found for {etf}")


# ================================================================================
# STEP 1: SHOCK IDENTIFICATION (event-specific threshold, no lookahead)
# ================================================================================

def identify_shocks_vectorized(ret_stocks: pd.DataFrame,
                                ret_bench:  pd.Series) -> pd.DataFrame:
    """
    Identifies shock events using event-specific thresholds.

    Key change from baseline: Instead of computing mean/std of CARs over the
    entire sample (lookahead bias), each event's threshold is computed from
    the estimation-window residual standard deviation:

        threshold = SHOCK_THRESHOLD * sqrt(H+1) * sigma_epsilon(t0)

    This follows Patell (1976) and standard event-study methodology.
    """
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

    event_indices = np.arange(min_event_idx, max_event_idx)
    est_starts = event_indices - ESTIMATION_WINDOW - GAP
    est_ends   = event_indices - GAP

    valid_mask    = est_starts >= 0
    event_indices = event_indices[valid_mask]
    est_starts    = est_starts[valid_mask]
    est_ends      = est_ends[valid_mask]
    n_valid       = len(event_indices)

    print(f"  Valid event dates: {n_valid:,}")
    n_stocks = ret_stocks.shape[1]

    for s_idx, ticker in enumerate(ret_stocks.columns):
        if s_idx % 10 == 0:
            print(f"    Processing stock {s_idx+1}/{n_stocks}: {ticker}")

        y_full = ret_stocks[ticker].fillna(0).values

        # Batch estimation windows
        idx_matrix = (est_starts[:, None] +
                      np.arange(ESTIMATION_WINDOW)[None, :])
        Y_batch = y_full[idx_matrix]
        X_batch = X_full[idx_matrix]

        # Batch OLS via normal equations
        XtX = np.einsum('kij,kil->kjl', X_batch, X_batch)
        Xty = np.einsum('kij,ki->kj',   X_batch, Y_batch)

        det  = XtX[:, 0, 0] * XtX[:, 1, 1] - XtX[:, 0, 1] * XtX[:, 1, 0]
        safe = np.abs(det) > 1e-12

        alphas = np.full(n_valid, np.nan)
        betas  = np.full(n_valid, np.nan)

        alphas[safe] = (XtX[safe, 1, 1] * Xty[safe, 0]
                        - XtX[safe, 0, 1] * Xty[safe, 1]) / det[safe]
        betas[safe]  = (XtX[safe, 0, 0] * Xty[safe, 1]
                        - XtX[safe, 1, 0] * Xty[safe, 0]) / det[safe]

        # -- Key change C1: compute residual std per estimation window --
        # Predicted Y in estimation window
        Y_pred = alphas[:, None] + betas[:, None] * X_batch[:, :, 1]
        residuals = Y_batch - Y_pred                        # (n_valid, W)
        sigma_eps = np.nanstd(residuals, axis=1, ddof=2)    # (n_valid,)

        # Event-specific CAR threshold: SHOCK_THRESHOLD * sqrt(H+1) * sigma_eps
        car_threshold = SHOCK_THRESHOLD * np.sqrt(EVENT_H + 1) * sigma_eps

        # Event window CARs
        ev_idx_matrix = (event_indices[:, None] +
                         np.arange(EVENT_H + 1)[None, :])
        ev_idx_matrix = np.clip(ev_idx_matrix, 0, T - 1)

        R_event  = y_full[ev_idx_matrix]
        Rm_event = bench[ev_idx_matrix]

        AR = (R_event
              - alphas[:, None]
              - betas[:, None] * Rm_event)
        CARs = AR.sum(axis=1)

        # Flag shocks using event-specific threshold
        for k in range(n_valid):
            if np.isnan(alphas[k]) or np.isnan(CARs[k]):
                continue
            if np.isnan(car_threshold[k]) or car_threshold[k] < 1e-10:
                continue
            if abs(CARs[k]) > car_threshold[k]:
                records.append({
                    "t0":              dates[event_indices[k]],
                    "stock_i":         ticker,
                    "car_i":           CARs[k],
                    "is_negative":     int(CARs[k] < 0),
                    "sigma_eps":       sigma_eps[k],
                    "car_threshold":   car_threshold[k],
                    "alpha":           alphas[k],
                    "beta":            betas[k],
                })

    return pd.DataFrame(records)


# ================================================================================
# STEP 2: MECHANISM VARIABLE UTILITIES
# ================================================================================

def illiquidity_proxy(ticker: str, t0: pd.Timestamp) -> float:
    """
    Returns log-transformed Amihud illiquidity for stock at t0-1.
    Key change C7: Returns np.nan if missing (no 0.0 imputation).
    Key change D6: Uses log(1 + 1e10 * ILLIQ) instead of raw Amihud.
    """
    if amihud_log.empty or ticker not in amihud_log.columns:
        return np.nan
    past = amihud_log.index[amihud_log.index < t0]
    if len(past) == 0:
        return np.nan
    val = amihud_log.loc[past[-1], ticker]
    if np.isnan(val):
        return np.nan
    return float(val)


def similarity(ticker_i: str, ticker_j: str) -> float:
    """
    GICS-based economic similarity between two stocks.
    1.00 = same industry, 0.50 = same sector, 0.00 = different sector.
    """
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
    """
    ETF premium/discount proxy: cumulative ETF return minus benchmark
    over a short pre-event window.
    """
    past = etf_ret.index[etf_ret.index < t0]
    if len(past) < MISPRICING_WINDOW:
        return 0.0
    w = past[-MISPRICING_WINDOW:]
    return float(etf_ret.reindex(w).sum() - bench_ret.reindex(w).sum())


# ================================================================================
# STEP 3: BUILD OBSERVATION PANEL
# ================================================================================

def compute_car_single(ret_stock: pd.Series,
                        ret_bench: pd.Series,
                        t0: pd.Timestamp) -> float:
    """
    Computes CAR for a single receiver stock at a single event date.
    """
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
    """
    Builds the observation panel (j, e) for the spillover regression.

    Key change C6: Filters out receiver stocks that have their own shock
    within [t0 - EVENT_H, t0 + EVENT_H] to avoid contamination.
    """
    rows = []
    hold_dates = sorted(pd.to_datetime(holdings_df["atDate"].unique()).tolist())
    n_shocks   = len(shocks)
    holdings_cache = {}
    car_cache = {}
    illiq_cache = {}
    similarity_cache = {}

    # Build a per-stock shock calendar for fast contamination checks.
    shock_lookup = {}
    for _, s in shocks.iterrows():
        stock = s["stock_i"]
        if stock not in shock_lookup:
            shock_lookup[stock] = []
        shock_lookup[stock].append(s["t0"])
    for stock, dates in shock_lookup.items():
        shock_lookup[stock] = sorted(pd.to_datetime(dates).tolist())

    for evt_idx, (_, shock) in enumerate(shocks.iterrows()):
        if evt_idx % 500 == 0:
            print(f"    Building panel: event {evt_idx}/{n_shocks}...", flush=True)

        t0      = shock["t0"]
        stock_i = shock["stock_i"]
        car_i   = shock["car_i"]
        is_neg  = shock["is_negative"]

        pos = bisect_right(hold_dates, t0) - 1
        if pos < 0:
            continue
        t_hold = hold_dates[pos]
        if t_hold not in holdings_cache:
            holdings_cache[t_hold] = holdings_df[holdings_df["atDate"] == t_hold]
        hold_t = holdings_cache[t_hold]

        # Weight of shocked stock i
        row_i = hold_t[hold_t["symbol"] == stock_i]
        if row_i.empty:
            continue
        w_i = float(row_i["weight"].iloc[0])
        if np.isnan(w_i) or w_i <= 0:
            continue

        # Mechanism variables at t0
        mispricing = mispricing_proxy(etf_ret, ret_bench, t0)

        # -- D2: Year-quarter for time fixed effects --
        year_quarter = f"{t0.year}Q{(t0.month - 1) // 3 + 1}"

        # Co-constituents j != i
        peers = hold_t[hold_t["symbol"] != stock_i]["symbol"].tolist()

        for stock_j in peers:
            if stock_j not in ret_stocks.columns:
                continue

            # -- C6: skip if receiver j has its own shock near t0 --
            if stock_j in shock_lookup:
                j_shock_dates = shock_lookup[stock_j]
                lo = t0 - pd.Timedelta(days=EVENT_H * 2)
                hi = t0 + pd.Timedelta(days=EVENT_H * 2)
                first = bisect_left(j_shock_dates, lo)
                overlap = (first < len(j_shock_dates)
                           and j_shock_dates[first] <= hi)
                if overlap:
                    continue

            r_j  = ret_stocks[stock_j]
            car_key = (stock_j, t0)
            if car_key not in car_cache:
                car_cache[car_key] = compute_car_single(r_j, ret_bench, t0)
            ar_j = car_cache[car_key]
            if np.isnan(ar_j):
                continue

            illiq_key = (stock_j, t0)
            if illiq_key not in illiq_cache:
                illiq_cache[illiq_key] = illiquidity_proxy(stock_j, t0)
            illiq_j = illiq_cache[illiq_key]

            sim_key = (stock_i, stock_j)
            if sim_key not in similarity_cache:
                similarity_cache[sim_key] = similarity(stock_i, stock_j)
            sim_ij = similarity_cache[sim_key]

            # Interaction terms (use NaN-safe illiq for interactions)
            illiq_safe = illiq_j if not np.isnan(illiq_j) else np.nan

            rows.append({
                "event_id":      f"{stock_i}_{t0.date()}",
                "etf":           etf_symbol,
                "stock_j":       stock_j,
                "stock_i":       stock_i,
                "t0":            t0,
                "year_quarter":  year_quarter,
                "AR_j":          ar_j,
                "Shock_i":       car_i,
                "w_i":           w_i,
                "Illiq_j":       illiq_j,
                "Mispricing_k":  mispricing,
                "Similarity_ij": sim_ij,
                "Neg_e":         is_neg,
                # Pre-computed interaction terms
                "b1_term":   car_i * w_i,
                "b2_term":   car_i * w_i * illiq_safe,
                "b4_term":   car_i * w_i * mispricing,
                "b5_term":   car_i * sim_ij,
                "asym_term": is_neg * car_i * w_i,
            })

    return pd.DataFrame(rows)


# ================================================================================
# STEP 4: SPILLOVER REGRESSION
# ================================================================================

def run_regression(panel: pd.DataFrame) -> None:
    """
    Estimates the improved intra-ETF spillover regression.

    The model absorbs receiver-stock and year-quarter fixed effects instead of
    materializing them as dummy variables. This is more stable for the large
    panel and allows one-way and two-way clustered standard errors.
    """
    main_terms = ["b1_term", "b2_term", "b4_term", "b5_term", "asym_term",
                  "Illiq_j", "Mispricing_k", "Similarity_ij", "Neg_e"]
    required = ["AR_j", *main_terms, "stock_j", "year_quarter", "event_id"]
    df = panel.dropna(subset=required).copy()

    if len(df) < 100:
        print(f"[WARN] Panel too small ({len(df)} obs). "
              f"Try lowering SHOCK_THRESHOLD.")
        return

    n_unique_events = df["event_id"].nunique()
    n_unique_stocks = df["stock_j"].nunique()
    n_unique_yq = df["year_quarter"].nunique()

    print(f"\n  Panel size : {len(df):,} observations")
    print(f"  Events     : {n_unique_events:,}")
    print(f"  Receivers  : {n_unique_stocks:,}")
    print(f"  Year-qtrs  : {n_unique_yq:,}")

    print("\nEstimating absorbed fixed-effects model...")
    result_oneway = _estimate_absorbed(
        df, main_terms, ["event_id"], "One-way clustered by event_id")

    print("\n" + "=" * 70)
    print("IMPROVED INTRA-ETF SPILLOVER REGRESSION RESULTS")
    print("(One-way clustered by event_id)")
    print("=" * 70)
    _print_results(result_oneway, main_terms)

    print("\nEstimating two-way clustered standard errors...")
    try:
        result_twoway = _estimate_absorbed(
            df, main_terms, ["event_id", "stock_j"],
            "Two-way clustered by event_id and stock_j")
        print("\n" + "=" * 70)
        print("TWO-WAY CLUSTERED STANDARD ERRORS (event_id x stock_j)")
        print("=" * 70)
        _print_results_twoway(result_twoway, main_terms)
    except Exception as e:
        print(f"  [WARN] Two-way clustering failed: {e}")
        print("  Falling back to one-way (event_id) clustering.")
        result_twoway = result_oneway

    txt_path = OUTPUT_DIR / "regression_improved.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("IMPROVED MODEL - One-way clustered by event_id\n")
        f.write("=" * 70 + "\n")
        f.write(_absorbed_summary_text(result_oneway, main_terms))
        if result_twoway is not result_oneway:
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("TWO-WAY CLUSTERED SE (event_id x stock_j)\n")
            f.write("=" * 70 + "\n")
            f.write(_twoway_summary_text(result_twoway, main_terms))
    print(f"\nFull summary saved: {txt_path}")

    _plot_coefs(result_oneway, main_terms[:5])
    asymmetry_analysis(df)

def _estimate_absorbed(df: pd.DataFrame, terms: list,
                       cluster_cols: list, cov_label: str) -> dict:
    """Fits the model with absorbed stock and year-quarter fixed effects."""
    y = df["AR_j"]
    x = df[terms]
    absorb = df[["stock_j", "year_quarter"]].astype("category")
    clusters = df[cluster_cols].copy()
    for col in cluster_cols:
        clusters[col] = pd.Categorical(clusters[col]).codes

    result = AbsorbingLS(y, x, absorb=absorb).fit(
        cov_type="clustered",
        clusters=clusters,
    )

    return {
        "params": result.params,
        "se": result.std_errors,
        "t": result.tstats,
        "p": result.pvalues,
        "nobs": result.nobs,
        "rsquared_adj": result.rsquared_adj,
        "cov_label": cov_label,
    }


def _print_results(result, terms):
    """Prints regression results for the main terms."""
    valid_terms = [t for t in terms if t in result["params"].index]
    tbl = pd.DataFrame({
        "Coef.": result["params"].loc[valid_terms],
        "Std.Err.": result["se"].loc[valid_terms],
        "t": result["t"].loc[valid_terms],
        "P>|t|": result["p"].loc[valid_terms],
    })
    print(tbl)
    print(f"\nN observations : {result['nobs']:.0f}")
    print(f"Adjusted R-sq  : {result['rsquared_adj']:.4f}")

    _print_interpretation(result["params"], result["p"], terms)


def _print_results_twoway(res_dict, terms):
    """Prints two-way clustered results."""
    valid_terms = [t for t in terms if t in res_dict["params"].index]
    print(f"{'Term':<20s} {'Coef':>10s} {'SE':>10s} {'t':>8s} {'p':>8s}")
    print("-" * 60)
    for t in valid_terms:
        stars = ("***" if res_dict["p"][t] < 0.01 else
                 "**"  if res_dict["p"][t] < 0.05 else
                 "*"   if res_dict["p"][t] < 0.10 else "  ")
        print(f"{t:<20s} {res_dict['params'][t]:>10.4f} "
              f"{res_dict['se'][t]:>10.4f} "
              f"{res_dict['t'][t]:>8.2f} "
              f"{res_dict['p'][t]:>8.4f} {stars}")
    print(f"\nN observations : {res_dict['nobs']:.0f}")
    print(f"Adjusted R-sq  : {res_dict['rsquared_adj']:.4f}")


def _absorbed_summary_text(res_dict, terms):
    """Returns text summary for absorbed fixed-effects results."""
    lines = [f"Covariance: {res_dict.get('cov_label', 'clustered')}"]
    lines.append(_twoway_summary_text(res_dict, terms))
    return "\n".join(lines)


def _twoway_summary_text(res_dict, terms):
    """Returns text summary of clustered results."""
    lines = []
    valid_terms = [t for t in terms if t in res_dict["params"].index]
    lines.append(f"{'Term':<20s} {'Coef':>10s} {'SE':>10s} "
                 f"{'t':>8s} {'p':>8s}")
    lines.append("-" * 60)
    for t in valid_terms:
        stars = ("***" if res_dict["p"][t] < 0.01 else
                 "**"  if res_dict["p"][t] < 0.05 else
                 "*"   if res_dict["p"][t] < 0.10 else "  ")
        lines.append(f"{t:<20s} {res_dict['params'][t]:>10.6f} "
                     f"{res_dict['se'][t]:>10.6f} "
                     f"{res_dict['t'][t]:>8.2f} "
                     f"{res_dict['p'][t]:>8.4f} {stars}")
    lines.append(f"\nN observations : {res_dict['nobs']:.0f}")
    lines.append(f"Adjusted R-sq  : {res_dict['rsquared_adj']:.4f}")
    return "\n".join(lines)


def _print_interpretation(coefs, pvals, terms):
    """Prints coefficient interpretation."""
    labels = {
        "b1_term":       "Beta1  Baseline propagation      (Shock x w_i)",
        "b2_term":       "Beta2  Liquidity channel          (Shock x w_i x Illiq_j)",
        "b4_term":       "Beta4  Arbitrage channel          (Shock x w_i x Mispricing)",
        "b5_term":       "Beta5  Informational spillover    (Shock x Similarity)",
        "asym_term":     "Delta   Negative asymmetry         (Neg x Shock x w_i)",
        "Illiq_j":       "Gamma1  Illiquidity main effect",
        "Mispricing_k":  "Gamma2  Mispricing main effect",
        "Similarity_ij": "Gamma3  Similarity main effect",
        "Neg_e":         "Gamma4  Negative shock main effect",
    }
    print("\n-- Interpretation --")
    for term in terms:
        if term not in coefs.index or term not in labels:
            continue
        stars = ("***" if pvals[term] < 0.01 else
                 "**"  if pvals[term] < 0.05 else
                 "*"   if pvals[term] < 0.10 else "  ")
        print(f"  {labels[term]}: {coefs[term]:+.6f}  "
              f"p={pvals[term]:.3f} {stars}")


def _plot_coefs(result, terms):
    """Saves a coefficient plot."""
    labels = {
        "b1_term":   "Beta1 Baseline",
        "b2_term":   "Beta2 Liquidity",
        "b4_term":   "Beta4 Arbitrage",
        "b5_term":   "Beta5 Informational",
        "asym_term": "Delta Asymmetry",
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    vals   = [result["params"][t] for t in terms]
    short  = [labels[t] for t in terms]
    colors = ["#E24B4A" if result["p"][t] < 0.05 else "#888780"
              for t in terms]
    ax.barh(short, vals, color=colors, height=0.5)
    ax.axvline(0, color="#2C2C2A", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coefficient estimate")
    ax.set_title("Improved spillover regression  (red = p<0.05)", fontsize=11)
    plt.tight_layout()
    out = FIG_DIR / "coef_plot_improved.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Coefficient plot saved: {out}")


def asymmetry_analysis(panel: pd.DataFrame) -> None:
    """Estimates the model separately for positive and negative shocks."""
    for sign, label in [(0, "POSITIVE"), (1, "NEGATIVE")]:
        sub = panel[panel["Neg_e"] == sign].dropna(
            subset=["AR_j", "b1_term", "b2_term", "b4_term", "b5_term",
                    "Illiq_j", "Mispricing_k", "Similarity_ij"])
        if len(sub) < 50:
            print(f"[WARN] Too few {label} shocks ({len(sub)}). Skipping.")
            continue
        formula = ("AR_j ~ b1_term + b2_term + b4_term + b5_term"
                   " + Illiq_j + Mispricing_k + Similarity_ij"
                   " + C(stock_j) + C(year_quarter)")
        try:
            res = smf.ols(formula, data=sub).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["event_id"]})
        except Exception as e:
            print(f"[WARN] {label} shock regression failed: {e}")
            continue
        print(f"\n{'=' * 60}")
        print(f"SUBSAMPLE: {label} SHOCKS  (N={res.nobs:.0f})")
        print(f"{'=' * 60}")
        terms = ["b1_term", "b2_term", "b4_term", "b5_term"]
        tbl = res.summary2().tables[1]
        available_cols = [c for c in ["Coef.", "Std.Err.", "t", "P>|t|",
                                       "z", "P>|z|"]
                          if c in tbl.columns]
        valid_terms = [t for t in terms if t in tbl.index]
        print(tbl.loc[valid_terms, available_cols])


# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":

    # Check for cached panel
    panel_path = OUTPUT_DIR / "panel_improved.csv"
    if panel_path.exists():
        print(f"\nLoading existing panel from {panel_path}...")
        full_panel = pd.read_csv(panel_path, parse_dates=["t0"])
        print(f"  Panel shape: {full_panel.shape}")
        run_regression(full_panel)
        import sys; sys.exit(0)

    panels_all = []
    all_shock_diagnostics = []

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

        etf_ret = (returns[etf].dropna()
                   if etf in returns.columns else ret_bench)

        # Constituent stocks with available returns
        etf_stocks   = hold_df["symbol"].unique().tolist()
        avail_stocks = [t for t in etf_stocks if t in returns.columns
                        and t != bench_sym]
        ret_stocks   = returns[avail_stocks]

        print(f"  Holdings symbols    : {len(etf_stocks):,}")
        print(f"  With return data    : {len(avail_stocks):,}")

        if len(avail_stocks) < 5:
            print(f"  [WARN] Too few stocks. Skipping {etf}.")
            continue

        # Identify shocks (event-specific threshold)
        print(f"\n  Identifying shocks "
              f"(threshold={SHOCK_THRESHOLD}sigma, event-specific)...")
        shocks = identify_shocks_vectorized(ret_stocks, ret_bench)
        print(f"  Shocks identified: {len(shocks):,}")

        if shocks.empty:
            print(f"  [WARN] No shocks found.")
            continue

        # Save shock diagnostics
        shocks["etf"] = etf
        all_shock_diagnostics.append(shocks)

        n_neg = shocks["is_negative"].sum()
        n_pos = len(shocks) - n_neg
        print(f"  Positive shocks: {n_pos:,}  |  Negative shocks: {n_neg:,}")

        # Build panel
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

        # Save panel
        full_panel.to_csv(panel_path, index=False)
        print(f"Panel saved: {panel_path}")

        # Save shock diagnostics
        if all_shock_diagnostics:
            diag = pd.concat(all_shock_diagnostics, ignore_index=True)
            diag_path = OUTPUT_DIR / "shock_diagnostics.csv"
            diag.to_csv(diag_path, index=False)
            print(f"Shock diagnostics saved: {diag_path}")
            print(f"  Mean sigma_eps   : {diag['sigma_eps'].mean():.6f}")
            print(f"  Mean threshold   : {diag['car_threshold'].mean():.6f}")
            print(f"  Mean |CAR|       : {diag['car_i'].abs().mean():.6f}")

        # Run regression
        run_regression(full_panel)












