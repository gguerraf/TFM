"""Shared helpers for model extensions"""

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS

BASE_DIR = Path(__file__).resolve().parents[2] / "holdings"
PROC_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

ETF_LIST = ["XME", "XLE", "IHE", "XLV"]
ETF_BENCHMARK = {
    "XME": "XLB",
    "XLE": "IXC",
    "IHE": "XLV",
    "XLV": "^SP500-35",
}
ETF_BENCHMARK_CANDIDATES = {
    "XME": ["XLB", "^SP500-15", "^GSPC"],
    "XLE": ["IXC", "^GSPC"],
    "IHE": ["XLV", "^SP500-35", "^GSPC"],
    "XLV": ["^SP500-35", "VHT", "^GSPC"],
}

ESTIMATION_WINDOW = 120
GAP = 5
EVENT_H = 3
SHOCK_THRESHOLD = 1.5
CORR_WINDOW = 60
CORR_MIN_OBS = 30

MAIN_TERMS = [
    "b1_term", "b2_term", "b4_term", "b5_term",
    "receiver_weight_term", "corr_term", "hhi_term", "asym_term",
]
CONTROL_TERMS = [
    "Illiq_j", "Mispricing_k", "Similarity_ij", "w_j",
    "Corr_ij_60d", "HHI_etf_t", "Neg_e",
]
MODEL_TERMS = MAIN_TERMS + CONTROL_TERMS

def load_returns():
    peek = pd.read_csv(PROC_DIR / "returns_clean.csv", nrows=0)
    idx_col = peek.columns[0]
    returns = pd.read_csv(PROC_DIR / "returns_clean.csv", index_col=idx_col, parse_dates=True)
    returns.index = pd.to_datetime(returns.index)
    return returns.sort_index()

def load_benchmark_returns():
    peek = pd.read_csv(PROC_DIR / "benchmarks.csv", nrows=0)
    idx_col = peek.columns[0]
    benchmarks = pd.read_csv(PROC_DIR / "benchmarks.csv", index_col=idx_col, parse_dates=True)
    benchmarks.index = pd.to_datetime(benchmarks.index)
    return np.log(benchmarks / benchmarks.shift(1)).iloc[1:].sort_index()

def load_panel():
    path = OUTPUT_DIR / "panel_improved.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run 21_estimate_improved_model.py first.")
    return pd.read_csv(path, parse_dates=["t0"])

def load_holdings(etf):
    return pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv", parse_dates=["atDate"])

def select_benchmark(etf, returns, bench_returns):
    ticker = ETF_BENCHMARK.get(etf)
    if ticker in bench_returns.columns:
        return ticker, bench_returns[ticker].dropna()
    if ticker in returns.columns:
        return ticker, returns[ticker].dropna()
    for candidate in ETF_BENCHMARK_CANDIDATES.get(etf, ["^GSPC"]):
        if candidate in bench_returns.columns:
            return candidate, bench_returns[candidate].dropna()
        if candidate in returns.columns:
            return candidate, returns[candidate].dropna()
    raise ValueError(f"No benchmark found for {etf}")

def make_weight_panel(holdings, return_index, return_columns):
    weights = holdings[["atDate", "symbol", "weight"]].dropna().copy()
    weights["atDate"] = pd.to_datetime(weights["atDate"])
    weights = weights[weights["symbol"].isin(return_columns)]
    pivot = weights.pivot_table(index="atDate", columns="symbol", values="weight", aggfunc="sum")
    pivot = pivot.sort_index().reindex(return_index, method="ffill").fillna(0.0)
    cols = [c for c in pivot.columns if c in return_columns]
    return pivot[cols].astype(float)

def synthetic_basket_return(weights, returns, exclude):
    exclude = set(exclude)
    cols = [c for c in weights.columns if c in returns.columns and c not in exclude]
    if not cols:
        return pd.Series(np.nan, index=weights.index)
    w = weights[cols].copy()
    r = returns.reindex(weights.index)[cols]
    valid = r.notna()
    denom = w.where(valid, 0.0).sum(axis=1)
    numer = (w.where(valid, 0.0) * r.fillna(0.0)).sum(axis=1)
    out = numer / denom.replace(0.0, np.nan)
    out.name = "synthetic_benchmark"
    return out.replace([np.inf, -np.inf], np.nan)

def batch_market_car_for_dates(ret_stock, ret_bench, event_dates, horizon=EVENT_H):
    event_dates = pd.to_datetime(pd.Series(event_dates).dropna().unique())
    idx = ret_stock.index.intersection(ret_bench.index).sort_values()
    y_full = ret_stock.reindex(idx).astype(float).fillna(0.0).to_numpy()
    b_full = ret_bench.reindex(idx).astype(float).fillna(0.0).to_numpy()
    pos = idx.get_indexer(event_dates)
    valid = (pos >= ESTIMATION_WINDOW + GAP) & (pos >= 0) & (pos + horizon < len(idx))
    dates = event_dates[valid]
    pos = pos[valid]
    if len(pos) == 0:
        return pd.DataFrame(columns=["t0", "car", "sigma_eps", "threshold"])
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
    fitted = coefs[:, 0, None] + coefs[:, 1, None] * x_batch[:, :, 1]
    resid = y_batch - fitted
    sigma = np.nanstd(resid, axis=1, ddof=2)
    ev_idx = pos[:, None] + np.arange(horizon + 1)[None, :]
    ar = y_full[ev_idx] - coefs[:, 0, None] - coefs[:, 1, None] * b_full[ev_idx]
    car = np.nansum(ar, axis=1)
    threshold = SHOCK_THRESHOLD * np.sqrt(horizon + 1) * sigma
    return pd.DataFrame({"t0": dates, "car": car, "sigma_eps": sigma, "threshold": threshold})

def batch_factor_car_for_dates(ret_stock, factors, event_dates, horizon=EVENT_H):
    event_dates = pd.to_datetime(pd.Series(event_dates).dropna().unique())
    idx = ret_stock.index.intersection(factors.index).sort_values()
    y = ret_stock.reindex(idx).astype(float) - factors.reindex(idx)["RF"].astype(float)
    y_full = y.fillna(0.0).to_numpy()
    x_vals = factors.reindex(idx).drop(columns=["RF"]).astype(float).fillna(0.0).to_numpy()
    x_full = np.column_stack([np.ones(len(idx)), x_vals])
    pos = idx.get_indexer(event_dates)
    valid = (pos >= ESTIMATION_WINDOW + GAP) & (pos >= 0) & (pos + horizon < len(idx))
    dates = event_dates[valid]
    pos = pos[valid]
    if len(pos) == 0:
        return pd.DataFrame(columns=["t0", "car", "sigma_eps", "threshold"])
    starts = pos - ESTIMATION_WINDOW - GAP
    win_idx = starts[:, None] + np.arange(ESTIMATION_WINDOW)[None, :]
    x_batch = x_full[win_idx]
    y_batch = y_full[win_idx]
    xtx = np.einsum("nwp,nwq->npq", x_batch, x_batch)
    xty = np.einsum("nwp,nw->np", x_batch, y_batch)
    p = x_full.shape[1]
    coefs = np.full((len(pos), p), np.nan)
    try:
        coefs = np.linalg.solve(xtx, xty[..., None]).squeeze(-1)
    except Exception:
        for k in range(len(pos)):
            try:
                coefs[k], *_ = np.linalg.lstsq(x_batch[k], y_batch[k], rcond=None)
            except Exception:
                pass
    fitted = np.einsum("nwp,np->nw", x_batch, coefs)
    resid = y_batch - fitted
    sigma = np.nanstd(resid, axis=1, ddof=p)
    ev_idx = pos[:, None] + np.arange(horizon + 1)[None, :]
    y_event = y_full[ev_idx]
    x_event = x_full[ev_idx]
    ar = y_event - np.einsum("nwp,np->nw", x_event, coefs)
    car = np.nansum(ar, axis=1)
    threshold = SHOCK_THRESHOLD * np.sqrt(horizon + 1) * sigma
    return pd.DataFrame({"t0": dates, "car": car, "sigma_eps": sigma, "threshold": threshold})

def recompute_interaction_terms(df, shock_col="Shock_i"):
    out = df.copy()
    out["Shock_i"] = out[shock_col]
    out["Neg_e"] = (out["Shock_i"] < 0).astype(int)
    out["b1_term"] = out["Shock_i"] * out["w_i"]
    out["b2_term"] = out["Shock_i"] * out["w_i"] * out["Illiq_j"]
    out["b4_term"] = out["Shock_i"] * out["w_i"] * out["Mispricing_k"]
    out["b5_term"] = out["Shock_i"] * out["Similarity_ij"]
    out["receiver_weight_term"] = out["Shock_i"] * out["w_i"] * out["w_j"]
    out["corr_term"] = out["Shock_i"] * out["w_i"] * out["Corr_ij_60d"]
    out["hhi_term"] = out["Shock_i"] * out["w_i"] * out["HHI_etf_t"]
    out["asym_term"] = out["Neg_e"] * out["Shock_i"] * out["w_i"]
    return out

def fit_absorbed(df, terms=None, y_col="AR_j", absorb_cols=None, cluster_cols=None):
    if terms is None:
        terms = MODEL_TERMS
    if absorb_cols is None:
        absorb_cols = ["stock_j", "year_quarter"]
    if cluster_cols is None:
        cluster_cols = ["event_id", "stock_j"]
    required = [y_col, *terms, *absorb_cols, *cluster_cols]
    sub = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    if len(sub) < 100:
        raise ValueError(f"Too few observations after dropping missing values: {len(sub)}")
    y = sub[y_col].astype(float)
    x = sub[terms].astype(float)
    absorb = sub[absorb_cols].astype("category")
    clusters = pd.DataFrame(index=sub.index)
    for col in cluster_cols:
        clusters[col] = pd.Categorical(sub[col]).codes
    res = AbsorbingLS(y, x, absorb=absorb).fit(cov_type="clustered", clusters=clusters)
    return res, sub

def result_row(label, res, terms=MAIN_TERMS):
    row = {"specification": label, "N": res.nobs, "adj_R2": res.rsquared_adj}
    for term in terms:
        row[f"{term}_coef"] = res.params.get(term, np.nan)
        row[f"{term}_se"] = res.std_errors.get(term, np.nan)
        row[f"{term}_pval"] = res.pvalues.get(term, np.nan)
    return row

def format_result_block(label, res, terms=MAIN_TERMS):
    lines = ["=" * 80, label, "=" * 80]
    lines.append(f"N observations: {res.nobs:.0f}")
    lines.append(f"Adjusted R2: {res.rsquared_adj:.6f}")
    lines.append(f"{'term':<24s} {'coef':>12s} {'se':>12s} {'t':>10s} {'p':>10s}")
    lines.append("-" * 80)
    for term in terms:
        if term not in res.params.index:
            continue
        lines.append(
            f"{term:<24s} {res.params[term]:>12.6f} "
            f"{res.std_errors[term]:>12.6f} {res.tstats[term]:>10.3f} "
            f"{res.pvalues[term]:>10.4f}"
        )
    return "\n".join(lines)

def parse_ken_french_csv(text):
    lines = text.replace("\r", "").split("\n")
    header = None
    rows = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if "Mkt-RF" in parts or "Mom" in parts:
            header = ["date"] + [p for p in parts[1:] if p]
            continue
        if re.match(r"^\d{8},", line):
            rows.append(line)
        elif rows:
            break
    if header is None or not rows:
        raise ValueError("Could not parse Ken French factor file")
    df = pd.read_csv(io.StringIO("\n".join(rows)), names=header)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    return df

def download_or_load_factors():
    out_path = PROC_DIR / "fama_french_carhart_daily.csv"
    if out_path.exists():
        factors = pd.read_csv(out_path, parse_dates=["date"]).set_index("date").sort_index()
        return factors
    urls = {
        "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    }
    parsed = {}
    for key, url in urls.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode("utf-8", errors="replace")
        parsed[key] = parse_ken_french_csv(text)
    factors = parsed["ff5"].join(parsed["mom"][["Mom"]], how="left")
    factors = factors.rename(columns={"Mkt-RF": "MKT_RF"})
    factors.to_csv(out_path, index_label="date")
    return factors
