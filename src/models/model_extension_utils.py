"""Shared helpers for model extensions"""

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS
from scipy.stats import norm

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
MIN_EST_OBS = 108
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

WEIGHTED_MODERATORS = {
    "b2_term": "Illiq_j",
    "b4_term": "Mispricing_k",
    "receiver_weight_term": "w_j",
    "corr_term": "Corr_ij_60d",
    "hhi_term": "HHI_etf_t",
    "asym_term": "Neg_e",
}
CONTINUOUS_MODERATORS = ["Illiq_j", "Mispricing_k", "w_j", "Corr_ij_60d", "HHI_etf_t"]
NON_OVERLAP_GAP_DAYS = 5

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

def holdings_paths():
    return [PROC_DIR / f"{etf.lower()}_holdings.csv" for etf in ETF_LIST]

def is_fresh(path, inputs):
    """True when path exists and is newer than every input file"""
    if not path.exists():
        return False
    mtime = path.stat().st_mtime
    return all(not p.exists() or p.stat().st_mtime <= mtime for p in inputs)

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
    pivot = pivot.sort_index()
    previous_day = pd.DatetimeIndex(return_index) - pd.Timedelta(days=1)
    pivot = pivot.reindex(previous_day, method="ffill").fillna(0.0)
    pivot.index = return_index
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

def window_abnormal_returns(y_full, x_full, pos, horizon=EVENT_H):
    """Model fitted on the estimation window and abnormal returns over the event window, skipping missing returns"""
    valid = np.isfinite(y_full) & np.isfinite(x_full).all(axis=1)
    y0 = np.where(valid, y_full, 0.0)
    x0 = np.where(valid[:, None], x_full, 0.0)
    starts = pos - ESTIMATION_WINDOW - GAP
    win_idx = starts[:, None] + np.arange(ESTIMATION_WINDOW)[None, :]
    x_batch = x0[win_idx]
    y_batch = y0[win_idx]
    n_obs = valid[win_idx].sum(axis=1)
    p = x_full.shape[1]
    coefs = np.full((len(pos), p), np.nan)
    enough = n_obs >= MIN_EST_OBS
    if enough.any():
        xtx = np.einsum("nwp,nwq->npq", x_batch[enough], x_batch[enough])
        xty = np.einsum("nwp,nw->np", x_batch[enough], y_batch[enough])
        try:
            coefs[enough] = np.linalg.solve(xtx, xty[..., None])[..., 0]
        except np.linalg.LinAlgError:
            for k in np.flatnonzero(enough):
                coefs[k] = np.linalg.lstsq(x_batch[k], y_batch[k], rcond=None)[0]
    resid = np.where(valid[win_idx], y_batch - np.einsum("nwp,np->nw", x_batch, coefs), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.sqrt((resid ** 2).sum(axis=1) / (n_obs - p))
    sigma[~enough] = np.nan
    ev_idx = pos[:, None] + np.arange(horizon + 1)[None, :]
    ar = y_full[ev_idx] - np.einsum("nwp,np->nw", x_full[ev_idx], coefs)
    return coefs, sigma, ar

def event_positions(index, event_dates, horizon):
    event_dates = pd.to_datetime(pd.Series(event_dates).dropna().unique())
    pos = index.get_indexer(event_dates)
    valid = (pos >= ESTIMATION_WINDOW + GAP) & (pos + horizon < len(index))
    return event_dates[valid], pos[valid]

def batch_market_car_for_dates(ret_stock, ret_bench, event_dates, horizon=EVENT_H):
    idx = ret_stock.index.intersection(ret_bench.index).sort_values()
    dates, pos = event_positions(idx, event_dates, horizon)
    if len(pos) == 0:
        return pd.DataFrame(columns=["t0", "car", "sigma_eps", "threshold"])
    y_full = ret_stock.reindex(idx).astype(float).to_numpy()
    x_full = np.column_stack([np.ones(len(idx)), ret_bench.reindex(idx).astype(float).to_numpy()])
    _, sigma, ar = window_abnormal_returns(y_full, x_full, pos, horizon)
    car = ar.sum(axis=1)
    threshold = SHOCK_THRESHOLD * np.sqrt(horizon + 1) * sigma
    return pd.DataFrame({"t0": dates, "car": car, "sigma_eps": sigma, "threshold": threshold})

def batch_factor_car_for_dates(ret_stock, factors, event_dates, horizon=EVENT_H):
    idx = ret_stock.index.intersection(factors.index).sort_values()
    dates, pos = event_positions(idx, event_dates, horizon)
    if len(pos) == 0:
        return pd.DataFrame(columns=["t0", "car", "sigma_eps", "threshold"])
    y_full = (ret_stock.reindex(idx).astype(float) - factors.reindex(idx)["RF"].astype(float)).to_numpy()
    x_vals = factors.reindex(idx).drop(columns=["RF"]).astype(float).to_numpy()
    x_full = np.column_stack([np.ones(len(idx)), x_vals])
    _, sigma, ar = window_abnormal_returns(y_full, x_full, pos, horizon)
    car = ar.sum(axis=1)
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

def _linear_combination(res, weights):
    terms = [t for t in weights.index if t in res.params.index]
    g = weights.loc[terms].astype(float)
    b = res.params.loc[terms]
    cov = res.cov.loc[terms, terms]
    est = float(g @ b)
    se = float(np.sqrt(g @ cov @ g))
    t_stat = est / se if se > 0 else np.nan
    pval = 2 * norm.sf(abs(t_stat)) if np.isfinite(t_stat) else np.nan
    return est, se, t_stat, pval

def shock_effects(res, sample):
    """Shock effect at the moderator means and average marginal effect of Shock_i"""
    terms = [t for t in MAIN_TERMS if t in res.params.index]
    at_means = pd.Series(0.0, index=terms)
    ame = pd.Series(0.0, index=terms)
    for term in terms:
        if term == "b1_term":
            at_means[term] = 1.0
            ame[term] = sample["w_i"].mean()
        elif term == "b5_term":
            ame[term] = sample["Similarity_ij"].mean()
        else:
            moderator = sample[WEIGHTED_MODERATORS[term]].astype(float)
            at_means[term] = moderator.mean()
            ame[term] = (sample["w_i"] * moderator).mean()
    out = {}
    for name, weights in [("direct_at_means", at_means), ("ame_shock", ame)]:
        est, se, t_stat, pval = _linear_combination(res, weights)
        out[name] = est
        out[f"{name}_se"] = se
        out[f"{name}_t"] = t_stat
        out[f"{name}_pval"] = pval
    return out

def centre_interaction_terms(df, moderators=None):
    """Rebuild the weighted interactions with centred moderators"""
    if moderators is None:
        moderators = CONTINUOUS_MODERATORS
    out = df.copy()
    base = out["Shock_i"] * out["w_i"]
    for term, moderator in WEIGHTED_MODERATORS.items():
        if moderator in moderators:
            centred = out[moderator] - out[moderator].mean()
            out[term] = base * centred
    return out

def first_day_of_episode_events(panel, gap_days=NON_OVERLAP_GAP_DAYS):
    """Keep the first event of each run of consecutive shocks in the same stock"""
    events = (panel[["etf", "event_id", "stock_i", "t0"]]
              .drop_duplicates(["etf", "event_id"])
              .sort_values(["etf", "stock_i", "t0"]))
    gap = events.groupby(["etf", "stock_i"])["t0"].diff().dt.days
    keep = events.loc[~(gap <= gap_days), ["etf", "event_id"]]
    return keep.reset_index(drop=True)

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
