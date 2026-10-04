"""Trading simulation of intra-ETF transmission with real-time signals and transaction costs"""

import warnings
warnings.filterwarnings("ignore")

from bisect import bisect_left

import numpy as np
import pandas as pd
import statsmodels.api as sm

from model_extension_utils import (
    CORR_MIN_OBS,
    CORR_WINDOW,
    ESTIMATION_WINDOW,
    ETF_LIST,
    EVENT_H,
    GAP,
    MAIN_TERMS,
    NON_OVERLAP_GAP_DAYS,
    OUTPUT_DIR,
    SHOCK_THRESHOLD,
    WEIGHTED_MODERATORS,
    _linear_combination,
    fit_absorbed,
    load_benchmark_returns,
    load_holdings,
    load_panel,
    load_returns,
    select_benchmark,
    window_abnormal_returns,
)

TRAIN_END = pd.Timestamp("2021-12-31")
VALID_END = pd.Timestamp("2023-12-31")
THRESHOLD_SIGMAS = SHOCK_THRESHOLD * np.sqrt(EVENT_H + 1)
SIGNALS = {"early": {"delay": 0, "hold": 3}, "post": {"delay": EVENT_H, "hold": 5}}
ENTRY_LAGS = [0, 1]
COSTS_BPS = [5, 10, 20]
CONNECTED_QUANTILE = 0.75
NW_LAGS = 5
TRADING_DAYS = 252

SUMMARY_PATH = OUTPUT_DIR / "trading_simulation_summary.csv"
DAILY_PATH = OUTPUT_DIR / "trading_simulation_daily.csv"
DETAILS_PATH = OUTPUT_DIR / "trading_simulation_details.txt"


def period_of(dates):
    dates = pd.to_datetime(dates)
    return np.where(dates <= TRAIN_END, "train", np.where(dates <= VALID_END, "validation", "test"))


def snapshot_lookup(holdings):
    holdings = holdings.dropna(subset=["symbol", "weight"])
    holdings = holdings[holdings["weight"].astype(float) > 0]
    snapshots = {}
    for date, group in holdings.groupby("atDate"):
        snapshots[pd.Timestamp(date)] = group.drop_duplicates("symbol").set_index("symbol")["weight"].astype(float)
    return sorted(snapshots), snapshots


def first_flags(flags):
    """Keep a flag only when the same stock had no flag in the previous days, using past information only"""
    flags = flags.sort_values(["stock_i", "t0"]).copy()
    gap = flags.groupby("stock_i")["t0"].diff().dt.days
    return flags[~(gap <= NON_OVERLAP_GAP_DAYS)]


def detect_flags(returns, bench, tickers):
    idx = returns.index
    x_full = np.column_stack([np.ones(len(idx)), bench.reindex(idx).to_numpy(float)])
    pos = np.arange(ESTIMATION_WINDOW + GAP, len(idx) - EVENT_H - 1)
    betas = pd.DataFrame(np.nan, index=idx, columns=tickers)
    records = {name: [] for name in SIGNALS}
    for ticker in tickers:
        coefs, sigma, ar = window_abnormal_returns(returns[ticker].to_numpy(float), x_full, pos, EVENT_H)
        betas.iloc[pos, betas.columns.get_loc(ticker)] = coefs[:, 1]
        threshold = THRESHOLD_SIGMAS * sigma
        values = {"early": ar[:, 0], "post": ar.sum(axis=1)}
        for name, value in values.items():
            with np.errstate(invalid="ignore"):
                flagged = np.isfinite(value) & np.isfinite(threshold) & (threshold >= 1e-10) & (np.abs(value) > threshold)
            for k in np.flatnonzero(flagged):
                records[name].append({
                    "stock_i": ticker,
                    "t0": idx[pos[k]],
                    "t0_pos": pos[k],
                    "signal": value[k],
                })
    flags = {name: first_flags(pd.DataFrame(rows)) for name, rows in records.items()}
    return flags, betas


def window_correlations(window, origin, receivers):
    y = window[origin].to_numpy(float)
    x = window[receivers].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)[:, None]
    n = mask.sum(axis=0)
    xx = np.where(mask, x, 0.0)
    yy = np.where(mask, y[:, None], 0.0)
    dx = np.where(mask, xx - xx.sum(axis=0) / np.maximum(n, 1), 0.0)
    dy = np.where(mask, yy - yy.sum(axis=0) / np.maximum(n, 1), 0.0)
    scale = np.sqrt((dx ** 2).sum(axis=0) * (dy ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (dx * dy).sum(axis=0) / scale
    return np.where((n >= CORR_MIN_OBS) & (scale > 1e-12), corr, np.nan)


def build_positions(etf, name, flags, returns, betas, hold_dates, snapshots):
    rows = []
    for flag in flags.itertuples(index=False):
        k = bisect_left(hold_dates, flag.t0) - 1
        if k < 0:
            continue
        weights = snapshots[hold_dates[k]]
        if flag.stock_i not in weights.index:
            continue
        receivers = [s for s in weights.index if s != flag.stock_i and s in betas.columns]
        if not receivers:
            continue
        window = returns.iloc[flag.t0_pos - CORR_WINDOW:flag.t0_pos]
        corr = window_correlations(window, flag.stock_i, receivers)
        beta = betas.iloc[flag.t0_pos][receivers].to_numpy(float)
        for receiver, c, b in zip(receivers, corr, beta):
            if np.isfinite(c) and np.isfinite(b):
                rows.append((etf, name, flag.stock_i, receiver, flag.t0, flag.t0_pos + SIGNALS[name]["delay"],
                             np.sign(flag.signal), c, b))
    return pd.DataFrame(rows, columns=["etf", "signal", "stock_i", "stock_j", "t0", "decision_pos",
                                       "direction", "corr", "beta"])


def hedged_paths(positions, simple_returns, bench_simple, lag, hold):
    n_days = len(simple_returns)
    start = positions["decision_pos"].to_numpy() + lag + 1
    keep = start + hold - 1 < n_days
    positions = positions[keep].reset_index(drop=True)
    start = start[keep]
    day_idx = start[:, None] + np.arange(hold)[None, :]
    cols = simple_returns.columns.get_indexer(positions["stock_j"])
    r_j = simple_returns.to_numpy()[day_idx, cols[:, None]]
    bench_cols = bench_simple.columns.get_indexer(positions["etf"])
    r_m = bench_simple.to_numpy()[day_idx, bench_cols[:, None]]
    hedged = positions["direction"].to_numpy()[:, None] * (r_j - positions["beta"].to_numpy()[:, None] * r_m)
    return positions, day_idx, np.nan_to_num(hedged, nan=0.0)


def daily_portfolio(day_idx, hedged, cost, n_days):
    adjusted = hedged.copy()
    adjusted[:, 0] -= cost
    adjusted[:, -1] -= cost
    counts = np.bincount(day_idx.ravel(), minlength=n_days)
    sums = np.bincount(day_idx.ravel(), weights=adjusted.ravel(), minlength=n_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        daily = np.where(counts > 0, sums / counts, 0.0)
    return daily, counts


def newey_west_t(series):
    if len(series) < 10 or np.std(series) == 0:
        return np.nan
    fit = sm.OLS(series, np.ones(len(series))).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return float(fit.tvalues[0])


def summarise(label, positions, day_idx, hedged, dates):
    rows = []
    n_days = len(dates)
    day_period = period_of(dates)
    trade_period = period_of(dates[positions["decision_pos"].to_numpy()])
    trade_return = hedged.sum(axis=1)
    gross_daily, counts = daily_portfolio(day_idx, hedged, 0.0, n_days)
    net_daily = {c: daily_portfolio(day_idx, hedged, c / 1e4, n_days)[0] for c in COSTS_BPS}
    for period in ["train", "validation", "test", "all"]:
        in_days = np.ones(n_days, bool) if period == "all" else day_period == period
        in_trades = np.ones(len(positions), bool) if period == "all" else trade_period == period
        first = np.flatnonzero(counts > 0)
        if len(first):
            in_days &= np.arange(n_days) >= first[0]
        g = gross_daily[in_days]
        row = {
            **label,
            "period": period,
            "events": positions.loc[in_trades, ["stock_i", "t0"]].drop_duplicates().shape[0],
            "trades": int(in_trades.sum()),
            "days": int(in_days.sum()),
            "share_days_invested": float((counts[in_days] > 0).mean()) if in_days.any() else np.nan,
            "mean_trade_bps": float(trade_return[in_trades].mean() * 1e4) if in_trades.any() else np.nan,
            "hit_rate": float((trade_return[in_trades] > 0).mean()) if in_trades.any() else np.nan,
            "breakeven_cost_bps": float(trade_return[in_trades].mean() * 1e4 / 2) if in_trades.any() else np.nan,
            "ann_return_gross": float(g.mean() * TRADING_DAYS) if len(g) else np.nan,
            "ann_vol": float(g.std() * np.sqrt(TRADING_DAYS)) if len(g) else np.nan,
            "sharpe_gross": float(g.mean() / g.std() * np.sqrt(TRADING_DAYS)) if len(g) and g.std() > 0 else np.nan,
            "nw_t_gross": newey_west_t(g),
        }
        for c in COSTS_BPS:
            n = net_daily[c][in_days]
            row[f"ann_return_net_{c}"] = float(n.mean() * TRADING_DAYS) if len(n) else np.nan
            row[f"sharpe_net_{c}"] = float(n.mean() / n.std() * np.sqrt(TRADING_DAYS)) if len(n) and n.std() > 0 else np.nan
            row[f"nw_t_net_{c}"] = newey_west_t(n)
        rows.append(row)
    return rows, gross_daily, net_daily[10]


def training_slope_at(panel, corr_levels):
    """Weighted slope of the main model estimated on the training period, at given correlation levels"""
    train = panel[panel["t0"] <= TRAIN_END]
    res, used = fit_absorbed(train)
    terms = [t for t in MAIN_TERMS if t in res.params.index and t != "b5_term"]
    out = {}
    for name, level in corr_levels.items():
        weights = pd.Series(0.0, index=terms)
        for term in terms:
            weights[term] = 1.0 if term == "b1_term" else used[WEIGHTED_MODERATORS[term]].astype(float).mean()
        weights["corr_term"] = level
        est, se, t_stat, _ = _linear_combination(res, weights)
        out[name] = (level, est, se, t_stat)
    return out


def write_details(summary, thresholds, slopes, flag_counts):
    lines = [
        "TRADING SIMULATION OF INTRA-ETF TRANSMISSION",
        "=" * 80,
        f"Signal threshold: |abnormal return| > {THRESHOLD_SIGMAS:.1f} x pre-event residual volatility",
        "early: day-0 abnormal return, decision at the close of t0, holding 3 days",
        "post: four-day CAR, decision at the close of t0+3, holding 5 days",
        "Entry lag 0 enters at the decision close, lag 1 at the next close",
        "Positions are hedged with beta times the ETF benchmark and equally weighted each day",
        f"Training ends {TRAIN_END.date()}, validation ends {VALID_END.date()}, test after",
        "",
    ]
    for name, count in flag_counts.items():
        lines.append(f"First-day flags, {name}: {count:,}")
    for name, level in thresholds.items():
        lv, est, se, t_stat = slopes[name]
        lines.append(f"Connected threshold, {name} (training P75 of correlation): {level:.3f}  "
                     f"training weighted slope at this level: {est:+.4f} (t={t_stat:.2f})")
    lines += ["", "TEST PERIOD", "-" * 80]
    test = summary[summary["period"] == "test"]
    for _, row in test.iterrows():
        lines.append(
            f"{row['signal']:<6s} lag={row['entry_lag']} {row['universe']:<9s} trades={row['trades']:>7,d}  "
            f"mean={row['mean_trade_bps']:+6.2f}bp  hit={row['hit_rate']:.3f}  "
            f"SR gross={row['sharpe_gross']:+.2f} (t={row['nw_t_gross']:.2f})  "
            f"SR net10={row['sharpe_net_10']:+.2f}  breakeven={row['breakeven_cost_bps']:+.2f}bp"
        )
    DETAILS_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    returns = load_returns()
    bench_returns = load_benchmark_returns()
    simple_returns = np.expm1(returns)
    dates = returns.index
    bench_simple = pd.DataFrame(index=dates)
    positions = []
    flag_counts = {name: 0 for name in SIGNALS}
    for etf in ETF_LIST:
        _, bench = select_benchmark(etf, returns, bench_returns)
        bench_simple[etf] = np.expm1(bench.reindex(dates).to_numpy(float))
        holdings = load_holdings(etf)
        hold_dates, snapshots = snapshot_lookup(holdings)
        tickers = sorted(set(holdings["symbol"].dropna()) & set(returns.columns))
        flags, betas = detect_flags(returns, bench, tickers)
        for name, etf_flags in flags.items():
            flag_counts[name] += len(etf_flags)
            positions.append(build_positions(etf, name, etf_flags, returns, betas, hold_dates, snapshots))
    positions = pd.concat(positions, ignore_index=True)
    train_pos = positions[positions["t0"] <= TRAIN_END]
    thresholds = {name: float(train_pos.loc[train_pos["signal"] == name, "corr"].quantile(CONNECTED_QUANTILE))
                  for name in SIGNALS}
    slopes = training_slope_at(load_panel(), thresholds)
    summary_rows = []
    daily = {}
    for name, spec in SIGNALS.items():
        base = positions[positions["signal"] == name]
        for universe in ["all", "connected"]:
            subset = base if universe == "all" else base[base["corr"] >= thresholds[name]]
            for lag in ENTRY_LAGS:
                kept, day_idx, hedged = hedged_paths(subset, simple_returns, bench_simple, lag, spec["hold"])
                label = {"signal": name, "universe": universe, "entry_lag": lag, "hold_days": spec["hold"]}
                rows, gross, net10 = summarise(label, kept, day_idx, hedged, dates)
                summary_rows += rows
                daily[f"{name}_{universe}_lag{lag}_gross"] = gross
                daily[f"{name}_{universe}_lag{lag}_net10"] = net10
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(daily, index=dates).to_csv(DAILY_PATH, index_label="date")
    write_details(summary, thresholds, slopes, flag_counts)


if __name__ == "__main__":
    main()
