"""
3_compute_returns.py
====================
Computes daily log returns from the raw price matrix and produces
a clean, aligned returns file ready for the spillover model.

Inputs:
    processed/prices_raw.csv

Outputs:
    processed/returns_clean.csv   Log returns, NaN-trimmed, date-aligned
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR = BASE_DIR / "processed"

# Minimum fraction of non-NaN observations required to keep a ticker
# Tickers with fewer than this fraction of valid days are dropped
MIN_COVERAGE = 0.30   # at least 30% of trading days must have a price

# ─── LOAD PRICES ──────────────────────────────────────────────────────────────
print("Loading prices_raw.csv...")
prices = pd.read_csv(PROC_DIR / "prices_raw.csv",
                     index_col="date", parse_dates=True)
print(f"  Shape: {prices.shape[0]} dates x {prices.shape[1]} tickers")

# ─── COMPUTE LOG RETURNS ──────────────────────────────────────────────────────
print("Computing log returns...")
returns = np.log(prices / prices.shift(1))

# Drop the first row (all NaN by construction)
returns = returns.iloc[1:]

# ─── COVERAGE FILTER ─────────────────────────────────────────────────────────
print(f"Filtering tickers with coverage < {MIN_COVERAGE*100:.0f}%...")
coverage = returns.notna().mean()
keep_mask = coverage >= MIN_COVERAGE
returns = returns.loc[:, keep_mask]

n_dropped = (~keep_mask).sum()
print(f"  Tickers kept   : {returns.shape[1]:,}")
print(f"  Tickers dropped: {n_dropped:,}  (insufficient history)")

# ─── FORWARD-FILL MINOR GAPS (1-2 day stale prices) ──────────────────────────
# Fill very short gaps (e.g. staggered exchange holidays) but cap at 2 days
# to avoid carrying stale prices across long suspensions
prices_filled = prices.ffill(limit=2)
returns_filled = np.log(prices_filled / prices_filled.shift(1)).iloc[1:]
returns_filled = returns_filled.loc[:, returns.columns]

# Use filled returns only where the original had NaN but filled version has data
# This is a conservative fill: we don't override existing observed returns
returns = returns.combine_first(returns_filled)

# ─── ALIGN INDEX ──────────────────────────────────────────────────────────────
# Keep only dates that are actual US trading days (days where S&P 500 traded)
# Use the SPY or ^GSPC column as the trading calendar anchor
calendar_anchor = "SPY" if "SPY" in returns.columns else "^GSPC"
if calendar_anchor in returns.columns:
    trading_days = returns[calendar_anchor].dropna().index
    returns = returns.reindex(trading_days)
    print(f"  Aligned to {len(trading_days):,} US trading days "
          f"(anchor: {calendar_anchor})")
else:
    print("  [WARN] No calendar anchor found. Keeping all dates.")

# ─── FINAL STATS ──────────────────────────────────────────────────────────────
nan_pct = returns.isna().mean().mean() * 100
print(f"\nFinal returns matrix:")
print(f"  Shape      : {returns.shape[0]} dates x {returns.shape[1]} tickers")
print(f"  Date range : {returns.index.min().date()} to {returns.index.max().date()}")
print(f"  Avg NaN %%  : {nan_pct:.1f}%%")

# ─── SAVE ─────────────────────────────────────────────────────────────────────
out_path = PROC_DIR / "returns_clean.csv"
returns.to_csv(out_path)
print(f"\nSaved: {out_path}")

# Also save a small coverage summary for reference
coverage_out = returns.notna().mean().rename("coverage").reset_index()
coverage_out.columns = ["ticker", "coverage"]
coverage_out = coverage_out.sort_values("coverage", ascending=False)
coverage_out.to_csv(PROC_DIR / "ticker_coverage.csv", index=False)
print(f"Saved: {PROC_DIR / 'ticker_coverage.csv'}")

