"""Compute clean log returns"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR = BASE_DIR / "processed"

MIN_COVERAGE = 0.30

print("Loading prices_raw.csv...")
prices = pd.read_csv(PROC_DIR / "prices_raw.csv",
                     index_col="date", parse_dates=True)
print(f"  Shape: {prices.shape[0]} dates x {prices.shape[1]} tickers")

print("Computing log returns...")
returns = np.log(prices / prices.shift(1))

returns = returns.iloc[1:]

print(f"Filtering tickers with coverage < {MIN_COVERAGE*100:.0f}%...")
coverage = returns.notna().mean()
keep_mask = coverage >= MIN_COVERAGE
returns = returns.loc[:, keep_mask]

n_dropped = (~keep_mask).sum()
print(f"  Tickers kept   : {returns.shape[1]:,}")
print(f"  Tickers dropped: {n_dropped:,}  (insufficient history)")

prices_filled = prices.ffill(limit=2)
returns_filled = np.log(prices_filled / prices_filled.shift(1)).iloc[1:]
returns_filled = returns_filled.loc[:, returns.columns]

returns = returns.combine_first(returns_filled)

calendar_anchor = "SPY" if "SPY" in returns.columns else "^GSPC"
if calendar_anchor in returns.columns:
    trading_days = returns[calendar_anchor].dropna().index
    returns = returns.reindex(trading_days)
    print(f"  Aligned to {len(trading_days):,} US trading days "
          f"(anchor: {calendar_anchor})")
else:
    print("  [WARN] No calendar anchor found. Keeping all dates.")

nan_pct = returns.isna().mean().mean() * 100
print(f"\nFinal returns matrix:")
print(f"  Shape      : {returns.shape[0]} dates x {returns.shape[1]} tickers")
print(f"  Date range : {returns.index.min().date()} to {returns.index.max().date()}")
print(f"  Avg NaN %%  : {nan_pct:.1f}%%")

out_path = PROC_DIR / "returns_clean.csv"
returns.to_csv(out_path)

coverage_out = returns.notna().mean().rename("coverage").reset_index()
coverage_out.columns = ["ticker", "coverage"]
coverage_out = coverage_out.sort_values("coverage", ascending=False)
coverage_out.to_csv(PROC_DIR / "ticker_coverage.csv", index=False)

