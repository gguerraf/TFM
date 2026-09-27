"""Compute clean log returns on US trading days and Amihud illiquidity"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR = BASE_DIR / "processed"

MIN_COVERAGE = 0.30
CALENDAR_ANCHORS = ["SPY", "^GSPC"]
AMIHUD_WINDOW = 20
AMIHUD_MIN_OBS = 10
WINSOR_QUANTILES = (0.01, 0.99)

def read_wide_csv(path: Path) -> pd.DataFrame:
    """Reads a date-indexed wide CSV"""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()

def trading_calendar(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """US trading days, taken as the days with a price for the calendar anchor"""
    for anchor in CALENDAR_ANCHORS:
        if anchor in prices.columns:
            days = prices[anchor].dropna().index
            print(f"  Aligned to {len(days):,} US trading days (anchor: {anchor})")
            return days
    print("  [WARN] No calendar anchor found. Keeping all dates.")
    return prices.index

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns, dropping tickers with low coverage and filling price gaps of up to two days"""
    returns = np.log(prices / prices.shift(1)).iloc[1:]

    coverage = returns.notna().mean()
    keep_mask = coverage >= MIN_COVERAGE
    returns = returns.loc[:, keep_mask]
    print(f"  Tickers kept   : {returns.shape[1]:,}")
    print(f"  Tickers dropped: {(~keep_mask).sum():,}  (insufficient history)")

    prices_filled = prices.ffill(limit=2)
    returns_filled = np.log(prices_filled / prices_filled.shift(1)).iloc[1:]
    returns_filled = returns_filled.loc[:, returns.columns]
    return returns.combine_first(returns_filled)

def compute_amihud(returns: pd.DataFrame, prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Rolling Amihud ratio, winsorised within each date"""
    common = [t for t in returns.columns if t in volume.columns]
    dollar_vol = volume.reindex(returns.index)[common] * prices.reindex(returns.index)[common]
    dollar_vol = dollar_vol.replace(0, np.nan)

    daily_amihud = returns[common].abs() / dollar_vol
    rolling_amihud = daily_amihud.rolling(window=AMIHUD_WINDOW, min_periods=AMIHUD_MIN_OBS).mean()

    lower = rolling_amihud.quantile(WINSOR_QUANTILES[0], axis=1)
    upper = rolling_amihud.quantile(WINSOR_QUANTILES[1], axis=1)
    return rolling_amihud.clip(lower=lower, upper=upper, axis=0)

print("Loading prices_raw.csv")
prices = read_wide_csv(PROC_DIR / "prices_raw.csv")
print(f"  Shape: {prices.shape[0]} dates x {prices.shape[1]} tickers")

prices = prices.reindex(trading_calendar(prices))

print("Computing log returns")
returns = compute_returns(prices)

nan_pct = returns.isna().mean().mean() * 100
print(f"\nFinal returns matrix:")
print(f"  Shape      : {returns.shape[0]} dates x {returns.shape[1]} tickers")
print(f"  Date range : {returns.index.min().date()} to {returns.index.max().date()}")
print(f"  Avg NaN %  : {nan_pct:.1f}%")

returns.to_csv(PROC_DIR / "returns_clean.csv")

coverage_out = returns.notna().mean().rename("coverage").reset_index()
coverage_out.columns = ["ticker", "coverage"]
coverage_out = coverage_out.sort_values("coverage", ascending=False)
coverage_out.to_csv(PROC_DIR / "ticker_coverage.csv", index=False)

volume_path = PROC_DIR / "volume_raw.csv"
if volume_path.exists():
    print(f"\nComputing Amihud illiquidity (rolling {AMIHUD_WINDOW}-day window)")
    amihud = compute_amihud(returns, prices, read_wide_csv(volume_path))
    flat = amihud.stack()
    print(f"  Tickers : {amihud.shape[1]:,}")
    print(f"  Median  : {flat.median():.2e}")
    amihud.to_csv(PROC_DIR / "amihud.csv")
else:
    print("\n[WARN] volume_raw.csv not found, run 10_download_volume.py first")
