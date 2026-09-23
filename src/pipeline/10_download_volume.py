"""Download volume and compute illiquidity"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

BASE_DIR   = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR   = BASE_DIR / "processed"

START_DATE  = "2014-01-01"
END_DATE    = "2026-02-12"
BATCH_SIZE  = 100
BATCH_PAUSE = 2

AMIHUD_WINDOW = 20

print("Loading ticker list...")
_peek    = pd.read_csv(PROC_DIR / "returns_clean.csv", nrows=0)
_idx_col = _peek.columns[0]
all_cols = pd.read_csv(PROC_DIR / "returns_clean.csv", nrows=0).columns.tolist()
tickers  = [c for c in all_cols if c != _idx_col]
print(f"  Tickers: {len(tickers):,}")

print("Loading prices_raw.csv...")
_ppeak  = pd.read_csv(PROC_DIR / "prices_raw.csv", nrows=0)
_pidx   = _ppeak.columns[0]
prices  = pd.read_csv(PROC_DIR / "prices_raw.csv",
                      index_col=_pidx, parse_dates=True)
prices.index = pd.to_datetime(prices.index)
print(f"  Prices shape: {prices.shape}")

print("Loading returns_clean.csv...")
returns = pd.read_csv(PROC_DIR / "returns_clean.csv",
                      index_col=_idx_col, parse_dates=True)
returns.index = pd.to_datetime(returns.index)
print(f"  Returns shape: {returns.shape}")

print(f"\nDownloading volume in batches of {BATCH_SIZE}...")

batches      = [tickers[i:i+BATCH_SIZE]
                for i in range(0, len(tickers), BATCH_SIZE)]
vol_frames   = []
failed       = []

for idx, batch in enumerate(batches):
    print(f"  Batch {idx+1}/{len(batches)}: {len(batch)} tickers...", end=" ")
    try:
        raw = yf.download(
            batch,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
            threads=True
        )

        if isinstance(raw.columns, pd.MultiIndex):
            vol = raw["Volume"]
        else:
            vol = raw[["Volume"]].rename(columns={"Volume": batch[0]})

        empty = [t for t in batch
                 if t not in vol.columns or vol[t].isna().all()]
        failed.extend(empty)

        vol_frames.append(vol.dropna(axis=1, how="all"))
        n_ok = len(batch) - len(empty)
        print(f"OK ({n_ok} tickers with data)")

    except Exception as e:
        print(f"ERROR: {e}")
        failed.extend(batch)

    if idx < len(batches) - 1:
        time.sleep(BATCH_PAUSE)

print("\nCombining volume data...")
volume = pd.concat(vol_frames, axis=1)
volume = volume.loc[:, ~volume.columns.duplicated()]
volume.index = pd.to_datetime(volume.index)
volume.index.name = "date"
volume.sort_index(inplace=True)

volume = volume.reindex(returns.index)

print(f"  Volume shape: {volume.shape}")
print(f"  Failed tickers: {len(failed):,}")

vol_path = PROC_DIR / "volume_raw.csv"
volume.to_csv(vol_path)

print(f"\nComputing Amihud illiquidity (rolling {AMIHUD_WINDOW}-day window)...")

common = [t for t in tickers
          if t in returns.columns and t in volume.columns]
print(f"  Tickers with both returns and volume: {len(common):,}")

prices_aligned = prices.reindex(returns.index)[common]
volume_aligned = volume[common]
returns_aligned = returns[common]

dollar_vol = volume_aligned * prices_aligned
dollar_vol = dollar_vol.replace(0, np.nan)

daily_amihud = returns_aligned.abs() / dollar_vol

rolling_amihud = daily_amihud.rolling(
    window=AMIHUD_WINDOW,
    min_periods=10
).mean()

q01 = rolling_amihud.stack().quantile(0.01)
q99 = rolling_amihud.stack().quantile(0.99)
rolling_amihud = rolling_amihud.clip(lower=q01, upper=q99)

print(f"  Amihud stats:")
flat = rolling_amihud.stack()
print(f"    Mean   : {flat.mean():.2e}")
print(f"    Median : {flat.median():.2e}")
print(f"    Std    : {flat.std():.2e}")

amihud_path = PROC_DIR / "amihud.csv"
rolling_amihud.to_csv(amihud_path)

