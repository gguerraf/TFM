"""Download daily trading volume"""

import time
import pandas as pd
import yfinance as yf
from pathlib import Path

BASE_DIR   = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR   = BASE_DIR / "processed"

START_DATE  = "2014-01-01"
END_DATE    = "2026-02-12"
BATCH_SIZE  = 100
BATCH_PAUSE = 2

print("Loading ticker list")
tickers = [c for c in pd.read_csv(PROC_DIR / "prices_raw.csv", nrows=0).columns[1:]]
print(f"  Tickers: {len(tickers):,}")

print(f"\nDownloading volume in batches of {BATCH_SIZE}")

batches      = [tickers[i:i+BATCH_SIZE]
                for i in range(0, len(tickers), BATCH_SIZE)]
vol_frames   = []
failed       = []

for idx, batch in enumerate(batches):
    print(f"  Batch {idx+1}/{len(batches)}: {len(batch)} tickers", end=" ")
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

print("\nCombining volume data")
volume = pd.concat(vol_frames, axis=1)
volume = volume.loc[:, ~volume.columns.duplicated()]
volume.index = pd.to_datetime(volume.index)
volume.index.name = "date"
volume.sort_index(inplace=True)

print(f"  Volume shape: {volume.shape}")
print(f"  Failed tickers: {len(failed):,}")

volume.to_csv(PROC_DIR / "volume_raw.csv")
