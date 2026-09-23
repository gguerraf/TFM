"""Download GICS labels"""

import time
import pandas as pd
import yfinance as yf
from pathlib import Path

BASE_DIR  = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR  = BASE_DIR / "processed"

REQUEST_PAUSE = 0.3

PRINT_EVERY   = 50

print("Loading ticker list from returns_clean.csv...")
_peek    = pd.read_csv(PROC_DIR / "returns_clean.csv", nrows=0)
_idx_col = _peek.columns[0]
tickers  = [c for c in pd.read_csv(
                PROC_DIR / "returns_clean.csv", nrows=0).columns
            if c != _idx_col]
print(f"  Tickers to classify: {len(tickers):,}")

print("\nDownloading GICS classifications from yfinance...")
print("(This may take 20-30 minutes due to per-ticker requests)\n")

records = []
failed  = []

for i, ticker in enumerate(tickers):
    if i % PRINT_EVERY == 0:
        print(f"  Progress: {i}/{len(tickers)} tickers "
              f"({i/len(tickers)*100:.1f}%)  "
              f"classified={len(records)}  failed={len(failed)}")
    try:
        info = yf.Ticker(ticker).info
        records.append({
            "ticker":       ticker,
            "sector":       info.get("sector",       None),
            "industry":     info.get("industry",     None),
            "longName":     info.get("longName",     None),
            "exchange":     info.get("exchange",     None),
            "quoteType":    info.get("quoteType",    None),
        })
    except Exception as e:
        failed.append({"ticker": ticker, "error": str(e)})

    time.sleep(REQUEST_PAUSE)

gics_df = pd.DataFrame(records)

n_with_sector   = gics_df["sector"].notna().sum()
n_with_industry = gics_df["industry"].notna().sum()
n_sectors       = gics_df["sector"].nunique()
n_industries    = gics_df["industry"].nunique()

print(f"\n{'=' * 50}")
print(f"GICS DOWNLOAD SUMMARY")
print(f"{'=' * 50}")
print(f"  Tickers processed       : {len(tickers):,}")
print(f"  Successfully classified : {len(records):,}")
print(f"  Failed                  : {len(failed):,}")
print(f"  With sector             : {n_with_sector:,}")
print(f"  With industry           : {n_with_industry:,}")
print(f"  Unique sectors          : {n_sectors}")
print(f"  Unique industries       : {n_industries}")

print(f"\nSector distribution:")
print(gics_df["sector"].value_counts().to_string())

out_path = PROC_DIR / "gics_data.csv"
gics_df.to_csv(out_path, index=False)

if failed:
    failed_df = pd.DataFrame(failed)
    failed_df.to_csv(PROC_DIR / "gics_failed.csv", index=False)
    print(f"Failed tickers: {PROC_DIR / 'gics_failed.csv'}")

