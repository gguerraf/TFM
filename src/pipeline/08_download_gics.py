"""
08_download_gics.py
===================
Downloads GICS sector and industry classification for all constituent
tickers from yfinance. Used to build a meaningful Similarity_ij variable
for the spillover model.

GICS hierarchy (4 levels):
    Sector      -> broadest  (e.g. Health Care)
    Industry Group          (e.g. Pharmaceuticals, Biotechnology & Life Sciences)
    Industry    -> medium    (e.g. Pharmaceuticals)
    Sub-Industry-> narrowest (e.g. Specialty Pharmaceuticals)

We use two levels:
    - sector   : for broad similarity (same sector = 0.5)
    - industry : for narrow similarity (same industry = 1.0)

Similarity_ij encoding:
    1.00  same industry   (most similar - likely informational spillover)
    0.50  same sector, different industry
    0.00  different sector (co-membership only, no informational link)

Inputs:  processed/returns_clean.csv (ticker list)
Outputs: processed/gics_data.csv
"""

import time
import pandas as pd
import yfinance as yf
from pathlib import Path

# --- CONFIGURATION ------------------------------------------------------------
BASE_DIR  = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR  = BASE_DIR / "processed"

# Pause between requests to avoid rate limiting (seconds)
REQUEST_PAUSE = 0.3
# Batch progress print every N tickers
PRINT_EVERY   = 50

# --- LOAD TICKER LIST ---------------------------------------------------------
print("Loading ticker list from returns_clean.csv...")
_peek    = pd.read_csv(PROC_DIR / "returns_clean.csv", nrows=0)
_idx_col = _peek.columns[0]
tickers  = [c for c in pd.read_csv(
                PROC_DIR / "returns_clean.csv", nrows=0).columns
            if c != _idx_col]
print(f"  Tickers to classify: {len(tickers):,}")

# --- DOWNLOAD GICS DATA -------------------------------------------------------
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

# --- BUILD DATAFRAME ----------------------------------------------------------
gics_df = pd.DataFrame(records)

# Summary
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

# --- SAVE ---------------------------------------------------------------------
out_path = PROC_DIR / "gics_data.csv"
gics_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

if failed:
    failed_df = pd.DataFrame(failed)
    failed_df.to_csv(PROC_DIR / "gics_failed.csv", index=False)
    print(f"Failed tickers: {PROC_DIR / 'gics_failed.csv'}")



