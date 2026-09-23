import re
import time
import pandas as pd
import yfinance as yf
from pathlib import Path

"""
04_retry_missing_tickers.py
==========================
Retries failed ticker downloads by cleaning symbol formats to match
Yahoo Finance conventions, then appends new prices to prices_raw.csv.

Cleaning rules applied (in order):
    1. Remove exchange suffixes:  "AAPL US" -> "AAPL"
    2. Remove when-issued suffix: "AMTM WI" -> "AMTM"
    3. Remove currency suffixes:  "AKRXEUR" -> "AKRX"
    4. Replace slash with dash:   "BRK/B"   -> "BRK-B"
    5. Remove trailing dots:      "BMTA."   -> "BMTA"
    6. Remove numeric-only codes: "9876590D" -> skip (not a real ticker)

Inputs:
    processed/download_report.csv   (from 02_download_prices.py)
    processed/prices_raw.csv        (from 02_download_prices.py)

Outputs:
    processed/prices_raw.csv        Updated with newly recovered tickers
    processed/download_report.csv   Updated with retry results
"""

BASE_DIR  = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR  = BASE_DIR / "processed"

START_DATE  = "2014-01-01"
END_DATE    = "2026-02-12"
BATCH_SIZE  = 50
BATCH_PAUSE = 2

SKIP_PATTERNS = [
    r'^\d+[A-Z]?$',
    r'^\d+\.\d+',
    r'^CASH_',
    r'^BLKFDS',
    r'XXXXX',
    r'\.',
    r'\.PA$|\.SW$|\.L$|\.AX$|\.BC$|\.HK$|\.SN$',
    r'^[0-9]',
]

def clean_ticker(symbol: str) -> str | None:
    """Attempts to convert a failed ticker symbol to Yahoo Finance format"""
    s = symbol.strip()

    for pattern in SKIP_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return None

    original = s

    s = re.sub(r'\s+(US|CA|LN|GY|FP|NA|SM|IM|AU|HK|SW|GB)$', '', s)

    s = re.sub(r'\s+WI$', '', s)

    cleaned = re.sub(r'(EUR|USD|CHF|GBP|JPY|SGD|CAD)$', '', s)
    if len(cleaned) >= 2:
        s = cleaned

    s = s.replace('/', '-')

    s = s.rstrip('.')

    s = s.strip()
    if not s or len(s) < 1:
        return None

    return s

print("Loading download report...")
report = pd.read_csv(PROC_DIR / "download_report.csv")
failed = report[~report["downloaded"]]["ticker"].tolist()
print(f"  Failed tickers to retry: {len(failed):,}")

print("\nApplying ticker cleaning rules...")

cleaning_map = {}
for symbol in failed:
    cleaned = clean_ticker(symbol)
    cleaning_map[symbol] = cleaned

skipped  = [s for s, c in cleaning_map.items() if c is None]
to_retry = {s: c for s, c in cleaning_map.items() if c is not None}

print(f"  Symbols skipped (non-Yahoo format): {len(skipped):,}")
print(f"  Symbols to retry after cleaning   : {len(to_retry):,}")

cleaned_unique = list(set(to_retry.values()))
print(f"  Unique cleaned tickers to attempt : {len(cleaned_unique):,}")

print("\nLoading existing prices_raw.csv...")
prices_existing = pd.read_csv(PROC_DIR / "prices_raw.csv",
                               index_col="date", parse_dates=True)
already_have = set(prices_existing.columns)
print(f"  Tickers already downloaded: {len(already_have):,}")

cleaned_to_download = [t for t in cleaned_unique if t not in already_have]
print(f"  New tickers to attempt    : {len(cleaned_to_download):,}")

if not cleaned_to_download:
    print("\nNothing new to download. All cleaned tickers already in prices_raw.csv.")
else:

    print(f"\nDownloading {len(cleaned_to_download):,} tickers "
          f"in batches of {BATCH_SIZE}...")

    batches = [cleaned_to_download[i:i+BATCH_SIZE]
               for i in range(0, len(cleaned_to_download), BATCH_SIZE)]

    new_frames  = []
    newly_ok    = []
    still_failed = []

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
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": batch[0]})

            for t in batch:
                if t in close.columns and not close[t].isna().all():
                    newly_ok.append(t)
                else:
                    still_failed.append(t)

            close = close.dropna(axis=1, how="all")
            if not close.empty:
                new_frames.append(close)

            print(f"OK ({len([t for t in batch if t in close.columns])} "
                  f"tickers with data)")

        except Exception as e:
            print(f"ERROR: {e}")
            still_failed.extend(batch)

        if idx < len(batches) - 1:
            time.sleep(BATCH_PAUSE)

    if new_frames:
        print(f"\nMerging {len(new_frames)} new batches into prices_raw.csv...")
        new_prices = pd.concat(new_frames, axis=1)
        new_prices = new_prices.loc[:, ~new_prices.columns.duplicated()]

        combined = pd.concat([prices_existing, new_prices], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined.sort_index(inplace=True)

        combined.to_csv(PROC_DIR / "prices_raw.csv")
        print(f"  prices_raw.csv updated: "
              f"{combined.shape[0]} dates x {combined.shape[1]} tickers")
        print(f"  New tickers added: {len(newly_ok):,}")
    else:
        print("\nNo new price data recovered.")
        combined = prices_existing

    print(f"  Still failed after retry: {len(still_failed):,}")

print("\nUpdating download_report.csv...")

final_tickers = set(combined.columns) if 'combined' in dir()\
                else set(prices_existing.columns)

report["cleaned_ticker"] = report["ticker"].map(
    lambda t: cleaning_map.get(t, t) if t in cleaning_map else t
)
report["downloaded"] = report.apply(
    lambda row: row["downloaded"] or (
        row["cleaned_ticker"] in final_tickers
        if pd.notna(row["cleaned_ticker"]) else False
    ),
    axis=1
)

report.to_csv(PROC_DIR / "download_report.csv", index=False)

n_total     = len(report)
n_ok        = report["downloaded"].sum()
n_fail      = (~report["downloaded"]).sum()
pct_ok      = n_ok / n_total * 100

print(f"\n{'=' * 50}")
print(f"RETRY SUMMARY")
print(f"{'=' * 50}")
print(f"  Total tickers requested : {n_total:,}")
print(f"  Successfully downloaded : {n_ok:,}  ({pct_ok:.1f}%)")
print(f"  Permanently failed      : {n_fail:,}")
print(f"\nThese are likely delisted stocks, internal fund codes,")
print(f"or foreign-listed securities not available on Yahoo Finance.")
print(f"They will be automatically excluded from the model panel.")

