import re
import time
import pandas as pd
import yfinance as yf
from pathlib import Path

"""
2b_retry_failed_tickers.py
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
    processed/download_report.csv   (from 2_download_prices.py)
    processed/prices_raw.csv        (from 2_download_prices.py)

Outputs:
    processed/prices_raw.csv        Updated with newly recovered tickers
    processed/download_report.csv   Updated with retry results
"""

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE_DIR  = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR  = BASE_DIR / "processed"

START_DATE  = "2014-01-01"
END_DATE    = "2026-02-12"
BATCH_SIZE  = 50    # smaller batches for retry to reduce noise
BATCH_PAUSE = 2

# Patterns that are clearly not real Yahoo tickers — skip these entirely
SKIP_PATTERNS = [
    r'^\d+[A-Z]?$',           # pure numeric codes: "9876590D", "2481632D"
    r'^\d+\.\d+',             # numeric with dots: "46729666"
    r'^CASH_',                # cash positions: "CASH_USD", "CASH_JPY"
    r'^BLKFDS',               # BlackRock internal fund codes
    r'XXXXX',                 # placeholder codes
    r'\.',                    # foreign exchange dots: "ABC.AX", "0Y2A.L"
    r'\.PA$|\.SW$|\.L$|\.AX$|\.BC$|\.HK$|\.SN$',  # foreign exchanges
    r'^[0-9]',                # starts with digit (usually foreign or internal)
]

# ─── TICKER CLEANING FUNCTION ─────────────────────────────────────────────────
def clean_ticker(symbol: str) -> str | None:
    """
    Attempts to convert a failed ticker symbol to Yahoo Finance format.
    Returns None if the symbol is clearly not a valid Yahoo ticker.

    Transformations applied in order:
        1. Strip whitespace
        2. Skip patterns that are never valid Yahoo tickers
        3. Remove exchange suffixes (space + 2-letter code): "AAPL US" -> "AAPL"
        4. Remove when-issued suffix: "AMTM WI" -> "AMTM"
        5. Remove glued currency suffixes: "AKRXEUR" -> "AKRX"
        6. Replace slash with dash: "BRK/B" -> "BRK-B"
        7. Remove trailing dots: "BMTA." -> "BMTA"
        8. Skip if result is empty or still looks invalid
    """
    s = symbol.strip()

    # Step 2: skip clearly non-Yahoo symbols
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, s, re.IGNORECASE):
            return None

    original = s

    # Step 3: remove exchange suffixes (space + 2-letter country code)
    s = re.sub(r'\s+(US|CA|LN|GY|FP|NA|SM|IM|AU|HK|SW|GB)$', '', s)

    # Step 4: remove when-issued suffix
    s = re.sub(r'\s+WI$', '', s)

    # Step 5: remove glued currency suffixes
    # Only remove if the remaining ticker is at least 2 chars long
    cleaned = re.sub(r'(EUR|USD|CHF|GBP|JPY|SGD|CAD)$', '', s)
    if len(cleaned) >= 2:
        s = cleaned

    # Step 6: replace slash with dash (share classes like BRK/B -> BRK-B)
    s = s.replace('/', '-')

    # Step 7: remove trailing dots
    s = s.rstrip('.')

    # Step 8: skip if result is empty, same as original (no change made),
    # or still contains problematic characters
    s = s.strip()
    if not s or len(s) < 1:
        return None

    return s


# ─── LOAD FAILED TICKERS ──────────────────────────────────────────────────────
print("Loading download report...")
report = pd.read_csv(PROC_DIR / "download_report.csv")
failed = report[~report["downloaded"]]["ticker"].tolist()
print(f"  Failed tickers to retry: {len(failed):,}")

# ─── APPLY CLEANING ───────────────────────────────────────────────────────────
print("\nApplying ticker cleaning rules...")

# Build mapping: original_symbol -> cleaned_symbol
cleaning_map = {}   # original -> cleaned (or None if skipped)
for symbol in failed:
    cleaned = clean_ticker(symbol)
    cleaning_map[symbol] = cleaned

skipped  = [s for s, c in cleaning_map.items() if c is None]
to_retry = {s: c for s, c in cleaning_map.items() if c is not None}

print(f"  Symbols skipped (non-Yahoo format): {len(skipped):,}")
print(f"  Symbols to retry after cleaning   : {len(to_retry):,}")

# Deduplicate cleaned tickers (multiple originals may map to same cleaned ticker)
# e.g. "AAPL US" and "AAPL" both clean to "AAPL"
cleaned_unique = list(set(to_retry.values()))
print(f"  Unique cleaned tickers to attempt : {len(cleaned_unique):,}")

# ─── LOAD EXISTING PRICES TO AVOID RE-DOWNLOADING ─────────────────────────────
print("\nLoading existing prices_raw.csv...")
prices_existing = pd.read_csv(PROC_DIR / "prices_raw.csv",
                               index_col="date", parse_dates=True)
already_have = set(prices_existing.columns)
print(f"  Tickers already downloaded: {len(already_have):,}")

# Only retry tickers we don't already have
cleaned_to_download = [t for t in cleaned_unique if t not in already_have]
print(f"  New tickers to attempt    : {len(cleaned_to_download):,}")

if not cleaned_to_download:
    print("\nNothing new to download. All cleaned tickers already in prices_raw.csv.")
else:
    # ─── BATCH DOWNLOAD ───────────────────────────────────────────────────────
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

            # Track per-ticker success
            for t in batch:
                if t in close.columns and not close[t].isna().all():
                    newly_ok.append(t)
                else:
                    still_failed.append(t)

            # Keep only columns with actual data
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

    # ─── MERGE WITH EXISTING PRICES ───────────────────────────────────────────
    if new_frames:
        print(f"\nMerging {len(new_frames)} new batches into prices_raw.csv...")
        new_prices = pd.concat(new_frames, axis=1)
        new_prices = new_prices.loc[:, ~new_prices.columns.duplicated()]

        # Align index to existing prices (same date range)
        combined = pd.concat([prices_existing, new_prices], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined.sort_index(inplace=True)

        # Save updated prices
        combined.to_csv(PROC_DIR / "prices_raw.csv")
        print(f"  prices_raw.csv updated: "
              f"{combined.shape[0]} dates x {combined.shape[1]} tickers")
        print(f"  New tickers added: {len(newly_ok):,}")
    else:
        print("\nNo new price data recovered.")
        combined = prices_existing

    print(f"  Still failed after retry: {len(still_failed):,}")

# UPDATE DOWNLOAD REPORT 
print("\nUpdating download_report.csv...")

# Mark originally failed tickers as recovered if their cleaned version
# is now in the prices matrix
final_tickers = set(combined.columns) if 'combined' in dir() \
                else set(prices_existing.columns)

# Add retry info to report
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

# FINAL SUMMARY 
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

