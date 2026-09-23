import time
import pandas as pd
import yfinance as yf
from pathlib import Path

"""
02_download_prices.py

Downloads daily adjusted closing prices for all constituent stocks
and benchmarks from yfinance, using batched requests to avoid timeouts.

Inputs:
    processed/{etf}_holdings.csv  (from 01_load_holdings.py)

Outputs:
    processed/prices_raw.csv      Wide format: index=date, columns=tickers
    processed/download_report.csv Summary of which tickers succeeded/failed
"""

BASE_DIR  = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR  = BASE_DIR / "processed"

START_DATE = "2014-01-01"
END_DATE   = "2026-02-12"

BATCH_SIZE = 100

BATCH_PAUSE = 2

ETF_LIST = ["SPY", "XME", "XLE", "IHE", "XLV"]

BENCHMARKS = {
    "SPY": "^GSPC",
    "XME": "XME",
    "XLE": "XLE",
    "IHE": "IHE",
    "XLV": "XLV",
}

print("Collecting all unique tickers from holdings CSVs...")

all_tickers = set()
for etf in ETF_LIST:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv",
                     usecols=["symbol"])
    tickers = df["symbol"].dropna().unique().tolist()
    all_tickers.update(tickers)
    print(f"  {etf}: {len(tickers)} unique tickers")

all_tickers.update(ETF_LIST)
all_tickers.update(BENCHMARKS.values())
all_tickers = sorted(all_tickers)

print(f"\nTotal unique tickers to download: {len(all_tickers):,}")

print(f"\nDownloading in batches of {BATCH_SIZE}...")

batches = [all_tickers[i:i+BATCH_SIZE]
           for i in range(0, len(all_tickers), BATCH_SIZE)]

price_frames = []
failed_tickers = []

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

        empty = [t for t in batch if t not in close.columns
                 or close[t].isna().all()]
        failed_tickers.extend(empty)

        price_frames.append(close)
        print(f"OK ({len(close.columns)} tickers with data)")

    except Exception as e:
        print(f"ERROR: {e}")
        failed_tickers.extend(batch)

    if idx < len(batches) - 1:
        time.sleep(BATCH_PAUSE)

print("\nCombining all batches...")
prices = pd.concat(price_frames, axis=1)

prices = prices.loc[:, ~prices.columns.duplicated()]
prices.index.name = "date"
prices.sort_index(inplace=True)

n_before = prices.shape[1]
prices.dropna(axis=1, how="all", inplace=True)
n_after = prices.shape[1]
print(f"  Columns with data: {n_after:,} / {n_before:,} requested")

out_path = PROC_DIR / "prices_raw.csv"
prices.to_csv(out_path)

report = pd.DataFrame({
    "ticker":  all_tickers,
    "downloaded": [t in prices.columns for t in all_tickers],
})
report.to_csv(PROC_DIR / "download_report.csv", index=False)

n_ok   = report["downloaded"].sum()
n_fail = (~report["downloaded"]).sum()
print(f"\nDownload report: {n_ok} succeeded  |  {n_fail} failed")

if n_fail > 0:
    failed = report[~report["downloaded"]]["ticker"].tolist()
    print(f"  First 20 failed: {failed[:20]}")

