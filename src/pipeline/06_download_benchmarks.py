"""Download benchmark price series"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

BASE_DIR  = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR  = BASE_DIR / "processed"

START_DATE = "2014-01-01"
END_DATE   = "2026-02-12"

MIN_REQUIRED_START = pd.Timestamp("2014-06-01")

BENCHMARK_CANDIDATES = {
    "SPY": [
        ("^GSPC",    "S&P 500 Index - exact benchmark for SPY"),
    ],
    "XME": [
        ("XLB",      "SPDR S&P 500 Materials ETF - XME is a subset"),
        ("^SP500-15","S&P 500 Materials Index - direct but may lack history"),
        ("^GSPC",    "S&P 500 fallback"),
    ],
    "XLE": [
        ("^SP500-10","S&P 500 Energy Index - direct but may lack history"),
        ("^GSPC",    "S&P 500 - standard fallback for sector ETFs"),
        ("IXC",      "iShares Global Energy ETF - includes non-US"),
    ],
    "IHE": [
        ("XLV",      "SPDR S&P 500 Health Care ETF - IHE is a subset"),
        ("^SP500-35","S&P 500 Health Care Index - direct"),
        ("^GSPC",    "S&P 500 fallback"),
    ],
    "XLV": [
        ("^SP500-35","S&P 500 Health Care Index - exact"),
        ("VHT",      "Vanguard Health Care ETF - alternative, data since 2004"),
        ("^GSPC",    "S&P 500 fallback"),
    ],
}

print("Downloading and evaluating benchmark candidates...")
print(f"Required date range: {START_DATE} to {END_DATE}")
print(f"Minimum start date for usability: {MIN_REQUIRED_START.date()}\n")

results   = {}
report    = []
selected  = {}

for etf, candidates in BENCHMARK_CANDIDATES.items():
    print(f"{'-' * 50}")
    print(f"ETF: {etf}")
    best_ticker = None
    best_series = None

    for ticker, description in candidates:
        try:
            raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                              auto_adjust=True, progress=False)
            if raw.empty:
                print(f"  {ticker:<15} EMPTY - no data returned")
                report.append({"etf": etf, "ticker": ticker,
                                "status": "empty", "start": None,
                                "end": None, "n_days": 0,
                                "description": description})
                continue

            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            close.index = pd.to_datetime(close.index)

            actual_start = close.index.min()
            actual_end   = close.index.max()
            n_days       = len(close)
            usable        = actual_start <= MIN_REQUIRED_START

            status = "OK" if usable else "INSUFFICIENT_HISTORY"
            print(f"  {ticker:<15} {status:<22} "
                  f"start={actual_start.date()}  "
                  f"end={actual_end.date()}  "
                  f"n={n_days:,}  | {description}")

            report.append({"etf": etf, "ticker": ticker,
                            "status": status,
                            "start": actual_start.date(),
                            "end":   actual_end.date(),
                            "n_days": n_days,
                            "description": description})

            if usable and best_ticker is None:
                best_ticker = ticker
                best_series = close
                results[ticker] = close

        except Exception as e:
            print(f"  {ticker:<15} ERROR: {e}")
            report.append({"etf": etf, "ticker": ticker,
                            "status": f"error: {e}",
                            "start": None, "end": None,
                            "n_days": 0, "description": description})

    if best_ticker:
        selected[etf] = best_ticker
        print(f"  --> SELECTED for {etf}: {best_ticker}")
    else:
        print(f"  --> WARNING: No usable benchmark found for {etf}")

print(f"\n{'=' * 50}")
print("SELECTED BENCHMARKS SUMMARY")
print(f"{'=' * 50}")
for etf, ticker in selected.items():
    s = results[ticker]
    print(f"  {etf} -> {ticker:<12} "
          f"({s.index.min().date()} to {s.index.max().date()}, "
          f"{len(s):,} days)")

if results:
    benchmarks_df = pd.DataFrame(results)
    benchmarks_df.index.name = "date"
    benchmarks_df.sort_index(inplace=True)

    out_path = PROC_DIR / "benchmarks.csv"
    benchmarks_df.to_csv(out_path)

report_df = pd.DataFrame(report)
report_df.to_csv(PROC_DIR / "benchmark_coverage.csv", index=False)

mapping_df = pd.DataFrame([
    {"etf": etf, "benchmark": ticker}
    for etf, ticker in selected.items()
])
mapping_df.to_csv(PROC_DIR / "benchmark_mapping.csv", index=False)


