"""
2d2_download_energy_benchmark.py
================================
Downloads IXC (iShares Global Energy ETF) as the benchmark for XLE.
Updates benchmarks.csv and benchmark_mapping.csv.

IXC is preferred over ^GSPC for XLE because:
  - ^GSPC (S&P 500) is a broad-market index that does NOT absorb
    oil/commodity shocks, causing them to appear as idiosyncratic CARs
  - IXC is a globally diversified energy ETF that captures energy-sector
    systematic risk without the endogeneity of using XLE itself
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR = BASE_DIR / "processed"

START_DATE = "2014-01-01"
END_DATE   = "2026-02-12"

# ─── Download IXC ────────────────────────────────────────────────────────────
print("Downloading IXC (iShares Global Energy ETF)...")
raw = yf.download("IXC", start=START_DATE, end=END_DATE, auto_adjust=True)

if raw.empty:
    print("[ERROR] IXC download returned empty. Check connection.")
    exit(1)

# Extract close prices
if isinstance(raw.columns, pd.MultiIndex):
    ixc_prices = raw["Close"]["IXC"]
else:
    ixc_prices = raw["Close"]

ixc_prices = ixc_prices.dropna()
ixc_prices.name = "IXC"
print(f"  IXC: {len(ixc_prices)} trading days, "
      f"{ixc_prices.index[0].date()} to {ixc_prices.index[-1].date()}")

# ─── Update benchmarks.csv ───────────────────────────────────────────────────
bench_path = PROC_DIR / "benchmarks.csv"
benchmarks = pd.read_csv(bench_path, index_col="date", parse_dates=True)
benchmarks.index = pd.to_datetime(benchmarks.index)

# Add IXC column (align to existing index)
benchmarks["IXC"] = ixc_prices.reindex(benchmarks.index)
benchmarks.to_csv(bench_path)
print(f"  Updated benchmarks.csv: added IXC column "
      f"({benchmarks['IXC'].notna().sum()} non-null days)")

# ─── Update benchmark_mapping.csv ────────────────────────────────────────────
map_path = PROC_DIR / "benchmark_mapping.csv"
mapping = pd.read_csv(map_path)
mapping.loc[mapping["etf"] == "XLE", "benchmark"] = "IXC"
mapping.to_csv(map_path, index=False)
print(f"  Updated benchmark_mapping.csv: XLE -> IXC")

# ─── Update benchmark_coverage.csv ───────────────────────────────────────────
cov_path = PROC_DIR / "benchmark_coverage.csv"
coverage = pd.read_csv(cov_path)

# Add IXC row for XLE
new_row = pd.DataFrame([{
    "etf": "XLE",
    "ticker": "IXC",
    "status": "OK",
    "start": str(ixc_prices.index[0].date()),
    "end": str(ixc_prices.index[-1].date()),
    "n_days": len(ixc_prices),
    "description": "iShares Global Energy ETF — selected as XLE benchmark "
                   "(energy sector, avoids ^GSPC endogeneity)"
}])
coverage = pd.concat([coverage, new_row], ignore_index=True)
coverage.to_csv(cov_path, index=False)
print(f"  Updated benchmark_coverage.csv: added IXC entry")

print("\nDone. XLE benchmark is now IXC.")

