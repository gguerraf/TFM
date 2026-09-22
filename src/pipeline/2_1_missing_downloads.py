import pandas as pd
from pathlib import Path

PROC_DIR = (Path(__file__).resolve().parents[2] / "holdings" / "processed")

# Load what we downloaded
prices = pd.read_csv(PROC_DIR / "prices_raw.csv", index_col="date", nrows=1)
downloaded = set(prices.columns)

# Load all holdings and check coverage per ETF
ETF_LIST = ["SPY", "XME", "XLE", "IHE", "XLV"]
for etf in ETF_LIST:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv",
                     usecols=["symbol", "atDate"])
    # Most recent snapshot only
    latest = df[df["atDate"] == df["atDate"].max()]
    symbols = set(latest["symbol"].dropna())
    matched = symbols & downloaded
    pct = len(matched) / len(symbols) * 100 if symbols else 0
    print(f"{etf}: {len(matched)}/{len(symbols)} current constituents "
          f"with price data ({pct:.0f}%)")
