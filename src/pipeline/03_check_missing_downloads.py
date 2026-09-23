import pandas as pd
from pathlib import Path

PROC_DIR = (Path(__file__).resolve().parents[2] / "holdings" / "processed")

prices = pd.read_csv(PROC_DIR / "prices_raw.csv", index_col="date", nrows=1)
downloaded = set(prices.columns)

ETF_LIST = ["SPY", "XME", "XLE", "IHE", "XLV"]
for etf in ETF_LIST:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv",
                     usecols=["symbol", "atDate"])

    latest = df[df["atDate"] == df["atDate"].max()]
    symbols = set(latest["symbol"].dropna())
    matched = symbols & downloaded
    pct = len(matched) / len(symbols) * 100 if symbols else 0
    print(f"{etf}: {len(matched)}/{len(symbols)} current constituents "
          f"with price data ({pct:.0f}%)")
