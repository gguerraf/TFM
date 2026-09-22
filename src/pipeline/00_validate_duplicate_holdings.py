import pandas as pd
from pathlib import Path

PROC_DIR = (Path(__file__).resolve().parents[2] / "holdings" / "processed")

for etf in ["SPY", "XME", "XLE", "IHE", "XLV"]:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv", parse_dates=["atDate"])
    dupes = df[df.duplicated(subset=["atDate", "symbol"], keep=False)]
    dupes = dupes[dupes["symbol"].notna()]  # exclude the missing-symbol rows
    if not dupes.empty:
        print(f"\n--- {etf}: sample duplicates ---")
        print(dupes.sort_values(["atDate","symbol"])
                   .head(10)[["atDate","symbol","name","isin","percent","assetType"]]
                   .to_string(index=False))
