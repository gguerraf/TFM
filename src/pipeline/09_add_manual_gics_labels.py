import pandas as pd
from pathlib import Path

PROC_DIR = (Path(__file__).resolve().parents[2] / "holdings" / "processed")
gics = pd.read_csv(PROC_DIR / "gics_data.csv", index_col="ticker")

manual = {
    "BHGE": {"sector": "Energy",        "industry": "Oil & Gas Equipment & Services"},
    "WLTW": {"sector": "Financial Services", "industry": "Insurance Brokers"},
    "CTL":  {"sector": "Communication Services", "industry": "Telecom Services"},
    "BBBY": {"sector": "Consumer Cyclical", "industry": "Specialty Retail"},
    "BMS":  {"sector": "Basic Materials", "industry": "Packaging & Containers"},
    "FISV": {"sector": "Technology",     "industry": "Information Technology Services"},
}

for ticker, vals in manual.items():
    if ticker in gics.index:
        gics.loc[ticker, "sector"]   = vals["sector"]
        gics.loc[ticker, "industry"] = vals["industry"]

gics.to_csv(PROC_DIR / "gics_data.csv")
print(f"Remaining NaN sectors: {gics['sector'].isna().sum()}")
