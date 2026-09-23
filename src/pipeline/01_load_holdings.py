import json
import pandas as pd
from pathlib import Path

"""
01_load_holdings.py
==================
Reads all JSON holdings files for each ETF and builds one DataFrame per ETF.
Result: dictionary `holdings` with key = ETF symbol and value = DataFrame.
"""

BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")

ETF_FOLDERS = {
    "SPY": "spy_holdings",
    "XME": "xme_holdings",
    "XLE": "xle_holdings",
    "IHE": "ihe_holdings",
    "XLV": "xlv_holdings",
}

def load_etf_holdings(folder_path: Path, etf_symbol: str) -> pd.DataFrame:
    """Reads all JSON files inside folder_path and returns a DataFrame"""
    records = []
    json_files = sorted(folder_path.glob("*.json"))

    if not json_files:
        print(f"  [WARN] No JSON files found in {folder_path}")
        return pd.DataFrame()

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  [WARN] Error reading {fpath.name}: {e}")
            continue

        at_date = data.get("atDate")
        holdings_list = data.get("holdings", [])

        if not holdings_list:
            continue

        for h in holdings_list:
            records.append({
                "atDate":    at_date,
                "etf":       etf_symbol,
                "symbol":    h.get("symbol", ""),
                "name":      h.get("name", ""),
                "isin":      h.get("isin", ""),
                "cusip":     h.get("cusip", ""),
                "assetType": h.get("assetType", ""),
                "percent":   h.get("percent", None),
                "share":     h.get("share", None),
                "value":     h.get("value", None),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["atDate"] = pd.to_datetime(df["atDate"])
    df = df.sort_values("atDate").reset_index(drop=True)

    df["weight"] = df["percent"] / 100.0

    return df

def clean_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Removes non-equity rows and resolves duplicate (atDate, symbol) pairs"""
    rows_before = len(df)

    NON_EQUITY_TYPES = {"Bond", "ETP", "Fund", "Other"}
    df = df[~df["assetType"].isin(NON_EQUITY_TYPES)].copy()

    df = df[
        df["symbol"].notna() &
        (df["symbol"].str.strip() != "") &
        (df["symbol"].str.strip() != "-")
    ]

    df = df.sort_values("percent", ascending=False)
    df = df.drop_duplicates(subset=["atDate", "symbol"], keep="first")
    df = df.sort_values(["atDate", "symbol"]).reset_index(drop=True)

    rows_after = len(df)
    print(f"  -> Cleaning: {rows_before:,} rows before  |  "
          f"{rows_after:,} rows after  |  "
          f"{rows_before - rows_after:,} rows removed")

    return df

holdings = {}

for etf, folder_name in ETF_FOLDERS.items():
    folder_path = BASE_DIR / folder_name
    print(f"Loading {etf} from {folder_path} ...")
    df = load_etf_holdings(folder_path, etf)
    df = clean_holdings(df)
    holdings[etf] = df
    if not df.empty:
        print(f"  -> {len(df):,} clean rows  |  "
              f"dates: {df['atDate'].min().date()} - {df['atDate'].max().date()}")
    else:
        print("  -> Empty DataFrame")

print("\n=== Holdings loading summary (after cleaning) ===")
for etf, df in holdings.items():
    if df.empty:
        print(f"  {etf}: EMPTY")
        continue
    n_dates  = df["atDate"].nunique()
    n_stocks = df["symbol"].nunique()
    avg_per_date = len(df) / n_dates
    print(f"  {etf}: {n_dates} dates  |  {n_stocks} unique symbols  |  "
          f"avg {avg_per_date:.1f} holdings/date")

OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

for etf, df in holdings.items():
    if not df.empty:
        out_path = OUTPUT_DIR / f"{etf.lower()}_holdings.csv"
        df.to_csv(out_path, index=False)

