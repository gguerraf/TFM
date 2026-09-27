import json
import re
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

def normalise_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Maps suffixed or foreign tickers to the US ticker held by the same ETF"""
    symbols = df["symbol"].astype(str)
    suffixed = symbols.str.contains(r"[.\s]|(?:EUR|USD|CHF)$")
    plain = set(symbols[~suffixed].unique())
    valid_isin = df["isin"].astype(str).str.len() >= 10
    pairs = df.loc[~suffixed & valid_isin, ["isin", "symbol"]].drop_duplicates()
    counts = pairs["isin"].value_counts()
    isin_owner = pairs[pairs["isin"].isin(counts[counts == 1].index)].set_index("isin")["symbol"]

    mapping = {}
    for symbol, isin in df.loc[suffixed, ["symbol", "isin"]].drop_duplicates("symbol").itertuples(index=False):
        base = re.sub(r"(EUR|USD|CHF)$", "", re.split(r"[.\s]", symbol)[0])
        if base in plain:
            mapping[symbol] = base
        elif isin in isin_owner.index:
            mapping[symbol] = isin_owner[isin]

    df["symbol"] = df["symbol"].map(mapping).fillna(df["symbol"])
    return df

def drop_unmatched_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Removes suffixed or foreign listings that could not be matched to a US ticker"""
    symbols = df["symbol"].astype(str)
    unmatched = (symbols.str.contains(r"[.\s]|(?:EUR|USD|CHF)$")
                 & ~symbols.str.match(r"^[A-Z]{1,5}\.[AB]$"))
    return df[~unmatched].copy()

SYMBOL_CORRECTIONS = {
    "ANTM": ("ELV", None),
    "PKI": ("RVTY", None),
    "BHI": ("BKR", None),
    "BHGE": ("BKR", None),
    "ABC": ("COR", None),
    "COG": ("CTRA", None),
    "CEIX": ("CNR", None),
    "TMST": ("MTUS", None),
    "MYL": ("VTRS", None),
    "DEPO": ("ASRT", None),
    "TSO": ("ANDV", None),
    "ARNC": ("HWM", "2020-03-31"),
    "CRTNUSD": ("TMO", None),
    "IX": ("KMI", None),
}

def correct_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Maps renamed or mislabelled tickers to the current ticker, whose price history covers the whole period"""
    for old, (new, last_date) in SYMBOL_CORRECTIONS.items():
        rows = df["symbol"] == old
        if last_date is not None:
            rows &= df["atDate"] <= pd.Timestamp(last_date)
        df.loc[rows, "symbol"] = new
    return df

NON_EQUITY_TYPES = {
    "BOND", "ETP", "FUND", "OTHER", "CASH", "CASH COLLATERAL AND MARGINS",
    "FUTURES", "MONEY MARKET", "MUTUAL FUND - MONEY MARKET",
}
NON_EQUITY_NAMES = (r"CASH|MONEY MARKET|LIQUID RES|TREASURY SL|E-MINI|FUTURES|MARGIN BALANCE"
                    r"|U\.S\. DOLLAR|US DOLLAR|WARRANT|\bCVR\b|\bSTIF\b|\bMM\b"
                    r"|\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s?\d{2}$")
NON_EQUITY_SYMBOLS = {"CASH", "CASH_USD", "USD", "MARGIN_USD"}

def drop_non_equity(df: pd.DataFrame) -> pd.DataFrame:
    """Removes bonds, funds, cash, money market, futures, warrants and rights"""
    asset_type = df["assetType"].fillna("").astype(str).str.strip().str.upper()
    name = df["name"].fillna("").astype(str).str.strip().str.upper()
    symbol = df["symbol"].fillna("").astype(str).str.strip().str.upper()
    non_equity = (asset_type.isin(NON_EQUITY_TYPES)
                  | name.str.contains(NON_EQUITY_NAMES, regex=True)
                  | symbol.isin(NON_EQUITY_SYMBOLS))
    return df[~non_equity].copy()

def drop_empty_snapshots(df: pd.DataFrame) -> tuple:
    """Removes dates on which no holding has a positive weight"""
    has_weight = df["weight"].gt(0).groupby(df["atDate"]).transform("any")
    empty_dates = df.loc[~has_weight, "atDate"].nunique()
    return df[has_weight].copy(), empty_dates

def clean_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Removes non-equity rows and empty snapshots and resolves duplicate (atDate, symbol) pairs"""
    rows_before = len(df)

    df = drop_non_equity(df)

    df = df[
        df["symbol"].notna() &
        (df["symbol"].str.strip() != "") &
        (df["symbol"].str.strip() != "-")
    ]
    df = normalise_symbols(df)
    df = correct_symbols(df)
    df = drop_unmatched_listings(df)
    df, empty_dates = drop_empty_snapshots(df)

    df = df.sort_values("percent", ascending=False)
    df = df.drop_duplicates(subset=["atDate", "symbol"], keep="first")
    df = df.sort_values(["atDate", "symbol"]).reset_index(drop=True)

    rows_after = len(df)
    print(f"  -> Cleaning: {rows_before:,} rows before  |  "
          f"{rows_after:,} rows after  |  "
          f"{rows_before - rows_after:,} rows removed  |  "
          f"{empty_dates} dates without weights removed")

    return df

holdings = {}

for etf, folder_name in ETF_FOLDERS.items():
    folder_path = BASE_DIR / folder_name
    print(f"Loading {etf} from {folder_path}")
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

