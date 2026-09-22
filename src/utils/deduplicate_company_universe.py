"""
deduplicate_company_universe.py
============================
Deduplicates all_companies.csv by grouping tickers that refer to the
same underlying company. Two tickers are considered the same company if:
    1. They share the same ISIN (primary key - most reliable), OR
    2. Their cleaned names are identical (fallback for missing ISINs)

For each group, the canonical ticker is the shortest clean US-format
ticker (e.g. "APD" wins over "APD US" and "APD.AX").

Outputs:
    processed/all_companies_deduped.csv    One row per unique company
    processed/ticker_to_canonical.csv      Mapping: ticker -> canonical ticker
"""

import re
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ------------------------------------------------------------
BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR = BASE_DIR / "processed"

# --- LOAD ---------------------------------------------------------------------
print("Loading all_companies.csv...")
df = pd.read_csv(PROC_DIR / "all_companies.csv", parse_dates=["first_seen", "last_seen"])
print(f"  Input: {len(df):,} rows")

# --- HELPER FUNCTIONS ---------------------------------------------------------

def clean_name(name: str) -> str:
    """
    Normalises a company name for fuzzy matching.
    Lowercases, strips legal suffixes, removes punctuation.
    """
    if pd.isna(name):
        return ""
    s = str(name).lower().strip()
    # Remove common legal suffixes
    suffixes = [
        r"\binc\.?$", r"\bincorporated$", r"\bcorp\.?$", r"\bcorporation$",
        r"\bltd\.?$", r"\blimited$", r"\bplc\.?$", r"\bllc\.?$",
        r"\bco\.?$", r"\bcompany$", r"\bgroup$", r"\bholdings?$",
        r"\binternational$", r"\bse$", r"\bag$", r"\bnv$", r"\bsa$",
    ]
    for suffix in suffixes:
        s = re.sub(suffix, "", s).strip()
    # Remove punctuation and extra spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ticker_priority(ticker: str) -> int:
    """
    Score for selecting the canonical ticker.
    Lower score = preferred canonical.
    Prefers: short, no spaces, no dots, no suffix.
    """
    t = str(ticker)
    score = 0
    score += len(t) * 2                          # shorter is better
    score += 100 if " " in t else 0              # penalise "APD US"
    score += 50  if "." in t else 0              # penalise "APD.AX"
    score += 30  if t.endswith("EUR") else 0     # penalise currency suffixes
    score += 30  if t.endswith("USD") else 0
    score += 20  if t[-1].isdigit() else 0       # penalise numeric suffixes
    score += 10  if len(t) > 5 else 0            # penalise long tickers
    return score


# --- BUILD ISIN GROUPS --------------------------------------------------------
print("Grouping by ISIN...")

df["name_clean"] = df["name"].apply(clean_name)

# Step 1: group by ISIN where ISIN is available and non-trivial
df["isin_valid"] = df["isin"].apply(
    lambda x: str(x).strip() if pd.notna(x) and len(str(x).strip()) >= 10 else None
)

# Build union-find structure to merge groups
parent = {i: i for i in df.index}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

# Group by valid ISIN
isin_to_indices = {}
for idx, row in df.iterrows():
    isin = row["isin_valid"]
    if isin:
        if isin not in isin_to_indices:
            isin_to_indices[isin] = []
        isin_to_indices[isin].append(idx)

for isin, indices in isin_to_indices.items():
    for i in range(1, len(indices)):
        union(indices[0], indices[i])

# Step 2: group by cleaned name as fallback (only for rows without ISIN)
name_to_indices = {}
for idx, row in df.iterrows():
    if not row["isin_valid"] and row["name_clean"]:
        name = row["name_clean"]
        if name not in name_to_indices:
            name_to_indices[name] = []
        name_to_indices[name].append(idx)

for name, indices in name_to_indices.items():
    for i in range(1, len(indices)):
        union(indices[0], indices[i])

# --- BUILD DEDUPLICATED TABLE -------------------------------------------------
print("Building deduplicated company table...")

df["group"] = df.index.map(find)

records = []
ticker_map = {}   # original ticker -> canonical ticker

for group_id, group_df in df.groupby("group"):
    # Choose canonical ticker (lowest priority score)
    group_df = group_df.copy()
    group_df["priority"] = group_df["ticker"].apply(ticker_priority)
    group_df = group_df.sort_values("priority")
    canonical = group_df.iloc[0]["ticker"]

    # Aggregate metadata across all tickers in the group
    all_tickers = sorted(group_df["ticker"].tolist())
    all_etfs    = sorted(set(
        etf.strip()
        for etfs in group_df["etfs"].dropna()
        for etf in etfs.split(",")
    ))
    best_name  = group_df.iloc[0]["name"]   # name of canonical ticker
    best_isin  = (
        group_df["isin"].dropna()
              .loc[group_df["isin"].str.len() >= 10]
              .iloc[0]
        if not group_df[group_df["isin"].str.len() >= 10]["isin"].empty
        else ""
    )

    first_seen = group_df["first_seen"].min()
    last_seen  = group_df["last_seen"].max()
    n_obs      = group_df["n_observations"].sum()

    records.append({
        "canonical_ticker": canonical,
        "name":             best_name,
        "isin":             best_isin,
        "all_tickers":      ", ".join(all_tickers),
        "n_tickers":        len(all_tickers),
        "etfs":             ", ".join(all_etfs),
        "n_etfs":           len(all_etfs),
        "first_seen":       first_seen,
        "last_seen":        last_seen,
        "n_observations":   n_obs,
    })

    for t in all_tickers:
        ticker_map[t] = canonical

deduped = pd.DataFrame(records)
deduped = deduped.sort_values(["n_etfs", "canonical_ticker"],
                               ascending=[False, True]).reset_index(drop=True)

# --- SUMMARY ------------------------------------------------------------------
print(f"\n{'=' * 55}")
print(f"DEDUPLICATION SUMMARY")
print(f"{'=' * 55}")
print(f"  Input tickers             : {len(df):,}")
print(f"  Unique companies (output) : {len(deduped):,}")
print(f"  Tickers merged away       : {len(df) - len(deduped):,}")
print(f"  Companies in 2+ ETFs      : {(deduped['n_etfs'] >= 2).sum():,}")
print(f"  Companies in 1 ETF only   : {(deduped['n_etfs'] == 1).sum():,}")

print(f"\nBreakdown by ETF (unique companies):")
for etf in ["SPY", "XME", "XLE", "IHE", "XLV"]:
    n = deduped["etfs"].str.contains(etf).sum()
    print(f"  {etf}: {n:,}")

print(f"\nSample of merged tickers (n_tickers > 1):")
merged = deduped[deduped["n_tickers"] > 1][
    ["canonical_ticker", "name", "all_tickers", "etfs"]
].head(15)
print(merged.to_string(index=False))

# --- SAVE ---------------------------------------------------------------------
out1 = PROC_DIR / "all_companies_deduped.csv"
deduped.to_csv(out1, index=False)
print(f"\nSaved: {out1}  ({len(deduped):,} unique companies)")

out2 = PROC_DIR / "ticker_to_canonical.csv"
pd.DataFrame([
    {"original_ticker": k, "canonical_ticker": v}
    for k, v in ticker_map.items()
]).sort_values("original_ticker").to_csv(out2, index=False)
print(f"Saved: {out2}  ({len(ticker_map):,} ticker mappings)")



