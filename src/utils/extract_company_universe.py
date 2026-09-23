"""Extract the company universe"""

import pandas as pd
from pathlib import Path

BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR = BASE_DIR / "processed"

ETF_LIST = ["SPY", "XME", "XLE", "IHE", "XLV"]

print("Loading holdings CSVs...\n")

all_records = []

for etf in ETF_LIST:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv",
                     parse_dates=["atDate"])
    n_dates   = df["atDate"].nunique()
    date_min  = df["atDate"].min().date()
    date_max  = df["atDate"].max().date()
    n_symbols = df["symbol"].nunique()

    print(f"  {etf}: {n_dates} dates  |  {date_min} -> {date_max}  |  "
          f"{n_symbols} unique symbols")

    df["etf"] = etf
    all_records.append(df)

full_df = pd.concat(all_records, ignore_index=True)

global_start = full_df["atDate"].min()
global_end   = full_df["atDate"].max()
print(f"\nOverall date range: {global_start.date()} -> {global_end.date()}")

print("\nBuilding unique company list...")

company_stats = (
    full_df.groupby("symbol")
    .agg(
        name_most_common = ("name",    lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else ""),
        isin_most_common = ("isin",    lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else ""),
        first_seen       = ("atDate",  "min"),
        last_seen        = ("atDate",  "max"),
        n_observations   = ("atDate",  "count"),
        etfs             = ("etf",     lambda x: ", ".join(sorted(x.unique()))),
        n_etfs           = ("etf",     "nunique"),
    )
    .reset_index()
    .rename(columns={
        "symbol":           "ticker",
        "name_most_common": "name",
        "isin_most_common": "isin",
    })
)

company_stats["first_seen"] = company_stats["first_seen"].dt.date
company_stats["last_seen"]  = company_stats["last_seen"].dt.date

company_stats = company_stats.sort_values(
    ["n_etfs", "ticker"], ascending=[False, True]
).reset_index(drop=True)

print(f"\n{'=' * 55}")
print(f"SUMMARY")
print(f"{'=' * 55}")
print(f"  Total unique tickers      : {len(company_stats):,}")
print(f"  In only 1 ETF             : {(company_stats['n_etfs'] == 1).sum():,}")
print(f"  In 2+ ETFs (cross-listed) : {(company_stats['n_etfs'] >= 2).sum():,}")
print(f"  Data range                : {global_start.date()} -> {global_end.date()}")

print(f"\nBreakdown by ETF:")
for etf in ETF_LIST:
    n = full_df[full_df["etf"] == etf]["symbol"].nunique()
    print(f"  {etf}: {n:,} unique tickers")

print(f"\nSample of cross-listed tickers (appear in 2+ ETFs):")
cross = company_stats[company_stats["n_etfs"] >= 2][["ticker","name","etfs"]].head(10)
print(cross.to_string(index=False))

out1 = PROC_DIR / "all_companies.csv"
company_stats.to_csv(out1, index=False)

etf_membership = (
    full_df.groupby(["symbol", "etf"])
    .agg(
        name       = ("name",   lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else ""),
        isin       = ("isin",   lambda x: x.dropna().mode().iloc[0] if not x.dropna().empty else ""),
        first_seen = ("atDate", "min"),
        last_seen  = ("atDate", "max"),
        n_dates    = ("atDate", "nunique"),
    )
    .reset_index()
    .rename(columns={"symbol": "ticker"})
)
etf_membership["first_seen"] = etf_membership["first_seen"].dt.date
etf_membership["last_seen"]  = etf_membership["last_seen"].dt.date
etf_membership = etf_membership.sort_values(["etf", "ticker"]).reset_index(drop=True)

out2 = PROC_DIR / "all_companies_etf.csv"
etf_membership.to_csv(out2, index=False)

