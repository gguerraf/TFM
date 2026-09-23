"""Validate processed holdings files"""

import json
import pandas as pd
from pathlib import Path

BASE_DIR  = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR  = BASE_DIR / "processed"

ETF_FOLDERS = {
    "SPY": "spy_holdings",
    "XME": "xme_holdings",
    "XLE": "xle_holdings",
    "IHE": "ihe_holdings",
    "XLV": "xlv_holdings",
}

EXPECTED_START = pd.Timestamp("2015-01-01")
EXPECTED_END   = pd.Timestamp("2026-02-12")

def count_json_holdings(folder_path: Path) -> tuple:
    """Counts the number of JSON files and total holding rows in a folder"""
    json_files = sorted(folder_path.glob("*.json"))
    total_rows  = 0
    empty_files = 0

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            n = len(data.get("holdings", []))
            total_rows += n
            if n == 0:
                empty_files += 1
        except Exception:
            empty_files += 1

    return len(json_files), total_rows, empty_files

def load_csv(etf: str) -> pd.DataFrame:
    path = PROC_DIR / f"{etf.lower()}_holdings.csv"
    return pd.read_csv(path, parse_dates=["atDate"])

print("=" * 70)
print("HOLDINGS VALIDATION REPORT")
print("=" * 70)

all_ok = True

for etf, folder_name in ETF_FOLDERS.items():
    folder_path = BASE_DIR / folder_name
    print(f"\n{'-' * 60}")
    print(f"ETF: {etf}")
    print(f"{'-' * 60}")

    csv_path = PROC_DIR / f"{etf.lower()}_holdings.csv"
    if not csv_path.exists():
        print(f"  [FAIL] CSV not found: {csv_path}")
        all_ok = False
        continue
    print(f"  [OK]   CSV file found: {csv_path.name}")

    df = load_csv(etf)

    n_json, json_rows, n_empty = count_json_holdings(folder_path)
    csv_rows = len(df)

    print(f"  [INFO] JSON files in folder   : {n_json:,}")
    print(f"  [INFO] Empty/unreadable JSONs : {n_empty:,}")
    print(f"  [INFO] Total rows in JSONs    : {json_rows:,}")
    print(f"  [INFO] Total rows in CSV      : {csv_rows:,}")

    if csv_rows == json_rows:
        print(f"  [OK]   Row count matches exactly.")
    else:
        diff = csv_rows - json_rows
        print(f"  [WARN] Row count mismatch: CSV has {diff:+,} rows vs JSONs.")
        all_ok = False

    actual_start = df["atDate"].min()
    actual_end   = df["atDate"].max()
    print(f"  [INFO] Date range: {actual_start.date()} to {actual_end.date()}")

    if actual_start <= EXPECTED_START + pd.Timedelta(days=5):
        print(f"  [OK]   Start date within expected range.")
    else:
        print(f"  [WARN] Start date {actual_start.date()} later than expected "
              f"({EXPECTED_START.date()}).")
        all_ok = False

    if actual_end >= EXPECTED_END - pd.Timedelta(days=5):
        print(f"  [OK]   End date within expected range.")
    else:
        print(f"  [WARN] End date {actual_end.date()} earlier than expected "
              f"({EXPECTED_END.date()}).")
        all_ok = False

    n_unique_dates = df["atDate"].nunique()

    n_json_with_data = n_json - n_empty
    print(f"  [INFO] Unique dates in CSV    : {n_unique_dates:,}")
    print(f"  [INFO] JSONs with data        : {n_json_with_data:,}")

    if n_unique_dates == n_json_with_data:
        print(f"  [OK]   Unique dates match non-empty JSON files.")
    else:
        diff = n_unique_dates - n_json_with_data
        print(f"  [WARN] Date/JSON count mismatch: {diff:+,}. "
              f"(Could be multiple snapshots per day - check if expected.)")

    dupes = df.duplicated(subset=["atDate", "symbol"]).sum()
    if dupes == 0:
        print(f"  [OK]   No duplicate (date, symbol) pairs.")
    else:
        print(f"  [WARN] {dupes:,} duplicate (date, symbol) pairs found.")
        all_ok = False

    equity_df = df[df["percent"] > 0].copy()
    weight_sum = equity_df.groupby("atDate")["percent"].sum()
    median_sum = weight_sum.median()
    pct_dates_near_100 = ((weight_sum >= 95) & (weight_sum <= 105)).mean() * 100

    print(f"  [INFO] Median sum of percent per date : {median_sum:.2f}%")
    print(f"  [INFO] Dates where sum is 95-105%     : {pct_dates_near_100:.1f}%")

    if pct_dates_near_100 >= 80:
        print(f"  [OK]   Weight sums look reasonable.")
    else:
        print(f"  [WARN] Many dates have weight sums far from 100%. "
              f"Check for missing holdings or data quality issues.")
        all_ok = False

    for col in ["atDate", "symbol", "percent"]:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            print(f"  [OK]   No missing values in '{col}'.")
        else:
            print(f"  [WARN] {n_missing:,} missing values in '{col}'.")
            all_ok = False

    avg_holdings_per_date = len(df) / n_unique_dates
    print(f"  [INFO] Avg holdings per date  : {avg_holdings_per_date:.1f}")
    print(f"  [INFO] Unique symbols total   : {df['symbol'].nunique():,}")

print(f"\n{'=' * 70}")
if all_ok:
    print("FINAL RESULT: ALL CHECKS PASSED. Data looks complete and consistent.")
else:
    print("FINAL RESULT: SOME WARNINGS FOUND. Review the items marked [WARN].")
print("=" * 70)

