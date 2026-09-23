"""Inspect raw JSON holdings"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR = BASE_DIR / "processed"
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

ETF_LIST = ["SPY", "XME", "XLE", "IHE", "XLV"]

ETF_FOLDERS = {
    "SPY": "spy_holdings",
    "XME": "xme_holdings",
    "XLE": "xle_holdings",
    "IHE": "ihe_holdings",
    "XLV": "xlv_holdings",
}

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

print("Starting Exploratory Data Analysis (EDA) & Quality Report...\n")

print("="*80)
print("1. CALENDAR & MISSING DATA ANALYSIS (READING RAW JSONs)")
print("="*80)

returns = pd.read_csv(PROC_DIR / "returns_clean.csv", index_col="date", parse_dates=True)
trading_days = returns.index

calendar_results = []

for etf in ETF_LIST:
    folder_path = BASE_DIR / ETF_FOLDERS[etf]
    json_files = list(folder_path.glob("*.json"))

    valid_dates_list = []
    empty_or_bad_files = 0

    print(f"Scanning {len(json_files)} raw JSON files for {etf}...", end=" ")

    for fpath in json_files:
        try:

            date_str = fpath.stem
            date_obj = pd.to_datetime(date_str, format="%Y%m%d")

            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                holdings = data.get("holdings", [])

                if len(holdings) > 0:
                    valid_dates_list.append(date_obj)
                else:
                    empty_or_bad_files += 1
        except Exception:
            empty_or_bad_files += 1

    print("Done.")

    etf_dates = pd.DatetimeIndex(valid_dates_list).sort_values()

    if len(etf_dates) == 0:
        print(f"  -> ERROR: No valid JSON data found for {etf} in {folder_path}")
        continue

    start_date = etf_dates.min()
    end_date = etf_dates.max()
    all_days = pd.date_range(start=start_date, end=end_date, freq='D')

    missing_days = all_days.difference(etf_dates)
    missing_weekends = missing_days[missing_days.dayofweek >= 5]
    missing_trading = missing_days.intersection(trading_days)

    total = len(all_days)
    n_missing = len(missing_days)

    trading_days_etf = trading_days[(trading_days >= start_date) & (trading_days <= end_date)]

    presence = pd.Series(index=trading_days_etf, data=np.nan)
    valid_trading_dates = etf_dates.intersection(trading_days_etf)
    presence.loc[valid_trading_dates] = 1

    presence_filled = presence.ffill(limit=2)

    unresolved_anomalies = presence_filled.isna().sum()

    pct_weekends = (len(missing_weekends) / n_missing) * 100 if n_missing > 0 else 0
    pct_trading  = (len(missing_trading) / n_missing) * 100 if n_missing > 0 else 0
    pct_holidays = 100 - pct_weekends - pct_trading if n_missing > 0 else 0
    pct_unresolved = (unresolved_anomalies / n_missing) * 100 if n_missing > 0 else 0

    print(f"[{etf}] Timeframe: {start_date.date()} to {end_date.date()} ({total} days)")
    print(f"  -> Empty or unreadable JSON files found: {empty_or_bad_files}")
    print(f"  -> Total days without valid JSON data : {n_missing} ({(n_missing/total)*100:.1f}% of total time)")

    if n_missing > 0:
        print(f"     - % OF MISSING (Weekend)         : {pct_weekends:.1f}%")
        print(f"     - % OF MISSING (Market Holiday)  : {pct_holidays:.1f}%")
        print(f"     - % OF MISSING (Anomalies)       : {pct_trading:.1f}% ({len(missing_trading)} days)")
        print(f"     - UNRESOLVED AFTER 2-DAY LOCF    : {pct_unresolved:.1f}% ({unresolved_anomalies} days)")

    calendar_results.append({
        "ETF": etf,
        "Total Days": total,
        "Days without JSON": f"{n_missing} ({(n_missing/total)*100:.1f}%)",
        "% OF MISSING (Weekend)": f"{pct_weekends:.1f}%",
        "% OF MISSING (Market Holiday)": f"{pct_holidays:.1f}%",
        "% OF MISSING (Anomalies)": f"{pct_trading:.1f}% ({len(missing_trading)} days)",
        "Unresolved anomalies after 2-day LOCF": f"{pct_unresolved:.1f}% ({unresolved_anomalies} days)"
    })

df_calendar = pd.DataFrame(calendar_results)
calendar_path = PROC_DIR / "calendar_analysis_jsons.csv"
df_calendar.to_csv(calendar_path, index=False)

print("\n" + "="*80)
print("2. OUTLIERS / TOP HOLDINGS EXTRACTION")
print("="*80)

outliers_list = []
latest_holdings = []

for etf in ETF_LIST:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv", parse_dates=["atDate"])
    latest_date = df["atDate"].max()
    df_latest = df[df["atDate"] == latest_date].copy()
    df_latest["ETF"] = etf
    latest_holdings.append(df_latest)

    outliers = df_latest[df_latest["percent"] > 5.0].sort_values("percent", ascending=False)
    if not outliers.empty:
        outliers_list.append(outliers[["ETF", "symbol", "name", "percent"]])

if outliers_list:
    df_outliers = pd.concat(outliers_list)
    outliers_path = PROC_DIR / "etf_outliers_jsons.csv"
    df_outliers.to_csv(outliers_path, index=False)

print("\nGenerating 'Weight Distribution' plot...")
df_weights = pd.concat(latest_holdings)

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_weights, x="ETF", y="percent", hue="ETF", palette="Set2", legend=False)
sns.stripplot(data=df_weights, x="ETF", y="percent", color="black", alpha=0.3, jitter=True, size=3)
plt.title("ETF Weight Distribution & Dominant Assets (Latest Snapshot)")
plt.xlabel("ETF")
plt.ylabel("Weight in Portfolio (%)")
plt.tight_layout()
plt.savefig(FIG_DIR / "2_weight_distribution.png", dpi=300)
plt.close()

print("Generating 'Returns Distribution & Outliers' plot...")
flat_returns = returns.values.flatten()
flat_returns = flat_returns[~np.isnan(flat_returns)]

mu = np.mean(flat_returns)
sigma = np.std(flat_returns)

plt.figure(figsize=(10, 5))
sample_returns = np.random.choice(flat_returns, size=min(100000, len(flat_returns)), replace=False)

sns.histplot(sample_returns, bins=100, kde=False, color="steelblue", stat="density")
plt.axvline(mu + 1.5 * sigma, color="red", linestyle="--", linewidth=2, label=r"+1.5 $\sigma$ (Positive Shock)")
plt.axvline(mu - 1.5 * sigma, color="darkred", linestyle="--", linewidth=2, label=r"-1.5 $\sigma$ (Negative Shock)")

plt.xlim(-0.15, 0.15)
plt.title("Distribution of Daily Returns & Shock Identification Thresholds")
plt.xlabel("Daily Log Return")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "3_returns_distribution.png", dpi=300)
plt.close()

print("Generating 'Market Illiquidity (Amihud)' plot...")
amihud = pd.read_csv(PROC_DIR / "amihud.csv", index_col="date", parse_dates=True)
market_illiquidity = amihud.mean(axis=1)

plt.figure(figsize=(12, 5))
plt.plot(market_illiquidity.index, market_illiquidity.values, color="purple", linewidth=1.5)
plt.axvspan(pd.to_datetime("2020-02-15"), pd.to_datetime("2020-04-15"),
            color="red", alpha=0.2, label="COVID-19 Market Crash")

plt.title("Average Market Illiquidity (Amihud Ratio 20-day rolling mean)")
plt.xlabel("Date")
plt.ylabel("Amihud Illiquidity (Higher = Less Liquid)")
plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.savefig(FIG_DIR / "4_amihud_liquidity.png", dpi=300)
plt.close()

