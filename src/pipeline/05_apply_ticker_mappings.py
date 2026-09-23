"""Apply ticker mappings to price data"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

BASE_DIR  = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR  = BASE_DIR / "processed"

START_DATE = "2014-01-01"
END_DATE   = "2026-02-12"

TICKER_MAP = {

    "ANTM":  {"new": "ELV",   "event_date": "2022-06-28", "action": "rename",
               "note": "Anthem -> Elevance Health"},
    "BHGE":  {"new": "BKR",   "event_date": "2019-09-16", "action": "rename",
               "note": "Baker Hughes GE -> Baker Hughes"},
    "WLTW":  {"new": "WTW",   "event_date": "2022-01-04", "action": "rename",
               "note": "Willis Towers Watson rename"},
    "DISCA": {"new": "WBD",   "event_date": "2022-04-11", "action": "rename",
               "note": "Discovery -> Warner Bros Discovery"},
    "DISCK": {"new": "WBD",   "event_date": "2022-04-11", "action": "rename",
               "note": "Discovery class K -> WBD"},
    "VIAC":  {"new": "PARA",  "event_date": "2022-02-16", "action": "rename",
               "note": "ViacomCBS -> Paramount Global"},
    "VIAB":  {"new": "PARA",  "event_date": "2022-02-16", "action": "rename",
               "note": "ViacomCBS class B -> Paramount"},
    "NLOK":  {"new": "GEN",   "event_date": "2022-11-07", "action": "rename",
               "note": "NortonLifeLock -> Gen Digital"},
    "CTL":   {"new": "LUMN",  "event_date": "2020-09-14", "action": "rename",
               "note": "CenturyLink -> Lumen Technologies"},
    "FLT":   {"new": "FLYW",  "event_date": "2024-02-01", "action": "rename",
               "note": "FleetCor -> Corpay"},
    "PKI":   {"new": "RVTY",  "event_date": "2023-03-06", "action": "rename",
               "note": "PerkinElmer -> Revvity"},
    "PEAK":  {"new": "DOC",   "event_date": "2024-03-01", "action": "rename",
               "note": "Healthpeak -> Physicians Realty"},
    "BLL":   {"new": "BALL",  "event_date": "2023-07-20", "action": "rename",
               "note": "Ball Corporation BLL -> BALL"},
    "GPS":   {"new": "GAP",   "event_date": "2024-08-01", "action": "rename",
               "note": "The Gap GPS -> GAP"},
    "WRK":   {"new": "SMFG",  "event_date": "2024-07-01", "action": "rename",
               "note": "WestRock merged -> Smurfit WestRock SMFG"},

    "CERN":  {"new": None, "event_date": "2022-06-07",  "action": "acquired",
               "note": "Cerner acquired by Oracle"},
    "PBCT":  {"new": None, "event_date": "2022-02-22",  "action": "acquired",
               "note": "People's United -> M&T Bank"},
    "ATVI":  {"new": None, "event_date": "2023-10-13",  "action": "acquired",
               "note": "Activision -> Microsoft"},
    "ABMD":  {"new": None, "event_date": "2022-12-22",  "action": "acquired",
               "note": "Abiomed -> J&J"},
    "XLNX":  {"new": None, "event_date": "2022-02-14",  "action": "acquired",
               "note": "Xilinx -> AMD"},
    "CTXS":  {"new": None, "event_date": "2022-09-30",  "action": "acquired",
               "note": "Citrix taken private"},
    "PXD":   {"new": None, "event_date": "2024-05-03",  "action": "acquired",
               "note": "Pioneer Natural -> ExxonMobil"},
    "TWTR":  {"new": None, "event_date": "2022-10-27",  "action": "acquired",
               "note": "Twitter taken private"},
    "ANSS":  {"new": None, "event_date": "2025-01-15",  "action": "acquired",
               "note": "Ansys -> Synopsys"},
    "FRC":   {"new": None, "event_date": "2023-05-01",  "action": "acquired",
               "note": "First Republic -> JPMorgan (FDIC)"},
    "SIVB":  {"new": None, "event_date": "2023-03-10",  "action": "acquired",
               "note": "SVB collapsed"},
    "DFS":   {"new": None, "event_date": "2024-11-18",  "action": "acquired",
               "note": "Discover -> Capital One"},
    "DRE":   {"new": None, "event_date": "2022-10-03",  "action": "acquired",
               "note": "Duke Realty -> Prologis"},
    "NLSN":  {"new": None, "event_date": "2023-10-03",  "action": "acquired",
               "note": "Nielsen taken private"},
    "DISH":  {"new": None, "event_date": "2024-01-02",  "action": "acquired",
               "note": "DISH -> EchoStar"},
    "CTLT":  {"new": None, "event_date": "2024-12-19",  "action": "acquired",
               "note": "Catalent -> Novo Holdings"},
    "ENDP":  {"new": None, "event_date": "2022-08-16",  "action": "acquired",
               "note": "Endo International bankruptcy"},
    "NGM":   {"new": None, "event_date": "2023-08-07",  "action": "acquired",
               "note": "NGM Bio -> AstraZeneca"},
    "RETA":  {"new": None, "event_date": "2023-03-08",  "action": "acquired",
               "note": "Reata -> Biogen"},

    "MMC":   {"new": "MMC",  "event_date": None, "action": "retry",
               "note": "Marsh McLennan - active"},
    "IPG":   {"new": "IPG",  "event_date": None, "action": "retry",
               "note": "Interpublic Group - active"},
    "HES":   {"new": "HES",  "event_date": None, "action": "retry",
               "note": "Hess Corp - active"},
    "MRO":   {"new": "MRO",  "event_date": None, "action": "retry",
               "note": "Marathon Oil - active"},
    "WBA":   {"new": "WBA",  "event_date": None, "action": "retry",
               "note": "Walgreens - active"},
    "CDAY":  {"new": "CDAY", "event_date": None, "action": "retry",
               "note": "Ceridian HCM - active"},
    "FBHS":  {"new": "FBHS", "event_date": None, "action": "retry",
               "note": "Fortune Brands - active"},
    "ARNC":  {"new": "ARNC", "event_date": None, "action": "retry",
               "note": "Arconic - active"},
    "ARCH":  {"new": "ARCH", "event_date": None, "action": "retry",
               "note": "Arch Resources - active"},
    "CEIX":  {"new": "CEIX", "event_date": None, "action": "retry",
               "note": "CONSOL Energy - active"},
    "SCHN":  {"new": "SCHN", "event_date": None, "action": "retry",
               "note": "Schnitzer Steel - active"},
    "TMST":  {"new": "TMST", "event_date": None, "action": "retry",
               "note": "TimkenSteel - active"},
    "HAYN":  {"new": "HAYN", "event_date": None, "action": "retry",
               "note": "Haynes International - active"},
    "AERI":  {"new": "AERI", "event_date": None, "action": "retry",
               "note": "Aerie Pharmaceuticals - active"},
    "ATRS":  {"new": "ATRS", "event_date": None, "action": "retry",
               "note": "Antares Pharma - active"},
    "ZGNX":  {"new": "ZGNX", "event_date": None, "action": "retry",
               "note": "Zogenix - active"},
    "RVNC":  {"new": "RVNC", "event_date": None, "action": "retry",
               "note": "Revance Therapeutics - active"},
    "ITCI":  {"new": "ITCI", "event_date": None, "action": "retry",
               "note": "Intra-Cellular Therapies - active"},
    "CBAY":  {"new": "CBAY", "event_date": None, "action": "retry",
               "note": "CymaBay Therapeutics - active"},
}

def download_ticker(ticker: str,
                    start: str = START_DATE,
                    end:   str = END_DATE) -> pd.Series:
    """Downloads adjusted close prices for a single ticker"""
    empty = pd.Series(dtype=float,
                      index=pd.DatetimeIndex([]),
                      name=ticker)
    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return empty
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        close.name = ticker

        close.index = pd.to_datetime(close.index)
        return close if len(close) > 20 else empty
    except Exception:
        return empty

def to_dt_series(s) -> pd.Series:
    """Converts any Series or None to a Series with a proper DatetimeIndex"""
    if s is None:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    if not isinstance(s, pd.Series) or len(s) == 0:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    try:
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

print("Loading prices_raw.csv...")

_peek      = pd.read_csv(PROC_DIR / "prices_raw.csv", nrows=0)
_first_col = _peek.columns[0]

prices = pd.read_csv(PROC_DIR / "prices_raw.csv",
                     index_col=_first_col, parse_dates=True)
prices.index.name = "date"
prices.index = pd.to_datetime(prices.index)

print(f"  Current shape: {prices.shape[0]} dates x {prices.shape[1]} tickers")

report_rows = []
n_patched   = 0
n_skipped   = 0

print(f"\nProcessing {len(TICKER_MAP)} ticker mappings...")

for old_ticker, info in TICKER_MAP.items():
    new_ticker  = info["new"]
    event_date  = pd.Timestamp(info["event_date"]) if info["event_date"] else None
    action      = info["action"]
    note        = info["note"]

    if action == "retry":
        if old_ticker in prices.columns and not prices[old_ticker].dropna().empty:
            print(f"  [SKIP] {old_ticker}: already in prices_raw")
            n_skipped += 1
            report_rows.append({"old_ticker": old_ticker, "new_ticker": new_ticker,
                                 "action": action, "status": "already_present",
                                 "note": note})
            continue

        series = download_ticker(old_ticker)
        if len(series) > 20:
            prices[old_ticker] = series
            print(f"  [OK]   {old_ticker}: downloaded {len(series)} days")
            n_patched += 1
            report_rows.append({"old_ticker": old_ticker, "new_ticker": new_ticker,
                                 "action": action, "status": "downloaded",
                                 "note": note})
        else:
            print(f"  [FAIL] {old_ticker}: no data available")
            report_rows.append({"old_ticker": old_ticker, "new_ticker": new_ticker,
                                 "action": action, "status": "failed",
                                 "note": note})
        continue

    if action == "rename":

        if old_ticker in prices.columns:
            old_series = to_dt_series(prices[old_ticker])
        else:
            old_series = download_ticker(old_ticker)

        new_series = download_ticker(new_ticker)

        if len(old_series) == 0 and len(new_series) == 0:
            print(f"  [FAIL] {old_ticker} -> {new_ticker}: no data for either")
            report_rows.append({"old_ticker": old_ticker, "new_ticker": new_ticker,
                                 "action": action, "status": "failed", "note": note})
            continue

        if event_date is not None:
            old_part = old_series[old_series.index <= event_date]
            new_part = new_series[new_series.index >  event_date]
        else:
            old_part = old_series
            new_part = new_series[~new_series.index.isin(old_series.index)]

        combined = pd.concat([old_part, new_part]).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        combined.name = old_ticker

        prices[old_ticker] = combined
        print(f"  [OK]   {old_ticker} -> {new_ticker}: "
              f"{len(old_part)} + {len(new_part)} = {len(combined)} days")
        n_patched += 1
        report_rows.append({"old_ticker": old_ticker, "new_ticker": new_ticker,
                             "action": action, "status": "concatenated",
                             "note": note})

    elif action == "acquired":
        if old_ticker in prices.columns and not prices[old_ticker].dropna().empty:
            print(f"  [SKIP] {old_ticker}: acquired, history already present")
            n_skipped += 1
            report_rows.append({"old_ticker": old_ticker, "new_ticker": None,
                                 "action": action, "status": "already_present",
                                 "note": note})
        else:
            series = download_ticker(old_ticker)
            if len(series) > 20:
                prices[old_ticker] = series
                print(f"  [OK]   {old_ticker} (acquired): {len(series)} days")
                n_patched += 1
                report_rows.append({"old_ticker": old_ticker, "new_ticker": None,
                                     "action": action, "status": "downloaded",
                                     "note": note})
            else:
                print(f"  [INFO] {old_ticker} (acquired): no historical data on Yahoo")
                report_rows.append({"old_ticker": old_ticker, "new_ticker": None,
                                     "action": action, "status": "no_history",
                                     "note": note})

prices = prices.loc[:, ~prices.columns.duplicated()]
prices.sort_index(inplace=True)
prices.to_csv(PROC_DIR / "prices_raw.csv")
print(f"\nprices_raw.csv updated: "
      f"{prices.shape[0]} dates x {prices.shape[1]} tickers")

report_df = pd.DataFrame(report_rows)
report_df.to_csv(PROC_DIR / "rename_report.csv", index=False)

print(f"\n{'=' * 55}")
print(f"RENAME / PATCH SUMMARY")
print(f"{'=' * 55}")
print(f"  Mappings processed  : {len(TICKER_MAP):,}")
print(f"  Tickers patched     : {n_patched:,}")
print(f"  Already present     : {n_skipped:,}")
for status, count in report_df["status"].value_counts().items():
    print(f"  {status:<22}: {count:,}")
print(f"\nTotal tickers in prices_raw.csv : {prices.shape[1]:,}")

