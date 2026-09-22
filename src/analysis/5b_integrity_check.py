import pandas as pd
from pathlib import Path

BASE_DIR = (Path(__file__).resolve().parents[2] / "holdings")
PROC_DIR = BASE_DIR / "processed"
ETF_LIST = ["SPY", "XME", "XLE", "IHE", "XLV"]

# Estas son las columnas que extraemos en 1_load_holdings.py
columns_to_check = ["atDate", "assetType", "cusip", "isin", "name", "percent", "share", "symbol", "value"]

results = []

for etf in ETF_LIST:
    df = pd.read_csv(PROC_DIR / f"{etf.lower()}_holdings.csv")
    total = len(df)
    
    missing_pcts = {"ETF": etf}
    for col in columns_to_check:
        if col in df.columns:
            # Contamos NaNs
            nans = df[col].isna().sum()
            # Contamos strings vacíos o guiones si es de tipo texto
            if df[col].dtype == object:
                empties = df[col].astype(str).str.strip().isin(["", "-", "nan", "None"]).sum()
            else:
                empties = 0
                
            total_missing = nans + empties
            pct = (total_missing / total) * 100
            missing_pcts[col] = f"{pct:.2f}%"
        else:
            missing_pcts[col] = "N/A"
            
    results.append(missing_pcts)

# Imprimir la tabla en formato Markdown para que sea fácil de copiar
report_df = pd.DataFrame(results)
print("\n% MISSING\n")
print(report_df.to_markdown(index=False))
