# Instructions for AI Agents & Collaborators

> **Project:** Intra-ETF Shock Transmission & Spillover Analysis  
> **Repository Base:** [`.`](.)  
> **Scripts Directory:** [`src`](./code)

---

## 1. Project Overview (Brief)

This Master's Thesis investigates whether firm-specific abnormal return shocks to major constituent stocks of exchange-traded funds (ETFs) spill over to other stocks within the same ETF. The study examines five sector/index ETFs (`SPY`, `XME`, `XLE`, `IHE`, `XLV`) over the 2015–2026 period and quantifies transmission through liquidity, arbitrage, and informational channels using panel econometrics and machine learning.

---

## 2. Mandatory Reading Order

Before proposing modifications or running scripts, AI agents **must** review the documentation in this order:

1. [`context.md`](src/context.md) — Research questions, economic hypotheses, dataset definitions, and high-level decisions.
2. [`methodology.md`](src/methodology.md) — Detailed econometric specifications, market model mathematics, shock detection rules, and robustness check design.
3. [`architecture.md`](src/architecture.md) — System architecture, data flow diagrams, file schemas, dimensions, and script dependencies.
4. [`progress.md`](src/progress.md) — Phase status, implementation log, preliminary results, and prioritized task backlog.

---

## 3. Python Environment & Execution

All Python operations must use the dedicated Python 3.11 installation on the system.

### Environment Specification
- **Interpreter Path:**  
  `C:\Users\gabri\AppData\Local\Programs\Python\Python311\python.exe`
- **Invocation Syntax (PowerShell):**
  ```powershell
  & "C:\Users\gabri\AppData\Local\Programs\Python\Python311\python.exe" script_name.py
  ```

### Primary Packages
| Package | Version / Purpose | Key Modules Used |
| :--- | :--- | :--- |
| `pandas` | High-performance tabular data | Time series alignment, date parsing, panel construction |
| `numpy` | Vectorized matrix algebra | Batch OLS using stride tricks & einsum |
| `statsmodels` | Econometric modeling | `smf.ols`, clustered covariance matrices |
| `linearmodels` | Panel data econometrics | `PanelOLS` for two-way fixed effects and multi-way clustering |
| `yfinance` | Market data ingestion | Daily adjusted prices, volume, dividends, splits |
| `matplotlib` | Visualization | Publication plots, coefficient bar charts |
| `seaborn` | Statistical visualization | Distribution plots, heatmaps |
| `lightgbm` | Machine learning baseline | Gradient boosting for non-linear spillover prediction |
| `shap` | Model interpretability | TreeExplainer, feature importance & interaction values |

---

## 4. Code & Directory Conventions

Agents must strictly respect the following code organization rules:

### Path Resolution
- Never hardcode user-specific temporary paths or relative paths without a base anchor.
- Always use `pathlib.Path` with a standard base directory pattern:
  ```python
  from pathlib import Path

  BASE_DIR   = (Path(__file__).resolve().parents[2] / "holdings")
  PROC_DIR   = BASE_DIR / "processed"
  OUTPUT_DIR = BASE_DIR / "results"
  FIG_DIR    = BASE_DIR / "figures"
  ```

### Directory Roles
- [`holdings/processed/`](holdings/processed/) — Invariable location for cleaned CSVs (`prices_raw.csv`, `returns_clean.csv`, `benchmarks.csv`, etc.).
- [`holdings/results/`](holdings/results/) — Destination for model estimation logs, tabular summaries, and regression tables (`panel_full.csv`, `regression_results.txt`).
- [`holdings/figures/`](holdings/figures/) — Output directory for plots, diagnostic charts, and visual figures.
- [`src/`](src/) — Repository for all executable Python scripts and Markdown documentation.

### Script Naming Conventions
- Scripts follow a sequential numeric prefix indicating their position in the pipeline:
  - `0_*` : Raw data validation and integrity verification.
  - `1_*` : Holdings parsing, cleaning, and filtering.
  - `2_*` : Market data, volume, and benchmark ingestion via yfinance.
  - `3_*` : Returns calculation, coverage screening, and calendar alignment.
  - `4_*` : Econometric modeling and panel regressions.
  - `5_*` : Exploratory data analysis (EDA), reporting, and visualization.

---

## 5. Sequential Execution Order

When executing the data pipeline from scratch, follow this exact sequence:

1. [`0_validate_holdings.py`](src/0_validate_holdings.py) — Verifies raw JSON integrity and completeness across all 5 ETF folders.
2. [`1_load_holdings.py`](src/1_load_holdings.py) — Extracts daily holding records, strips cash/bonds, and outputs `{etf}_holdings.csv`.
3. [`get_all_companies_deduped.py`](src/get_all_companies_deduped.py) — Deduplicates tickers across funds via ISIN and clean names.
4. [`2_download_prices.py`](src/2_download_prices.py) — Downloads adjusted closes for all constituent tickers into `prices_raw.csv`.
5. [`2b_retry_failed_tickers.py`](src/2b_retry_failed_tickers.py) — Retries missing tickers with alternative symbols.
6. [`2c_rename_tickers.py`](src/2c_rename_tickers.py) — Stitches price histories for renamed or acquired companies (e.g., ANTM $\rightarrow$ ELV).
7. [`2d_download_benchmarks.py`](src/2d_download_benchmarks.py) — Downloads benchmark series (`^GSPC`, `XLB`, `IXC`, `XLV`, `^SP500-35`).
8. [`2e_download_gics.py`](src/2e_download_gics.py) — Fetches GICS sector/industry metadata into `gics_data.csv`.
9. [`2f_download_volume.py`](src/2f_download_volume.py) — Downloads daily trading volume and computes rolling Amihud illiquidity (`amihud.csv`).
10. [`3_compute_returns.py`](src/3_compute_returns.py) — Computes daily log returns, applies $\le 2$-day LOCF, and filters tickers with $<30\%$ coverage.
11. [`4_spillover_model.py`](src/4_spillover_model.py) (Baseline reference) or improved model scripts (`4a_improved_model.py`).
12. [`5_exploratory_data_analysis.py`](src/5_exploratory_data_analysis.py) — Generates thesis figures, calendar breakdown, and concentration tables.

---

## 6. Key Model Parameters

Always maintain consistency with the core parameters established in the research design:

```python
ESTIMATION_WINDOW = 120   # Trading days for market model OLS estimation (W)
GAP               = 5     # Trading days between estimation window and event date
EVENT_H           = 3     # CAR event window horizon: [t0, t0 + EVENT_H] (4 days total)
SHOCK_THRESHOLD   = 1.5   # Threshold multiplier for dynamic shock identification
ILLIQ_WINDOW      = 20    # Rolling window for Amihud illiquidity ratio
MISPRICING_WINDOW = 5     # Rolling window for pre-event ETF premium/discount proxy
```

---

## 7. Common Pitfalls & Critical Guardrails

> [!WARNING]
> **1. Lookahead Bias in Shock Identification:**  
> Never compute the shock standard deviation $\sigma_{\epsilon}$ across the entire multi-year sample. In early implementations, this created severe lookahead bias. The standard error of the regression **must be calculated strictly from the 120-day pre-event estimation window residuals**:
> $$\hat{\sigma}_{\epsilon, i}(t_0) = \sqrt{\frac{1}{W - 2} \sum_{t=t_0 - 125}^{t_0 - 6} \hat{\epsilon}_{i,t}^2}$$

> [!WARNING]
> **2. Benchmark Endogeneity:**  
> Never use the ETF return as the market benchmark for its own constituents. Doing so deflates abnormal returns for heavy constituents because they move the ETF price mechanically. Always use the designated external sector benchmark (e.g., `XLB` for `XME`, `IXC` for `XLE`).

> [!WARNING]
> **3. Temporal Splits for Machine Learning:**  
> Never use random $K$-fold cross-validation or standard `train_test_split` on panel time-series data. Financial markets exhibit strong temporal dependencies. Always use chronological walk-forward splits (e.g., train on 2015–2021, validate on 2022–2023, test on 2024–2026) or Purged Group TimeSeriesSplit with an embargo period around event windows.

> [!WARNING]
> **4. Standard Error Clustering:**  
> Standard OLS standard errors underestimate variance due to contemporaneously correlated shocks across peer stocks on date $t_0$ and repeated observations of stock $j$. Regressions must specify two-way clustering on `event_id` ($stock_i \times t_0$) and receiver stock `stock_j`.

> [!WARNING]
> **5. Non-Linear Illiquidity Scaling:**  
> The raw Amihud ratio has severe skewness and extreme outliers that destabilize linear regressions. Always log-transform the measure using $\text{Illiq}_j = \ln(1 + 10^{10} \times \text{ILLIQ}_{j,t_0})$.

---

## 8. Protocol for Modifying Models

To ensure strict reproducibility and enable before-and-after comparison:

1. **Preserve Baseline:** **NEVER overwrite [`4_spillover_model.py`](src/4_spillover_model.py)**. It serves as the benchmark reference representing preliminary Phase 0 results.
2. **Versioned Scripts:** Create new files following a clear naming scheme:
   - [`4a_improved_model.py`](src/4a_improved_model.py) : Econometric improvements (event-specific threshold, IXC benchmark, two-way clustering, main effects, time FE).
   - `4b_robustness_checks.py` : Systematic loop over the 12 robustness specifications.
   - `4c_ml_spillover.py` : LightGBM and SHAP feature importance analysis.
3. **Artifact Logging:** Save all regression logs and metrics with distinct filenames in `holdings/results/` (e.g., `results_improved_model.txt`, `results_robustness.csv`).


