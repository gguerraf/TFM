# Structural Dynamics and Contagion Mechanisms of Intra-ETF Shock Transmission

> **Degree Program:** Master's Thesis in Data Science (Trabajo de Fin de Máster — TFM)  
> **Author:** Gabriel  
> **Workspace Path:** [`.`](.)  
> **Code Directory:** [`src`](./code)  
> **Status:** Phase 0 Complete (Pipeline Validated) | Phase 1 In Progress (Econometric Model Refinement)

---

## 1. Executive Summary & Research Question

Exchange-Traded Funds (ETFs) have become dominant vehicles in modern equity markets. While traditional asset pricing treats constituent stocks as independent entities subject to market and industry factors, ETF basket trading creates mechanical and institutional linkages between co-constituents.

### Core Research Question
> **Does a firm-specific corporate or price shock to a large-weight ETF constituent propagate to other constituent stocks within the same ETF, and through which economic mechanisms (liquidity frictions, arbitrage trading, or informational spillovers) does this transmission operate?**

Unlike prior literature that focuses primarily on aggregate ETF-level comovement or broad volatility spillovers from index trading, this thesis isolates **firm-specific idiosyncratic shocks** using an endogenous event-study methodology and measures the directional abnormal returns induced in peer constituents of the same fund.

---

## 2. Theoretical Framework & Hypotheses

The thesis formulates five testable hypotheses addressing the propagation of idiosyncratic shocks across ETF baskets:

| Hypothesis | Description | Theoretical Channel | Expected Sign |
| :--- | :--- | :--- | :--- |
| **H1: Baseline Propagation** | A firm-specific shock to stock $i$ generates abnormal returns of the same sign in co-constituent stock $j$, scaled by the portfolio weight $w_i$. | Institutional / Basket Comovement | $\beta_1 > 0$ |
| **H2: Liquidity Channel** | Spillover magnitude is amplified (or conditioned) when the receiving stock $j$ is illiquid, as liquidity frictions exacerbate price pressure from basket rebalancing. | Market Microstructure / Price Impact | $\beta_2 \neq 0$ |
| **H3: Weight Channel** | Shocks originating in larger constituent holdings generate substantially larger cross-asset spillovers than shocks to small-weight stocks. | Basket Imbalance & Arbitrage Exposure | Interaction with $w_i$ |
| **H4: Flow Amplification** | Spillovers intensify on days characterized by large ETF creation/redemption activity and high trading volume. | Authorized Participant (AP) Arbitrage Flow | $\beta_3 > 0$ |
| **H5: Asymmetric Transmission** | Negative shocks (drawdowns/distress) propagate more strongly than positive shocks, reflecting flight-to-liquidity, loss aversion, and short-sale frictions. | Asymmetric Contagion / Downside Risk | $\delta > 0$ |

---

## 3. Scope of Analysis & ETF Universe

The sample covers daily trading from **2015-01-01 to 2026-02-12** (over 11 years, encompassing 3,155 clean trading days) across five sector and broad-market ETFs:

| ETF Ticker | Fund Name | Sector / Mandate | Benchmark Assigned | Main Analysis Status |
| :--- | :--- | :--- | :--- | :--- |
| **SPY** | SPDR S&P 500 ETF Trust | Broad US Large-Cap | `^GSPC` (S&P 500) | **Excluded** (self-benchmarking endogeneity) |
| **XME** | SPDR S&P Metals & Mining ETF | Metals & Mining (Materials) | [`XLB`](holdings/processed/benchmarks.csv) (SPDR Materials) | **Included** (~30–45 stocks, cap-tilted) |
| **XLE** | Energy Select Sector SPDR Fund | US Energy Sector | [`IXC`](holdings/processed/benchmarks.csv) (iShares Global Energy) | **Included** (~25–35 stocks, highly concentrated) |
| **IHE** | iShares U.S. Pharmaceuticals ETF | Pharmaceuticals Subsector | [`XLV`](holdings/processed/benchmarks.csv) (SPDR Health Care) | **Included** (~40–55 stocks, concentrated) |
| **XLV** | Health Care Select Sector SPDR Fund | Broad US Health Care Sector | [`^SP500-35`](holdings/processed/benchmarks.csv) (S&P Health Care) | **Included** (~60–70 stocks, sector-wide) |

> [!IMPORTANT]
> **Key Decision: SPY Exclusion from Main Econometric Estimation**  
> `SPY` constituents comprise ~500 large-cap stocks that replicate the S&P 500 index (`^GSPC`). Using `^GSPC` as the benchmark creates an intractable self-benchmarking problem: top constituents (e.g., AAPL, NVDA, MSFT) have heavy weights in the index, biasing the market model beta upwards and artificially compressing idiosyncratic abnormal returns. SPY is preserved in the data pipeline for robustness, calendar anchoring, and market-wide comparisons, but excluded from primary spillover regressions.

---

## 4. Endogenous Shock Identification

Rather than relying on noisy or incomplete corporate event databases (which suffer from publication lag, survivorship bias, and classification errors), shocks are identified **endogenously** from daily returns:

1. **Market Model:** For each constituent stock $i$ and event candidate date $t_0$, an OLS market model is estimated against its sector benchmark:
   $$R_{i,t} = \alpha_i + \beta_i R_{m,t} + \epsilon_{i,t}, \quad t \in [t_0 - 125, t_0 - 6]$$
   - Estimation window: $W = 120$ trading days.
   - Gap window: $GAP = 5$ trading days prior to $t_0$ to prevent pre-event information leakage from contaminating benchmark betas.
2. **Cumulative Abnormal Return (CAR):** Computed over the event window $[t_0, t_0 + 3]$ ($H = 3$ days, 4 trading days total):
   $$CAR_i(t_0) = \sum_{t=t_0}^{t_0+3} \left( R_{i,t} - \hat{\alpha}_i - \hat{\beta}_i R_{m,t} \right)$$
3. **Event-Specific Dynamic Threshold:** A shock is flagged if:
   $$|CAR_i(t_0)| > 1.5 \times \sqrt{H + 1} \times \hat{\sigma}_{\epsilon, i}(t_0)$$
   where $\hat{\sigma}_{\epsilon, i}(t_0)$ is the standard deviation of residuals from the stock's own 120-day estimation window.
   > [!NOTE]
   > Computing $\hat{\sigma}_{\epsilon, i}(t_0)$ from the rolling historical estimation window eliminates **lookahead bias** that was present in preliminary implementations where sample-wide standard deviations were used.

---

## 5. Key Methodological Decisions

1. **External Sector Benchmarks (Avoiding Endogeneity):**
   - Benchmarking against the ETF itself causes mechanical bias (the shocked stock is inside the index).
   - Each ETF is mapped to an external sectoral index or broader industry ETF:
     - `XME` $\rightarrow$ `XLB` (replicates broad materials, data since 1998)
     - `XLE` $\rightarrow$ `IXC` (iShares Global Energy ETF, avoids broad-market dilution of `^GSPC`)
     - `IHE` $\rightarrow$ `XLV` (broad health care contains pharma)
     - `XLV` $\rightarrow$ `^SP500-35` (official S&P 500 Health Care index)
2. **Standard Error Clustering:**
   - Observations within the same event date share cross-sectional market conditions, and repeated observations of the same receiver stock $j$ exhibit serial correlation.
   - Standard errors are two-way clustered by `event_id` ($stock_i \times t_0$) and receiver stock `stock_j`.
3. **Overlapping Event Filtering:**
   - If receiver stock $j$ experiences an idiosyncratic shock within $[t_0 - 3, t_0 + 3]$, it is purged from that event's cross-section to avoid conflating transmission with independent shocks.
4. **Log Amihud Illiquidity Metric:**
   - Replaced simple rolling return volatility with the Amihud (2002) ratio:
     $$ILLIQ_{j,t} = \frac{|R_{j,t}|}{Volume_{j,t} \times Price_{j,t}}$$
   - Transformed via $Illiq_j = \ln(1 + 10^{10} \times ILLIQ)$ to address extreme positive skewness.

---

## 6. Project Directory & Key Data Files

All data files reside in [`.`](.):

```
.\
├── src\                                # Python analysis & modeling scripts
├── holdings\                            # Raw JSON holdings and processed datasets
│   ├── spy_holdings\                    # Daily JSON holding snapshots (SPY)
│   ├── xme_holdings\                    # Daily JSON holding snapshots (XME)
│   ├── xle_holdings\                    # Daily JSON holding snapshots (XLE)
│   ├── ihe_holdings\                    # Daily JSON holding snapshots (IHE)
│   ├── xlv_holdings\                    # Daily JSON holding snapshots (XLV)
│   ├── processed\                       # Cleaned, aligned tabular CSV files
│   ├── results\                         # Regression summaries, panels, estimates
│   └── figures\                         # Thesis plots and diagnostic charts
```

### Processed Data Files (in [`holdings/processed/`](holdings/processed/))

| File Name | Dimensions / Size | Content Description |
| :--- | :--- | :--- |
| [`spy_holdings.csv`](holdings/processed/spy_holdings.csv) | 1,388,616 rows × 11 cols (151.3 MB) | Daily holdings for SPY (2015–2026), equity-filtered |
| [`xlv_holdings.csv`](holdings/processed/xlv_holdings.csv) | 170,818 rows × 11 cols (18.4 MB) | Daily holdings for XLV (Health Care) |
| [`ihe_holdings.csv`](holdings/processed/ihe_holdings.csv) | 116,218 rows × 11 cols (12.4 MB) | Daily holdings for IHE (Pharmaceuticals) |
| [`xme_holdings.csv`](holdings/processed/xme_holdings.csv) | 82,555 rows × 11 cols (9.0 MB) | Daily holdings for XME (Metals & Mining) |
| [`xle_holdings.csv`](holdings/processed/xle_holdings.csv) | 78,911 rows × 11 cols (8.5 MB) | Daily holdings for XLE (Energy) |
| [`prices_raw.csv`](holdings/processed/prices_raw.csv) | 3,157 rows × 865 cols (40.0 MB) | Daily adjusted closing prices for 864 tickers + date index |
| [`returns_clean.csv`](holdings/processed/returns_clean.csv) | 3,155 rows × 803 cols (46.2 MB) | Daily log returns for 802 tickers, filtered & calendar-aligned |
| [`benchmarks.csv`](holdings/processed/benchmarks.csv) | 3,046 rows × 5 cols (0.24 MB) | Benchmark prices (`^GSPC`, `XLB`, `IXC`, `XLV`, `^SP500-35`) |
| [`amihud.csv`](holdings/processed/amihud.csv) | 3,155 rows × 795 cols (49.5 MB) | Rolling 20-day Amihud illiquidity ratios |
| [`volume_raw.csv`](holdings/processed/volume_raw.csv) | 3,155 rows × 795 cols (21.0 MB) | Daily trading volume for constituent stocks |
| [`gics_data.csv`](holdings/processed/gics_data.csv) | 802 rows × 6 cols (0.06 MB) | GICS Sector and Industry classifications |

---

## 7. Execution Pipeline

The analytical pipeline follows an end-to-end reproducible workflow:

```
[0_validate_holdings.py] ──> Validates raw JSON counts and structures
          │
[1_load_holdings.py]     ──> Ingests JSONs, purges non-equities, resolves M&A duplicates
          │
[2_download_prices.py]   ──> Batched yfinance download of constituent prices (prices_raw.csv)
          ├─> [2b_retry_failed_tickers.py]  (Retries ticker formats)
          ├─> [2c_rename_tickers.py]        (Stitches corporate name changes)
          ├─> [2d_download_benchmarks.py]   (Downloads sector benchmarks)
          ├─> [2e_download_gics.py]         (Downloads GICS sectors & industries)
          └─> [2f_download_volume.py]       (Downloads volumes, computes Amihud)
          │
[3_compute_returns.py]   ──> Computes log returns, calendar alignment, coverage filtering
          │
[4_spillover_model.py]   ──> Vectorized shock detection, panel construction, OLS regression
          │
[5_exploratory_data_analysis.py] ──> Summary tables, concentration outliers, thesis plots
```

---

## 8. Current Project Status

- **Phase 0 (Data Pipeline Complete):** All raw holdings across 5 ETFs ingested, price histories downloaded for 864 tickers, clean returns matrix built for 802 tickers, GICS classifications compiled, and Amihud ratios computed. Validation run on XME+XLE yielded initial empirical support for $H1, H2, H5$.
- **Phase 1 (Econometric Model Refinement — Current Focus):** Transitioning from preliminary `4_spillover_model.py` to an enhanced specification incorporating event-specific dynamic thresholds, external energy benchmark `IXC`, receiver stock main effects, year-quarter time fixed effects, two-way clustered standard errors, and overlapping event controls.
- **Phases 2–6 (Upcoming):** Robustness matrix (12 checks), LightGBM/SHAP machine learning benchmark, neural/GNN extensions, and creation/redemption flow data integration.


