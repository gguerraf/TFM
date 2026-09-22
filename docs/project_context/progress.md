# Project Roadmap, Progress Tracking & Changelog

> **Master's Thesis:** *Structural Dynamics and Contagion Mechanisms of Intra-ETF Shock Transmission*  
> **Repository:** [`.`](.)  
> **Code Directory:** [`src`](./code)  
> **Last Updated:** February 2026

---

## 1. Overall Project Status Summary

| Phase | Milestone / Objective | Status | Target Completion | Key Deliverables |
| :---: | :--- | :---: | :---: | :--- |
| **Phase 0** | Data Pipeline Ingestion, Cleaning & Validation | **COMPLETED** | Jan 2026 | Ingested 5 ETFs, cleaned prices/returns, Amihud, GICS, validation run ($N=285,752$) |
| **Phase 1** | Econometric Model Refinements & Panel Re-estimation | **IN PROGRESS** | Feb 2026 | Dynamic shock thresholds, IXC benchmark, two-way clustering, main effects, time FE |
| **Phase 2** | Robustness Check Suite (12 Specifications) | **PENDING** | Mar 2026 | Parameter sensitivity, multi-factor models, sub-period tests, placebo checks |
| **Phase 3** | Machine Learning Baseline (LightGBM + SHAP) | **PENDING** | Mar 2026 | Non-linear feature importance, partial dependence, purged temporal CV |
| **Phase 4** | Deep Neural Network Baseline | **PENDING** | Apr 2026 | Multi-layer perceptron with entity embeddings (conditional on ML gain) |
| **Phase 5** | Graph Neural Network (GNN) Formulation | **PENDING** | Apr 2026 | Dynamic constituent bipartite graph (conditional on non-linear value) |
| **Phase 6** | Creation/Redemption ETF Flow Channel ($\beta_3$) | **PENDING** | May 2026 | Daily shares outstanding changes, flow-induced price pressure estimation |

---

## 2. Phase 0: Data Pipeline & Validation Run (COMPLETED)

### Key Achievements (Completed January 2026)
1. **Raw Holdings Ingestion:**
   - Ingested daily JSON holdings files for all 5 target ETFs (`SPY`, `XME`, `XLE`, `IHE`, `XLV`) spanning 2015-01-01 to 2026-02-12.
   - Total parsed holdings rows across ETFs:
     - `SPY`: 1,388,616 records
     - `XLV`: 170,818 records
     - `IHE`: 116,218 records
     - `XME`: 82,555 records
     - `XLE`: 78,911 records
   - Filtered out cash equivalents, bond positions, and third-party fund placeholders while strictly preserving reported equity portfolio weights without artificial renormalisation.
   - Handled intra-day corporate actions and M&A ticker collisions by keeping the highest-weight active constituent row.
2. **Constituent Price & Return Matrix Construction:**
   - Identified 1,752 raw constituent tickers, deduplicated to 1,120 unique corporate entities via ISIN mapping and legal name clustering.
   - Successfully downloaded daily historical prices for 864 tickers via `yfinance` in batches of 100.
   - Executed ticker format cleaning (`2b_retry_failed_tickers.py`) and historical ticker renaming stitches (`2c_rename_tickers.py`, e.g., ANTM $\rightarrow$ ELV).
   - Filtered out ill-covered tickers ($<30\%$ history), producing a clean log-return matrix of **802 constituent stocks across 3,155 trading days** (`returns_clean.csv`) aligned to the US trading calendar anchored on SPY.
3. **Sector Benchmark Integration:**
   - Downloaded daily historical benchmarks (`benchmarks.csv`) covering 2014-01-02 to 2026-02-11 (3,046 trading days) for `^GSPC`, `XLB`, `IXC`, `XLV`, and `^SP500-35`.
4. **Mechanism Variable Extraction:**
   - Extracted GICS Sector and Industry classifications for 802 tickers (`gics_data.csv`), enabling the continuous informational similarity variable:
     $$Similarity_{ij} \in \{0.0, 0.5, 1.0\}$$
   - Downloaded daily share trading volume (`volume_raw.csv`) and computed rolling 20-day Amihud (2002) illiquidity ratios (`amihud.csv`), replacing primitive return volatility proxies.

### Preliminary Results: Validation Run (XME + XLE)
- **Sample:** Validation run restricted to `XME` (Metals & Mining, ~30 stocks) and `XLE` (Energy, ~30 stocks) to verify pipeline integrity prior to scaling.
- **Panel Dimensions:** 285,752 observation rows across 27,948 identified shock events.
- **Model Comparison (Initial Proxy vs. Enhanced Specification):**

| Regression Coefficient | Economic Hypothesis | Version 1 (Vol & Binary Sim) | Version 2 (Amihud & GICS) | Statistical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **$\beta_1$ (Baseline Propagation)** | H1: Basket Comovement | $+0.023$ ($p = 0.928$) | **$+2.043$** ($p < 0.001$) | Highly Significant (***) |
| **$\beta_2$ (Liquidity Channel)** | H2: Price Pressure / Friction | $+33.41$ ($p < 0.001$) | **$-10.647$** ($p = 0.004$) | Highly Significant (***) |
| **$\beta_4$ (Arbitrage Channel)** | H4: ETF Premium/Discount | $+0.847$ ($p = 0.714$) | **$-4.423$** ($p = 0.068$) | Marginally Significant (*) |
| **$\beta_5$ (Informational)** | H1 (Info): GICS Relatedness | $+0.235$ ($p < 0.001$) | **$+0.403$** ($p < 0.001$) | Highly Significant (***) |
| **$\delta$ (Negative Asymmetry)** | H5: Downside Contagion | — | $+0.433$ ($p = 0.104$) | Subsample Significant |

#### Key Insights from Validation Run
- **Sign Reversal of $\beta_2$:** Under rolling return volatility, $\beta_2$ was positive, but substituting true Amihud price impact caused a sign reversal to $-10.65$ ($p=0.004$). This reveals that return volatility was capturing noise, whereas genuine illiquidity acts as an execution friction dampening immediate mechanical basket transmission over 3 days.
- **Asymmetry Subsample Finding:** While $\delta$ was marginally insignificant in the pooled panel, subsample regressions revealed strong structural asymmetry:
  - Negative shocks activated both the liquidity channel ($\beta_2 = -22.83, p < 0.001$) and arbitrage channel ($\beta_4 = -7.50, p = 0.027$).
  - Positive shocks showed no statistically significant activation of these friction channels.

---

## 3. Phase 1: Econometric Model Improvements (IN PROGRESS)

Based on diagnostic evaluations of [`4_spillover_model.py`](src/4_spillover_model.py), Phase 1 implements 8 vital econometric corrections in [`4a_improved_model.py`](src/4a_improved_model.py):

- [ ] **1. Eliminate Lookahead Bias in Shock Identification:**
  - *Current issue:* `4_spillover_model.py` flags shocks using the stock's full-sample standard deviation of CARs ($\text{std}(CAR_i)$ over 2015–2026).
  - *Fix:* Compute dynamic shock threshold strictly using the estimation window residual variance:
    $$|CAR_i(t_0)| > 1.5 \times \sqrt{H + 1} \times \hat{\sigma}_{\epsilon, i}(t_0), \quad \hat{\sigma}_{\epsilon, i}(t_0) = \sqrt{\frac{1}{118}\sum_{t=t_0-125}^{t_0-6} \hat{\epsilon}_{i,t}^2}$$
- [ ] **2. Correct XLE Energy Benchmark Assignment:**
  - *Current issue:* `XLE` defaulted to `^GSPC` (S&P 500), which conflated market-wide movements with energy sector shocks.
  - *Fix:* Map `XLE` to [`IXC`](holdings/processed/benchmarks.csv) (iShares Global Energy ETF), which has complete daily coverage since 2014.
- [ ] **3. Formally Exclude SPY from Main Regression Analysis:**
  - *Current issue:* Including `SPY` causes severe self-benchmarking endogeneity against `^GSPC`.
  - *Fix:* Exclude `SPY` from primary spillover estimations; retain `XME`, `XLE`, `IHE`, and `XLV` as the main analytical universe.
- [ ] **4. Add Main Effects to Regression Specification:**
  - *Current issue:* Interacting terms ($Shock \times w_i \times Illiq$) without individual main effects violates the hierarchy principle of regression analysis, risking omitted variable bias.
  - *Fix:* Add level terms $\gamma_1 Illiq_j + \gamma_2 Mispricing_k + \gamma_3 Similarity_{ij} + \gamma_4 Neg_e$.
- [ ] **5. Implement Two-Way Clustered Standard Errors:**
  - *Current issue:* One-way clustering on `event_id` only accounts for cross-sectional peer correlations on date $t_0$, ignoring serial correlation in repeated observations of receiver stock $j$.
  - *Fix:* Estimate standard errors clustered simultaneously by `event_id` and `stock_j` via `linearmodels.panel.PanelOLS`.
- [ ] **6. Add Year-Quarter Time Fixed Effects ($FE_{yq}$):**
  - *Current issue:* Stock fixed effects control for cross-sectional heterogeneity, but macroeconomic cycles (e.g., COVID market panic, interest rate tightening) remain uncontrolled.
  - *Fix:* Include categorical time dummies for each calendar year-quarter.
- [ ] **7. Overlapping Event Decontamination Filter:**
  - *Current issue:* If receiver stock $j$ announces its own major news concurrently, $AR_j$ is contaminated.
  - *Fix:* Discard peer observation $(e, j)$ if stock $j$ experienced a firm-specific shock within $[t_0 - 3, t_0 + 3]$.
- [ ] **8. Log-Transform and Sector-Median Imputation for Illiquidity:**
  - *Current issue:* Raw Amihud values contain extreme positive outliers, and missing days fell back to unscaled volatility.
  - *Fix:* Apply $Illiq_j = \ln(1 + 10^{10} \times ILLIQ)$ and impute missing values using the date-specific sector median.

---

## 4. Phase 2: Comprehensive Robustness Suite (PENDING)

Once the baseline model in Phase 1 is estimated across all four ETFs (`XME`, `XLE`, `IHE`, `XLV`), Phase 2 will execute 12 systematic robustness checks:

1. **Threshold Multiplier Sensitivity:** Re-estimate with $\theta \in \{1.0\sigma, 2.0\sigma, 2.5\sigma\}$.
2. **Event Horizon Length:** Test transmission horizon $H \in \{1, 2, 5\}$ trading days.
3. **Estimation Window Length:** Vary estimation window $W \in \{60, 90, 180, 252\}$ trading days.
4. **Estimation Gap Sensitivity:** Test pre-event gaps of $GAP \in \{2, 10\}$ trading days.
5. **Multi-Factor Risk Model:** Estimate abnormal returns using the Fama-French 3-factor model (Market, SMB, HML) instead of the single sector index.
6. **Sub-Period Analysis:** Split into Pre-COVID (2015–2019), COVID Crisis (2020–2021), and Rate-Hike Era (2022–2026).
7. **Cross-Sector Sub-Sample Regressions:** Individual regressions for `XME`, `XLE`, `IHE`, and `XLV`.
8. **Heavyweight Constituent Sample:** Restrict shocks to constituents with weight $w_i > 2\%$ and $w_i > 5\%$.
9. **Placebo Pseudo-Event Shuffling:** Permute event dates randomly across time to generate empirical $t$-statistic distributions under the sharp null hypothesis.
10. **Alternative Liquidity Metrics:** Re-estimate using share turnover and Roll (1984) effective bid-ask spread proxies.
11. **Contamination Filter Sensitivity:** Narrow and widen the overlapping shock purge window ($[t_0 - 1, t_0 + 1]$ vs $[t_0 - 5, t_0 + 5]$).
12. **Leave-One-Out Basket Residualization:** Compare sector benchmark abnormal returns against synthetic leave-one-out basket returns $AR^{(-i)}$.

---

## 5. Phase 3: Machine Learning Benchmark (PENDING)

### Objectives
- Benchmark linear econometric findings against non-linear, non-parametric tree models.
- Determine whether complex feature interactions (e.g., non-linear interactions between illiquidity, portfolio weight, and sector similarity) improve out-of-sample predictability of $AR_j$.

### Methodological Framework
1. **Model:** LightGBM Regressor (`lightgbm`).
2. **Cross-Validation Scheme:** Purged Group Time-Series Split (Walk-forward chronological validation, embargoing 5 days around event boundaries to avoid leakage).
3. **Feature Space:**
   - Shocks: $Shock_i$, $|Shock_i|$, $Neg_e$.
   - Holdings: $w_i$, $w_j$, $w_i / w_j$, constituent rank.
   - Market Microstructure: $Illiq_j$, $\Delta Illiq_j$, volume percentile, market cap proxy.
   - ETF Level: $Mispricing_k$, ETF return volatility.
   - Economic Linkages: $Similarity_{ij}$ (GICS match degree).
4. **Interpretability:**
   - TreeSHAP (`shap.TreeExplainer`) to compute global feature importance and feature interaction plots.
   - Partial Dependence Plots (PDP) to visualize non-linearities in the liquidity and weight transmission channels.

---

## 6. Phases 4–6: Advanced Extensions (PENDING)

- **Phase 4: Neural Network Benchmark (Conditional):** If LightGBM demonstrates significant non-linear predictive lift over OLS ($R^2$ gain $> 5\%$), deploy a Multi-Layer Perceptron (MLP) with entity embeddings for constituent stocks and industry sectors.
- **Phase 5: Graph Neural Network (GNN) Spillover Modeling (Conditional):** Model the ETF constituent universe as a time-varying dynamic graph where nodes represent individual stocks (attributed with returns and liquidity) and edges represent ETF co-holdings and GICS relatedness. Use Graph Convolutional Networks (GCN) or Graph Attention Networks (GAT) to model multi-hop shock contagion.
- **Phase 6: Ingestion of ETF Creation/Redemption Flow Data:** Ingest daily shares outstanding for the ETFs to construct true dollar flow variables:
  $$Flow_{k,t} = (Shares_{k,t} - Shares_{k,t-1}) \times P_{k,t}$$
  allowing direct empirical testing of Hypothesis **H4** ($\beta_3$ flow channel).


