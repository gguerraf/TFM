# Econometric Methodology & Empirical Design

> **Document:** Econometric Specification & Estimation Protocol
> **Master's Thesis:** *Structural Dynamics and Contagion Mechanisms of Intra-ETF Shock Transmission*
> **Location:** [`src/methodology.md`](src/methodology.md)

---

## 1. Introduction & Econometric Framework

The objective of this empirical design is to isolate firm-specific idiosyncratic shocks occurring in major constituent stocks of Exchange-Traded Funds (ETFs) and quantify their directional transmission to co-constituent stocks of the same ETF.

To overcome the well-known limitations of corporate event studies (e.g., selection bias, news release lags, announcement clustering, and incomplete databases), this thesis employs an **endogenous event identification methodology** grounded in standard asset pricing and event-study theory (MacKinlay, 1997; Ben-David, Franzoni, & Moussawi, 2012, 2018).

---

## 2. Market Model Specification

For every constituent stock $i$ and candidate event date $t_0$, abnormal returns are extracted by purging systematic risk exposures using an ordinary least squares (OLS) market model.

### 2.1 Model Equation
$$R_{i,t} = \alpha_i + \beta_i R_{m(k),t} + \epsilon_{i,t}, \quad t \in [t_0 - W - GAP, \; t_0 - GAP - 1]$$

Where:
- $R_{i,t} = \ln(P_{i,t} / P_{i,t-1})$ is the daily logarithmic return of constituent stock $i$.
- $R_{m(k),t} = \ln(P_{m(k),t} / P_{m(k),t-1})$ is the daily return of the designated external sector benchmark $m(k)$ corresponding to ETF $k$.
- $\alpha_i, \beta_i$ are OLS parameters estimated over the estimation window.
- $\epsilon_{i,t}$ is the zero-mean idiosyncratic error term representing firm-specific return innovations.

### 2.2 Estimation Windows & Timing Structure
The timeline around an event date $t_0$ is divided into three distinct segments:

```
[--------- Estimation Window: W = 120 days ---------] [--- GAP: 5 days ---] [--- Event Window: H = 3 days ---]
t0 - 125                                              t0 - 6               t0 - 1    t0                      t0 + 3
```

1. **Estimation Window ($W = 120$ trading days):** Spans $[t_0 - 125, t_0 - 6]$. This provides sufficient statistical power (degrees of freedom $df = 118$) to accurately estimate beta while remaining responsive to time-varying systematic risk.
2. **Gap Window ($GAP = 5$ trading days):** Spans $[t_0 - 5, t_0 - 1]$. A 5-day buffer is inserted between the estimation window and event date $t_0$ to ensure that pre-announcement drift, insider trading, or rumor-driven information leakage does not bias the parameter estimates $(\hat{\alpha}_i, \hat{\beta}_i)$.
3. **Event Window ($H = 3$ days, length 4 trading days):** Spans $[t_0, t_0 + 3]$. Captures the cumulative multi-day transmission and mechanical basket trading response.

### 2.3 Cumulative Abnormal Return (CAR)
The abnormal return on any day $t \in [t_0, t_0 + 3]$ is defined as:
$$AR_{i,t} = R_{i,t} - \hat{\alpha}_i - \hat{\beta}_i R_{m(k),t}$$

The Cumulative Abnormal Return over the full event horizon is:
$$CAR_i(t_0) = \sum_{t=t_0}^{t_0 + 3} AR_{i,t}$$

---

## 3. Shock Identification Protocol

### 3.1 Endogenous Event Flagging
An event at $(stock_i, t_0)$ is identified as a firm-specific shock if its Cumulative Abnormal Return exceeds an event-specific dynamic volatility threshold:

$$|CAR_i(t_0)| > \theta \times \sqrt{H + 1} \times \hat{\sigma}_{\epsilon, i}(t_0)$$

Where:
- $\theta = 1.5$ is the standard deviation scaling multiplier.
- $H + 1 = 4$ is the number of trading days in the event window $[t_0, t_0 + 3]$. The factor $\sqrt{H + 1} = \sqrt{4} = 2$ accounts for the variance accumulation of independent daily abnormal returns:
  $$\text{Var}(CAR_{i,[t_0, t_0+H]}) = \sum_{t=t_0}^{t_0+H} \text{Var}(AR_{i,t}) \approx (H + 1) \sigma_{\epsilon, i}^2$$
- $\hat{\sigma}_{\epsilon, i}(t_0)$ is the standard deviation of residuals from the **120-day pre-event estimation window**:
  $$\hat{\sigma}_{\epsilon, i}(t_0) = \sqrt{\frac{1}{W - 2} \sum_{t=t_0 - 125}^{t_0 - 6} \hat{\epsilon}_{i,t}^2}$$

> [!IMPORTANT]
> **Elimination of Lookahead Bias:**
> In preliminary Phase 0 modeling (`20_estimate_baseline_model.py`), the shock cutoff used the full 11-year sample standard deviation of CARs. This introduced lookahead bias (conditioning past shock classification on future market volatility). Computing $\hat{\sigma}_{\epsilon, i}(t_0)$ dynamically from the historical estimation window strictly confines information to the pre-event information set $\mathcal{F}_{t_0 - 6}$.

### 3.2 Shock Characteristics
Each detected shock is characterized by:
- **Magnitude:** $Shock_{i,e} = CAR_i(t_0)$
- **Sign Indicator:**
  $$Neg_e = \begin{cases} 1 & \text{if } CAR_i(t_0) < 0 \quad (\text{negative shock / adverse event}) \\ 0 & \text{if } CAR_i(t_0) > 0 \quad (\text{positive shock / favorable event}) \end{cases}$$

---

## 4. Benchmark Selection Rationale

Using the ETF itself as the benchmark index creates severe endogeneity and collinearity: constituent stocks are mechanically contained within the ETF basket, deflating abnormal returns for high-weight stocks. Therefore, external benchmarks are assigned:

| ETF Ticker | Mandate | Assigned Sector Benchmark | Benchmark Description & Rationale |
| :--- | :--- | :--- | :--- |
| **XME** | Metals & Mining | [`XLB`](holdings/processed/benchmarks.csv) | SPDR Materials ETF. XME is a specialized metals/mining sub-industry within the materials sector. XLB tracks broad materials since 1998 with zero stock-level overlap bias. |
| **XLE** | Energy Sector | [`IXC`](holdings/processed/benchmarks.csv) | iShares Global Energy ETF. Replaces `^GSPC`. Broad S&P 500 diluted energy-specific macro factors (crude oil fluctuations, OPEC decisions). IXC provides global energy factor control while avoiding single-stock US domination. |
| **IHE** | Pharmaceuticals | [`XLV`](holdings/processed/benchmarks.csv) | SPDR Health Care ETF. IHE is a pure-play pharma fund. XLV encompasses pharmaceuticals, medical devices, and healthcare providers, capturing sector-wide trends. |
| **XLV** | Health Care | [`^SP500-35`](holdings/processed/benchmarks.csv) | Official S&P 500 Health Care Index. Clean, broad sector benchmark directly tracking the universe without replication drift. |
| **SPY** | S&P 500 Broad Market | `^GSPC` | **Excluded from primary regressions.** SPY constituents comprise the S&P 500; benchmarking against `^GSPC` creates an inescapable self-benchmarking feedback loop. |

---

### 4.1 Benchmark Risk to Test Next

The current external benchmarks remain the main reference specification. They are economically motivated and avoid using the ETF itself as the benchmark in the primary model. However, they do not fully eliminate overlap risk. Some external benchmarks can still contain the shocked stock or the receiver stock, especially in large sector funds such as health care and energy.

For this reason, the next methodological extension should test synthetic leave-one-out and leave-two-out ETF benchmarks. The key idea is:

- For shock identification of stock i, use ETF(-i), so the shocked stock is not included in its own benchmark.
- For receiver abnormal return of stock j, use ETF(-i,-j) when possible, so neither the shocked stock nor the receiver stock is included in the receiver benchmark.
- If ETF(-i,-j) becomes too noisy in smaller ETFs, report ETF(-j) or ETF(-i) as fallback robustness checks.

This check is important before finalizing the interpretation of the negative b1_term. If b1_term changes sign or becomes much smaller under leave-two-out benchmarks, part of the previous result may come from benchmark absorption rather than true economic reversal.

---

## 5. Panel Construction & Filtering

For each identified shock event $e = (stock_i, t_0)$ occurring in ETF $k$, the observation unit is the pair $(e, stock_j)$ for all co-constituents $j \neq i$ present in ETF $k$ on date $t_0$.

### 5.1 Receiver Abnormal Return ($AR_j$)
The dependent variable $AR_{j,e}$ is the cumulative abnormal return of receiver stock $j$ over the identical window $[t_0, t_0 + 3]$, estimated using stock $j$'s **own market model** against the sector benchmark $m(k)$ over its own pre-event estimation window $[t_0 - 125, t_0 - 6]$:
$$AR_{j,e} = \sum_{t=t_0}^{t_0 + 3} \left( R_{j,t} - \hat{\alpha}_j - \hat{\beta}_j R_{m(k),t} \right)$$

### 5.2 Overlapping Event Filtering (Contamination Purge)
To prevent confounding the transmission effect with independent firm-specific news occurring at receiver stock $j$:
- **Exclusion Rule:** If receiver stock $j$ experienced an idiosyncratic shock of its own within the window $[t_0 - 3, t_0 + 3]$ ($|CAR_j(t)| > 1.5 \sqrt{4}\hat{\sigma}_{\epsilon,j}$), the observation $(e, j)$ is removed from the estimation panel.
- This ensures that $AR_{j,e}$ reflects passive spillover rather than concurrent firm-level earnings announcements or FDA approvals.

---

## 6. Econometric Model Specification

The primary empirical model tests the transmission hypotheses within a two-way fixed effects panel regression framework. The project now keeps two specifications documented: the previous improved specification and the expanded specification. This makes it possible to compare how the results change when receiver weight, pre-event return comovement and ETF concentration are added.

### 6.1 Previous Improved Specification

The previous improved specification included the original direct, liquidity, mispricing, informational and asymmetry channels:

$\begin{aligned}
AR_{j,e} = \alpha_j &+ \beta_1 (Shock_{i,e} \times w_{i,t_0}) \\
&+ \beta_2 (Shock_{i,e} \times w_{i,t_0} \times Illiq_{j,t_0}) \\
&+ \beta_4 (Shock_{i,e} \times w_{i,t_0} \times Mispricing_{k,t_0}) \\
&+ \beta_5 (Shock_{i,e} \times Similarity_{ij}) \\
&+ \delta (Neg_e \times Shock_{i,e} \times w_{i,t_0}) \\
&+ \gamma_1 Illiq_{j,t_0} + \gamma_2 Mispricing_{k,t_0} + \gamma_3 Similarity_{ij} + \gamma_4 Neg_e \\
&+ FE_j + FE_{yq} + \varepsilon_{j,e}
\end{aligned}$

### 6.2 Expanded Specification

The expanded specification keeps the previous channels and adds receiver weight, pre-event return comovement and ETF concentration:

$\begin{aligned}
AR_{j,e} = \alpha_j &+ \beta_1 (Shock_{i,e} \times w_{i,t_0}) \\
&+ \beta_2 (Shock_{i,e} \times w_{i,t_0} \times Illiq_{j,t_0}) \\
&+ \beta_4 (Shock_{i,e} \times w_{i,t_0} \times Mispricing_{k,t_0}) \\
&+ \beta_5 (Shock_{i,e} \times Similarity_{ij}) \\
&+ \beta_6 (Shock_{i,e} \times w_{i,t_0} \times w_{j,t_0}) \\
&+ \beta_7 (Shock_{i,e} \times w_{i,t_0} \times Corr_{ij,60d}) \\
&+ \beta_8 (Shock_{i,e} \times w_{i,t_0} \times HHI_{k,t_0}) \\
&+ \delta (Neg_e \times Shock_{i,e} \times w_{i,t_0}) \\
&+ \gamma_1 Illiq_{j,t_0} + \gamma_2 Mispricing_{k,t_0} + \gamma_3 Similarity_{ij} \\
&+ \gamma_4 w_{j,t_0} + \gamma_5 Corr_{ij,60d} + \gamma_6 HHI_{k,t_0} + \gamma_7 Neg_e \\
&+ FE_j + FE_{yq} + \varepsilon_{j,e}
\end{aligned}$

### 6.3 Key Econometric Enhancements over the Preliminary Model

1. **Inclusion of Main Effects:** Main effects are included for all moderating variables used in interactions. This avoids interpreting interaction terms as if they were isolated from level differences.
2. **Receiver Weight Channel:** `w_j` captures whether the receiver stock's own importance in the ETF affects the strength of transmission.
3. **Pre-Event Return Comovement Channel:** `Corr_ij_60d` captures recent market connectedness between the shocked and receiver stock using only information before the event date.
4. **ETF Concentration Channel:** `HHI_etf_t` captures whether the structure of the ETF basket changes the transmission mechanism.
5. **Two-Way Fixed Effects:** Receiver-stock fixed effects control for time-invariant firm characteristics, while year-quarter fixed effects control for macro-financial regimes.
6. **Two-Way Clustered Standard Errors:** Standard errors are clustered by event and receiver stock.

---

## 7. Operationalization of Variables

| Variable | Mathematical Definition | Data Source / Processing | Economic Role & Interpretation |
| :--- | :--- | :--- | :--- |
| **$AR_j$** | $\sum_{t=t_0}^{t_0+3} (R_{j,t} - \hat{\alpha}_j - \hat{\beta}_j R_{m,t})$ | `returns_clean.csv`, `benchmarks.csv` | **Dependent Variable:** Cumulative abnormal return of receiver stock $j$. |
| **$Shock_i$** | $\sum_{t=t_0}^{t_0+3} (R_{i,t} - \hat{\alpha}_i - \hat{\beta}_i R_{m,t})$ | Market model residuals | **Shock Impulse:** Cumulative abnormal return of shocked stock $i$. |
| **$w_{i,t_0}$** | Weight of shocked stock $i$ in ETF $k$ before the event | `{etf}_holdings.csv` | **Origin Weight:** Measures the ETF importance of the shocked stock. |
| **$w_{j,t_0}$** | Weight of receiver stock $j$ in ETF $k$ before the event | `{etf}_holdings.csv` | **Receiver Weight:** Tests whether larger receiver positions react more strongly. |
| **$Illiq_{j,t_0}$** | $\ln(1 + 10^{10} \times Amihud_{j,t_0})$ | `amihud.csv` | **Liquidity Friction:** Higher values mean less liquid receiver stocks. |
| **$Mispricing_{k,t_0}$** | Five-day ETF return minus benchmark return before the event | ETF and benchmark returns | **Arbitrage Incentive:** Proxy for pre-event tracking discrepancy. |
| **$Similarity_{ij}$** | 1 same industry, 0.5 same sector, 0 different sector | `gics_data.csv` | **Informational Link:** Static economic relatedness. |
| **$Corr_{ij,60d}$** | Pearson correlation of stock $i$ and stock $j$ returns over the 60 trading days before $t_0$ | `returns_clean.csv` | **Market Connectedness:** Dynamic pre-event co-movement. |
| **$HHI_{k,t_0}$** | $\sum_m s_{m,k,t_0}^2$, where $s_m$ are normalized ETF weights | `{etf}_holdings.csv` | **ETF Concentration:** Measures whether the ETF is concentrated in a few names. |
| **$Neg_e$** | $\mathbb{I}_{\{Shock_i < 0\}}$ | Derived from `Shock_i` sign | **Asymmetry Dummy:** Flags negative shocks. |

---

## 8. Hypothesis Testing & Theoretical Interpretation

| Parameter | Channel | Null Hypothesis | Economic Interpretation of Rejection |
| :--- | :--- | :--- | :--- |
| $\beta_1$ | Baseline propagation | $\beta_1 = 0$ | Tests whether shocks propagate through direct weighted exposure. |
| $\beta_2$ | Liquidity channel | $\beta_2 = 0$ | Tests whether illiquid receivers react more strongly to shocks. |
| $\beta_4$ | Arbitrage / mispricing channel | $\beta_4 = 0$ | Tests whether pre-event ETF benchmark deviation changes transmission. |
| $\beta_5$ | Informational similarity | $\beta_5 = 0$ | Tests whether economically similar firms receive stronger spillovers. |
| $\beta_6$ | Receiver weight channel | $\beta_6 = 0$ | Tests whether larger receiver positions in the ETF amplify transmission. |
| $\beta_7$ | Return comovement channel | $\beta_7 = 0$ | Tests whether stocks that moved together before the event transmit shocks more strongly. |
| $\beta_8$ | ETF concentration channel | $\beta_8 = 0$ | Tests whether concentrated ETF structures change shock transmission. |
| $\delta$ | Asymmetry | $\delta = 0$ | Tests whether negative shocks transmit differently from positive shocks. |

---

## 9. Comprehensive Robustness Check Protocol

To establish the validity of the empirical results against model assumptions, the following 12 robustness checks are prioritized:

1. **Alternative Shock Cutoffs ($\theta \in \{1.0\sigma, 2.0\sigma, 2.5\sigma\}$):**
   Evaluates whether spillover coefficients are sensitive to the threshold defining a discrete shock event versus continuous noise.
2. **Alternative Event Horizon Lengths ($H \in \{1, 2, 5\}$ trading days):**
   Tests whether transmission occurs immediately ($H=1$, $[t_0, t_0+1]$) or requires time to diffuse ($H=5$, $[t_0, t_0+5]$).
3. **Alternative Estimation Window Lengths ($W \in \{60, 90, 180, 252\}$ trading days):**
   Verifies that market model betas and residual standard errors are invariant to the calibration horizon length.
4. **Alternative Estimation Gap ($GAP \in \{2, 10\}$ trading days):**
   Ensures that results are not sensitive to the 5-day pre-event information exclusion window.
5. **Multi-Factor Systematic Purging (Fama-French 3-Factor Model):**
   Replaces the single-index market model with FMB, HML, and Market factors to verify that abnormal returns are truly idiosyncratic and not latent size or value factor co-loadings.
6. **Subsample Temporal Stability (Pre-COVID, COVID Shock, Post-COVID Rate Hike):**
   Splits sample into 2015–2019 (low rate regime), 2020–2021 (COVID volatility shock), and 2022–2026 (Fed quantitative tightening) to test structural stability.
7. **ETF-by-ETF Sector Heterogeneity:**
   Estimates the regression separately for `XME`, `XLE`, `IHE`, and `XLV` to examine cross-sector differences in liquidity and concentration.
8. **Top Constituent Concentration Restriction ($w_i > 2\%$ and $w_i > 5\%$):**
   Restricts the shock sample to dominant constituents (e.g., XOM/CVX in XLE, LLY/JNJ in IHE) where basket arbitrage pressure is theoretically greatest.
9. **Placebo / Pseudo-Event Date Shuffling:**
   Randomly permutes event dates $t_0$ across the calendar to confirm that empirical $t$-statistics on $\beta_1..\beta_5$ do not arise spuriously from background market noise.
10. **Alternative Liquidity Proxy (Turnover & Roll Spread):**
    Replaces Amihud illiquidity with share turnover ($\text{Volume} / \text{Shares Outstanding}$) and Roll (1984) effective bid-ask spread proxy to confirm the liquidity channel.
11. **Contamination Window Sensitivity ($[t_0 - 1, t_0 + 1]$ vs $[t_0 - 5, t_0 + 5]$):**
    Varies the strictness of the overlapping event exclusion filter.
12. **Leave-One-Out / Leave-Two-Out Synthetic ETF Benchmark:**
    Re-estimates the abnormal returns using synthetic ETF benchmarks built from the ETF constituents themselves, excluding the stock or stocks that could mechanically contaminate the benchmark.

    Planned implementation:

    - For shock identification of stock i: use ETF(-i).
    - For receiver abnormal return of stock j: use ETF(-i,-j) when possible.
    - Compare against ETF(-j), ETF(-i), and the current external benchmark if the leave-two-out version is unstable.

    Academic purpose:

    - Tests whether the negative b1_term is robust to benchmark construction.
    - Tests whether the current external benchmarks mechanically absorb part of the shock or receiver movement.
    - Gives a clean diagnostic before moving to more complex models.

## 10. Planned Model Extensions After Benchmark Robustness

The agreed order for future work is:

1. Leave-one-out / leave-two-out benchmark robustness.
2. Local Projections following Jorda (2005), with horizons h=0,...,10, to study dynamic transmission.
3. Fama-French / Carhart factor robustness, to check dependence on the expected-return model.
4. Pooled ETF-specific interaction tests, to formally test heterogeneity across funds.
5. Quantile regression as a secondary extension for tail spillovers.

Matched-control / DiD designs, Double/Debiased ML and GNNs are not the immediate priority. They may be useful later, but they do not solve the main open issue: whether abnormal returns are sensitive to benchmark construction.


## 11. Implemented Extensions and How to Interpret Them

The planned benchmark and model extensions have now been implemented in scripts `24` to `28`.

The benchmark robustness script compares the main external-benchmark design with synthetic ETF benchmarks. For shock identification, it uses `ETF(-i)`. For receiver abnormal returns, it estimates both `ETF(-i,-j)` and `ETF(-j)` variants. The key result is that the direct coefficient remains negative and statistically significant under these synthetic benchmarks.

The Local Projections script estimates the model separately for horizons `h=0,...,10`. This gives an impulse-response-style view of the transmission path and shows that the negative direct term remains visible beyond the original event window.

The factor-robustness script uses Fama-French 5 factors plus momentum. This check shows that the direct channel is sensitive to abnormal-return construction, while the return-comovement channel remains robust.

The pooled ETF-interaction script formally tests whether channels differ by ETF, instead of relying only on separate ETF regressions.

The quantile-regression script is a secondary diagnostic. It uses ETF and year-quarter controls, but not the full receiver fixed effects, so it should be interpreted as descriptive evidence about tails rather than as the main causal/econometric result.
