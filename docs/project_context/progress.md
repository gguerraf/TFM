# Project Progress

Last updated: 2026-09-22

## Current Status

The project has moved from a preliminary spillover prototype to a validated econometric and machine-learning pipeline for intra-ETF shock transmission. The repository is organized into `src/pipeline`, `src/models`, `src/analysis`, `src/utils`, and `docs/project_context`.

The current empirical universe excludes `SPY` from the main model because it is a broad market ETF and creates a self-benchmarking problem. The main analysis uses `XME`, `XLE`, `IHE`, and `XLV`. `XLE` uses `IXC` as its benchmark instead of a broad market index.

The latest model extension adds three variables with stronger academic motivation: receiver stock weight in the ETF (`w_j`), 60-day pre-event return correlation between the shocked and receiver stock (`Corr_ij_60d`), and ETF concentration (`HHI_etf_t`). The purpose of this extension is not to erase the previous specification, but to compare how the empirical results change when ETF basket structure and market connectedness are explicitly included.

## Completed Work

### Data Pipeline

- Parsed raw ETF holdings JSON files into cleaned holdings CSVs.
- Filtered non-equity holdings and handled duplicate ticker records.
- Built a deduplicated company universe.
- Downloaded and repaired constituent price histories.
- Added ticker mapping logic for renamed or acquired companies.
- Downloaded benchmarks, including `IXC` for `XLE`.
- Downloaded GICS sector and industry metadata.
- Downloaded daily volume and computed Amihud illiquidity.
- Computed cleaned daily log returns.

### Econometric Model

The improved model is implemented in `src/models/21_estimate_improved_model.py`.

Main corrections implemented before the latest extension:

1. Event-specific shock thresholds based only on the pre-event estimation window residual variance.
2. `SPY` excluded from the main analysis.
3. `XLE` benchmark changed to `IXC`.
4. Receiver-stock and year-quarter fixed effects included.
5. Main effects added for liquidity, mispricing, similarity and negative-shock status.
6. Receiver contamination filter added for overlapping events.
7. Amihud illiquidity log-transformed.
8. Two-way clustered standard errors fixed using `AbsorbingLS`, clustered by `event_id` and `stock_j`.

Latest extension:

1. Added `w_j`, the receiver stock's weight in the ETF.
2. Added `Corr_ij_60d`, the pre-event return correlation between the shocked and receiver stock.
3. Added `HHI_etf_t`, the ETF concentration index on the holding date.
4. Added the interaction channels `receiver_weight_term`, `corr_term`, and `hhi_term`.
5. Updated robustness and ML scripts so the same expanded feature set can be tested consistently.

The move to `AbsorbingLS` was necessary because explicit dummy-variable OLS with more than 743,000 observations and many fixed effects caused numerical and memory failures in `statsmodels`. Absorbing the fixed effects avoids materializing the dummy matrix and produces stable one-way and two-way clustered inference.

### Robustness Checks

Robustness checks are implemented in `src/models/22_run_robustness_checks.py` and use absorbed fixed effects.

Current checks include:

- Baseline specification.
- Positive-shock and negative-shock subsamples.
- Pre-COVID and post-COVID subsamples.
- Per-ETF regressions.
- Excluding top-weight observations.
- Trimming `AR_j` and `Shock_i` at the 1st and 99th percentiles.
- Placebo event assignment within ETF.
- Randomized receiver membership.
- Shuffled receiver abnormal returns.
- No year-quarter fixed effects.

Important correction: the old `R16_Placebo_dates` check changed dates but did not recompute outcomes or regressors, so it produced results identical to the baseline. It has been replaced by `R16_Placebo_event_assignment`, which randomly reassigns shock variables within each ETF and recomputes all interaction terms.

### Machine Learning and Neural Network

The ML benchmark is implemented in `src/models/23_run_ml_baselines.py`.

The script compares:

- Simple OLS without fixed effects.
- LightGBM.
- Feed-forward neural network with fixed `tanh` activation.

The neural network follows the supervisor's suggestion: the activation function is fixed to `tanh`, and the modest grid varies mainly the number of hidden layers and the number of nodes per layer. Learning rate, dropout and batch size are fixed to keep the grid small and interpretable.

Current best neural network configuration:

- Hidden layers: 3
- Nodes per layer: 16
- Activation: `tanh`
- Learning rate: 0.01
- Dropout: 0.0
- Batch size: 2048

## Model Version Comparison

### Previous Improved Specification

This is the specification before adding `w_j`, `Corr_ij_60d`, and `HHI_etf_t`.

Panel and regression sample:

- Panel rows: 758,013.
- Regression rows: 744,666.
- Shock events: 41,393.
- Receiver stocks: 208.
- Year-quarter periods: 45.

Two-way clustered results:

| Term | Coef. | SE | p-value | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `b1_term` | -0.4638 | 0.1454 | 0.0014 | Direct weighted shock propagation is negative and significant. |
| `b2_term` | 0.3326 | 0.0540 | <0.001 | Liquidity channel is positive and significant. |
| `b4_term` | -2.7383 | 2.1661 | 0.2062 | Mispricing/arbitrage channel is not robustly significant. |
| `b5_term` | 0.0187 | 0.0029 | <0.001 | Informational similarity channel is positive and significant. |
| `asym_term` | -0.1410 | 0.1001 | 0.1588 | Negative-shock asymmetry is not robustly significant. |

ML results:

| Model | Test R2 | Test MAE | Directional accuracy |
| --- | ---: | ---: | ---: |
| OLS baseline | 0.0002 | 0.0221 | 0.5230 |
| LightGBM | 0.0163 | 0.0220 | 0.5407 |
| Tanh neural network | 0.0146 | 0.0220 | 0.5377 |

### Expanded Specification

This is the latest specification after adding receiver weight, pre-event return correlation and ETF concentration.

Panel and regression sample:

- Panel rows: 757,368.
- Regression rows: 743,033.
- Shock events: 41,349.
- Receiver stocks: 206.
- Year-quarter periods: 45.

Two-way clustered results:

| Term | Coef. | SE | p-value | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `b1_term` | -1.2020 | 0.2161 | <0.001 | Direct weighted shock propagation remains negative and significant. |
| `b2_term` | 0.4854 | 0.0533 | <0.001 | Liquidity channel remains positive and significant. |
| `b4_term` | -3.1222 | 2.1493 | 0.1463 | Mispricing/arbitrage channel remains weaker under two-way clustering. |
| `b5_term` | 0.0205 | 0.0028 | <0.001 | Informational similarity channel remains positive and significant. |
| `receiver_weight_term` | 7.5488 | 3.6623 | 0.0393 | Receiver ETF weight adds a positive transmission channel. |
| `corr_term` | 2.6345 | 0.2258 | <0.001 | Pre-event return comovement strongly amplifies transmission. |
| `hhi_term` | -6.1627 | 1.9303 | 0.0014 | ETF concentration changes the transmission mechanism and enters with a negative sign. |
| `asym_term` | -0.0996 | 0.1052 | 0.3436 | Negative-shock asymmetry remains not robustly significant. |

ML results:

| Model | Test R2 | Test MAE | Directional accuracy |
| --- | ---: | ---: | ---: |
| OLS baseline | 0.0004 | 0.0221 | 0.5274 |
| LightGBM | -0.0122 | 0.0222 | 0.5449 |
| Tanh neural network | 0.0146 | 0.0220 | 0.5473 |

## Effect of the New Variables

Adding the new variables changes the econometric interpretation in a useful way. The direct weighted shock term becomes more negative, while the liquidity and informational channels remain significant. This suggests that the previous model was capturing part of the basket-structure and connectedness effects inside the original coefficients.

The most important new result is `corr_term`: pre-event return comovement is positive and strongly significant. This supports the idea that intra-ETF transmission is stronger among stocks that were already connected in the market before the shock.

The receiver-weight channel is also positive and significant, which means that the reaction is stronger when the receiving stock has a larger position in the ETF. The concentration channel is significant with a negative sign, suggesting that ETF structure matters, but its interpretation should be more cautious.

The predictive results are more mixed. The expanded features do not improve LightGBM's out-of-sample R2, although directional accuracy remains higher than OLS. The neural network remains similar in R2 and improves slightly in directional accuracy. This means the new variables are valuable for econometric interpretation, but not enough to justify a GNN as a core model.

## Remaining Work

Short-term priorities:

1. Decide whether all three new channels should remain in the final thesis specification or whether one should be presented as a robustness extension.
2. Use `results_interpretation.md` as the basis for the thesis results section.
3. Consider whether LightGBM needs a simpler feature set because the latest expanded feature set worsens test R2.
4. Keep `holdings/`, generated results and local machine-specific files out of GitHub.

Potential future extensions:

- Add ETF flow or creation/redemption proxy if shares outstanding data can be obtained reliably.
- Add a market-state or volatility-regime interaction.
- Consider a GNN only as a future extension, not as a core model, because the current ML evidence is still limited.
