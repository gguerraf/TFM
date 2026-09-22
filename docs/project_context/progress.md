# Project Progress

Last updated: 2026-09-22

## Current Status

The project has moved from a preliminary spillover prototype to a validated econometric and machine-learning pipeline for intra-ETF shock transmission. The repository has also been reorganized into a clearer GitHub-style structure with `src/pipeline`, `src/models`, `src/analysis`, `src/utils`, and `docs/project_context`.

The current empirical universe excludes `SPY` from the main model because it is a broad market ETF and creates a self-benchmarking problem. The main analysis uses `XME`, `XLE`, `IHE`, and `XLV`. `XLE` now uses `IXC` as its benchmark instead of the broad market index.

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

Main corrections now implemented:

1. Event-specific shock thresholds based only on the pre-event estimation window residual variance.
2. `SPY` excluded from the main analysis.
3. `XLE` benchmark changed to `IXC`.
4. Receiver-stock and year-quarter fixed effects included.
5. Main effects added for liquidity, mispricing, similarity and negative-shock status.
6. Receiver contamination filter added for overlapping events.
7. Amihud illiquidity log-transformed.
8. Two-way clustered standard errors fixed using `AbsorbingLS`, clustered by `event_id` and `stock_j`.

The move to `AbsorbingLS` was necessary because explicit dummy-variable OLS with more than 744,000 observations and many fixed effects caused numerical and memory failures in `statsmodels` (`init_gesdd`, QR and SVD failures). Absorbing the fixed effects avoids materializing the dummy matrix and produces stable one-way and two-way clustered inference.

### Robustness Checks

Robustness checks are implemented in `src/models/22_run_robustness_checks.py` and now also use absorbed fixed effects.

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

Important correction: the previous `R16_Placebo_dates` check changed dates but did not recompute outcomes or regressors, so it produced results identical to the baseline. It has been replaced by `R16_Placebo_event_assignment`, which randomly reassigns shock variables within each ETF and recomputes the interaction terms. This breaks the event-outcome link while preserving ETF-level structure.

### Machine Learning and Neural Network

The ML benchmark is implemented in `src/models/23_run_ml_baselines.py`.

The script compares:

- Simple OLS without fixed effects.
- LightGBM.
- Feed-forward neural network with fixed `tanh` activation.

The neural network now follows the supervisor's suggestion more closely: the activation function is fixed to `tanh`, and the modest grid varies mainly the number of hidden layers and the number of nodes per layer. Learning rate, dropout and batch size are fixed to keep the grid small and interpretable.

Current best neural network configuration:

- Hidden layers: 3
- Nodes per layer: 16
- Activation: `tanh`
- Learning rate: 0.01
- Dropout: 0.0
- Batch size: 2048

Compared with the earlier neural network run, the adjusted grid improved the neural network:

| Neural network version | Test R2 | Test MAE | Directional accuracy |
| --- | ---: | ---: | ---: |
| Previous NN grid | 0.0121 | 0.0220 | 0.5311 |
| Current supervisor-aligned NN grid | 0.0146 | 0.0220 | 0.5377 |

LightGBM remains slightly better than the neural network:

| Model | Test R2 | Test MAE | Directional accuracy |
| --- | ---: | ---: | ---: |
| OLS baseline | 0.0002 | 0.0221 | 0.5230 |
| LightGBM | 0.0163 | 0.0220 | 0.5407 |
| Tanh neural network | 0.0146 | 0.0220 | 0.5377 |

## Current Main Results

The final improved panel contains 758,013 observations. The main regression uses 744,666 observations after dropping missing model variables.

One-way clustered results are similar to the previous OLS results, but the final interpretation should prioritize two-way clustered standard errors.

Two-way clustered results:

| Term | Coef. | SE | p-value | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `b1_term` | -0.4638 | 0.1454 | 0.0014 | Direct weighted shock propagation is negative and significant. |
| `b2_term` | 0.3326 | 0.0540 | <0.001 | Liquidity channel is positive and significant. |
| `b4_term` | -2.7383 | 2.1661 | 0.2062 | Mispricing/arbitrage channel is not robustly significant. |
| `b5_term` | 0.0187 | 0.0029 | <0.001 | Informational spillover channel is positive and significant. |
| `asym_term` | -0.1410 | 0.1001 | 0.1588 | Negative-shock asymmetry is not robustly significant. |

Main conclusion: the strongest robust channels are the liquidity channel and the informational similarity channel. The direct weighted shock term is negative and significant, which suggests the average pooled response is not a simple same-direction contagion effect. Arbitrage/mispricing and negative-shock asymmetry are weaker under two-way clustered inference.

## Remaining Work

Short-term priorities:

1. Decide whether to commit and push the current local changes.
2. Use the updated `results_interpretation.md` as the basis for the thesis results section.
3. Decide whether to add one additional econometric variable, but only after discussing whether the new variable has strong academic value and available data.
4. Keep `holdings/`, generated results and local machine-specific files out of GitHub.

Potential future extensions:

- Add ETF flow or creation/redemption proxy if shares outstanding data can be obtained reliably.
- Add receiver stock portfolio weight `w_j` or weight-rank variables.
- Add a market-state or volatility-regime interaction.
- Consider a GNN only as a future extension, not as a core model, because LightGBM and the MLP only show limited predictive gains over OLS.
