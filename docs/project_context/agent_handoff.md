# AI Handoff Notes

Last updated: 2026-09-22

This document replaces the removed root-level `AGENTS.md`. The root `AGENTS.md` is intentionally ignored and not pushed to GitHub. Use this file as the safe, repository-tracked handoff for future AI agents.

## What This Project Does

The thesis studies intra-ETF shock transmission. It asks whether abnormal return shocks to one ETF constituent transmit to other constituents in the same ETF, and through which mechanisms: direct weighted exposure, liquidity, ETF mispricing/arbitrage, informational similarity, receiver ETF weight, pre-event return comovement and ETF concentration.

## Current Script Layout

Run scripts from the repository root.

Pipeline:

1. `src/pipeline/00_validate_holdings.py`
2. `src/pipeline/00_validate_duplicate_holdings.py`
3. `src/pipeline/01_load_holdings.py`
4. `src/utils/extract_company_universe.py`
5. `src/utils/deduplicate_company_universe.py`
6. `src/pipeline/02_download_prices.py`
7. `src/pipeline/03_check_missing_downloads.py`
8. `src/pipeline/04_retry_missing_tickers.py`
9. `src/pipeline/05_apply_ticker_mappings.py`
10. `src/pipeline/06_download_benchmarks.py`
11. `src/pipeline/07_download_energy_benchmark.py`
12. `src/pipeline/08_download_gics.py`
13. `src/pipeline/09_add_manual_gics_labels.py`
14. `src/pipeline/10_download_volume.py`
15. `src/pipeline/11_compute_returns.py`

Models:

1. `src/models/20_estimate_baseline_model.py`
2. `src/models/21_estimate_improved_model.py`
3. `src/models/22_run_robustness_checks.py`
4. `src/models/23_run_ml_baselines.py`

Analysis:

1. `src/analysis/40_exploratory_analysis.py`
2. `src/analysis/41_raw_json_analysis.py`
3. `src/analysis/42_integrity_checks.py`

## Important Data Policy

Do not commit raw or processed holdings data. The local `holdings/` directory is ignored by Git and must remain local. Generated outputs under `holdings/results/` and `holdings/figures/` are also local outputs, not repository files.

Before any push, check:

- No `holdings/` files staged.
- No ZIPs staged.
- No local machine-specific paths staged.
- No credentials or API keys staged.
- No root `AGENTS.md` staged.

## Main Model Details

The current preferred model is `src/models/21_estimate_improved_model.py`.

Key decisions:

- Exclude `SPY` from the main analysis.
- Use `XME`, `XLE`, `IHE`, and `XLV`.
- Use `IXC` as the benchmark for `XLE`.
- Detect shocks using event-specific thresholds based on the pre-event estimation window.
- Build a receiver-event panel `(j, e)`.
- Filter receiver observations contaminated by overlapping receiver shocks.
- Use log-transformed Amihud illiquidity.
- Add `w_j`, `Corr_ij_60d` and `HHI_etf_t` to the panel.
- Estimate `receiver_weight_term = Shock_i x w_i x w_j`.
- Estimate `corr_term = Shock_i x w_i x Corr_ij_60d`.
- Estimate `hhi_term = Shock_i x w_i x HHI_etf_t`.
- Absorb receiver-stock and year-quarter fixed effects with `AbsorbingLS`.
- Use two-way clustered standard errors by `event_id` and `stock_j` for final inference.

Do not switch back to explicit dummy-variable OLS for the final model. It caused memory and numerical failures on the full panel.

The model script checks whether the cached `holdings/results/panel_improved.csv` contains the new columns. If not, it rebuilds the panel.

## Robustness Details

`src/models/22_run_robustness_checks.py` also uses `AbsorbingLS`. This keeps robustness specifications consistent with the main model and avoids the memory failures seen with explicit dummy variables.

The corrected placebo is `R16_Placebo_event_assignment`. It randomly reassigns event-level shock variables within each ETF and recomputes all interaction terms, including the receiver-weight, correlation and concentration channels. This replaced the old placebo that only shifted dates and did not change the regression design in a meaningful way.

## Neural Network Details

`src/models/23_run_ml_baselines.py` implements the supervisor-aligned MLP:

- Fixed activation: `tanh`.
- Grid: hidden layers and nodes per layer.
- Current grid: hidden layers `[1, 2, 3]`, nodes `[16, 32, 64]`.
- Fixed learning rate: 0.01.
- Fixed dropout: 0.0.
- Fixed batch size: 2048.

Current best NN after the reproducible seed-24 run: 3 hidden layers and 32 nodes per layer.

Latest ML results after adding the new variables:

- OLS test R2 0.0004, MAE 0.0221, directional accuracy 0.5274.
- LightGBM test R2 -0.0159, MAE 0.0222, directional accuracy 0.5444.
- Tanh NN test R2 0.0171, MAE 0.0220, directional accuracy 0.5395.

A GNN is not recommended as a core thesis model unless the research question is explicitly expanded toward graph learning.

## Result Versions to Preserve

Keep both result versions in the project documentation. The expanded model should not erase the previous improved model, because the thesis may discuss how the new variables changed the estimated channels.

### Previous Improved Specification

This version was estimated before adding `w_j`, `Corr_ij_60d` and `HHI_etf_t`.

Sample:

- Panel rows: 758,013.
- Regression rows: 744,666.
- Events: 41,393.
- Receiver stocks: 208.
- Year-quarter periods: 45.

Final two-way clustered results:

- `b1_term`: coefficient -0.4638, p=0.0014.
- `b2_term`: coefficient 0.3326, p<0.001.
- `b4_term`: coefficient -2.7383, p=0.2062.
- `b5_term`: coefficient 0.0187, p<0.001.
- `asym_term`: coefficient -0.1410, p=0.1588.

ML results:

- OLS test R2 0.0002, MAE 0.0221, directional accuracy 0.5230.
- LightGBM test R2 0.0163, MAE 0.0220, directional accuracy 0.5407.
- Tanh NN test R2 0.0146, MAE 0.0220, directional accuracy 0.5377.

### Expanded Specification

This version adds `w_j`, `Corr_ij_60d`, `HHI_etf_t` and their interaction channels.

Sample:

- Panel rows: 757,368.
- Regression rows: 743,033.
- Events: 41,349.
- Receiver stocks: 206.
- Year-quarter periods: 45.

Final two-way clustered results:

- `b1_term`: coefficient -1.2020, p<0.001.
- `b2_term`: coefficient 0.4854, p<0.001.
- `b4_term`: coefficient -3.1222, p=0.1463.
- `b5_term`: coefficient 0.0205, p<0.001.
- `receiver_weight_term`: coefficient 7.5488, p=0.0393.
- `corr_term`: coefficient 2.6345, p<0.001.
- `hhi_term`: coefficient -6.1627, p=0.0014.
- `asym_term`: coefficient -0.0996, p=0.3436.

ML results:

- OLS test R2 0.0004, MAE 0.0221, directional accuracy 0.5274.
- LightGBM test R2 -0.0159, MAE 0.0222, directional accuracy 0.5444.
- Tanh NN test R2 0.0171, MAE 0.0220, directional accuracy 0.5395.

Interpretation:

- The old robust findings survive: liquidity and informational similarity remain significant.
- The direct weighted shock term becomes more negative after controlling for receiver weight, comovement and concentration.
- The strongest new channel is pre-event return comovement.
- Receiver weight also adds a positive channel.
- ETF concentration adds a significant negative channel, but its economic interpretation should be cautious.
- ML evidence remains limited, so a GNN is not recommended as a core thesis model.

## How to Continue

This is the current decision point after the latest local work:

- The expanded model has already been implemented and executed.
- The benchmark and model extensions have already been implemented and executed.
- Scripts 24 through 30 are present locally.
- Script 30 adds sequential specification comparison and has been executed. Its output table is `holdings/results/model_specification_comparison.csv`.
- The neural network fixes reproducibility seeds at 24 for Python random, NumPy and PyTorch, and the ML baseline has been re-run with that setting.
- A detailed ETF-level interpretation document has been added at docs/project_context/etf_model_results_interpretation.md.
- The latest pushed code state before these local edits is commit 4a0ec3d with message cleaning code.
- The current local work should not be committed or pushed until the user reviews the scripts and gives explicit approval.

The priority is now review and interpretation, not new model implementation. Any future AI should first read the new ETF-level interpretation document, then inspect the extension scripts and generated summaries. The main open task is to decide how to explain the results in the thesis, especially the negative direct channel, the stable positive comovement channel, ETF heterogeneity and the sensitivity of factor-model abnormal returns.

Before any future commit or push, run a privacy review. Do not commit raw holdings, generated model outputs, local paths, credentials, temporary files or machine-specific files.

## Handoff Update After Implementing Extensions

Scripts `24` through `30` have been added and executed successfully.

Do not describe LOO/LTO, Local Projections, Fama-French / Carhart robustness, ETF interactions, quantile regression, or ETF-level result compilation as merely planned work anymore. They are implemented and have generated local outputs under `holdings/results/` and `holdings/figures/`.

Important result memory:

- LOO/LTO benchmark robustness: `b1_term` remains negative and significant. This reduces benchmark absorption concerns.
- Local Projections: `b1_term` remains negative through `h=10`; `corr_term` remains positive.
- FF5+momentum: direct `b1_term` is sensitive and can become insignificant when using factor-surviving events; `corr_term` remains strongly positive.
- ETF interactions: heterogeneity is real and should be discussed formally.
- Quantile regression: secondary diagnostic only, but it supports negative `b1_term` across the distribution.
- ETF-level compilation: `29_compile_etf_results.py` creates summary CSVs used by `docs/project_context/etf_model_results_interpretation.md`.
- Sequential specification comparison: `30_run_specification_comparison.py` compares core, previous improved, full expanded and variable-removal specifications. It records N, adjusted R2, main-channel estimates, p-values, significant channels, runtime and two-way clustering status.

Matched-control / DiD and DML / GNN have not been implemented and remain lower priority.
Implementation note: `26_run_factor_robustness.py` was reviewed after implementation. The factor robustness merge keys must include `etf` to avoid cross-ETF contamination when the same stock appears in multiple ETFs. The local factor robustness outputs were regenerated after this fix.


Sequential specification result memory:

- The specification comparison has been executed.
- Full expanded adj R2 is 0.0355 versus 0.0309 in the previous improved specification.
- Significant main channels rise from 3 to 6.
- Removing the correlation channel makes `b1_term` insignificant (`b1 = -0.4058`, p=0.1070), so pre-event comovement is central to the expanded model.
- Removing all three new variables reproduces the previous improved specification exactly.
