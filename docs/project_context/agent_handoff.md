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

1. `src/analysis/30_exploratory_analysis.py`
2. `src/analysis/31_raw_json_analysis.py`
3. `src/analysis/32_integrity_checks.py`

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

Current best NN: 3 hidden layers and 16 nodes per layer.

Latest ML results after adding the new variables:

- OLS test R2 0.0004, MAE 0.0221, directional accuracy 0.5274.
- LightGBM test R2 -0.0122, MAE 0.0222, directional accuracy 0.5449.
- Tanh NN test R2 0.0146, MAE 0.0220, directional accuracy 0.5473.

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
- LightGBM test R2 -0.0122, MAE 0.0222, directional accuracy 0.5449.
- Tanh NN test R2 0.0146, MAE 0.0220, directional accuracy 0.5473.

Interpretation:

- The old robust findings survive: liquidity and informational similarity remain significant.
- The direct weighted shock term becomes more negative after controlling for receiver weight, comovement and concentration.
- The strongest new channel is pre-event return comovement.
- Receiver weight also adds a positive channel.
- ETF concentration adds a significant negative channel, but its economic interpretation should be cautious.
- ML evidence remains limited, so a GNN is not recommended as a core thesis model.

## How to Continue

Best next steps:

1. Decide whether all three new channels should remain in the final thesis specification or whether one should be presented as a robustness extension.
2. If simplifying the model, compare against both documented result versions before removing terms.
3. Consider a simpler ML feature set if predictive R2 matters, because LightGBM worsened after adding the expanded features.
4. Re-run `21_estimate_improved_model.py`, `22_run_robustness_checks.py`, and `23_run_ml_baselines.py` after any model change.
5. Update `README.md`, `docs/project_context/progress.md`, `docs/project_context/methodology.md`, `docs/project_context/architecture.md`, `docs/project_context/results_interpretation.md`, and this handoff document after any substantive code change.
