# AI Handoff Notes

Last updated: 2026-09-22

This document replaces the removed root-level `AGENTS.md`. The root `AGENTS.md` is intentionally ignored and not pushed to GitHub. Use this file as the safe, repository-tracked handoff for future AI agents.

## What This Project Does

The thesis studies intra-ETF shock transmission. It asks whether abnormal return shocks to one ETF constituent transmit to other constituents in the same ETF, and through which mechanisms: direct weighted exposure, liquidity, ETF mispricing/arbitrage and informational similarity.

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
- Absorb receiver-stock and year-quarter fixed effects with `AbsorbingLS`.
- Use two-way clustered standard errors by `event_id` and `stock_j` for final inference.

Do not switch back to explicit dummy-variable OLS for the final model. It caused memory and numerical failures on the full panel.

## Robustness Details

`src/models/22_run_robustness_checks.py` also uses `AbsorbingLS`. This keeps robustness specifications consistent with the main model and avoids the memory failures seen with explicit dummy variables.

The corrected placebo is `R16_Placebo_event_assignment`. It randomly reassigns event-level shock variables within each ETF and recomputes interaction terms. This replaced the old placebo that only shifted dates and therefore did not change the regression design in a meaningful way.

## Neural Network Details

`src/models/23_run_ml_baselines.py` implements the supervisor-aligned MLP:

- Fixed activation: `tanh`.
- Grid: hidden layers and nodes per layer.
- Current grid: hidden layers `[1, 2, 3]`, nodes `[16, 32, 64]`.
- Fixed learning rate: 0.01.
- Fixed dropout: 0.0.
- Fixed batch size: 2048.

Current best NN: 3 hidden layers and 16 nodes per layer.

The current NN improves over the previous NN run but remains slightly weaker than LightGBM. Therefore, a GNN is not recommended as a core thesis model unless the research question is explicitly expanded toward graph learning.

## Current Results to Preserve

Final two-way clustered main results:

- `b1_term`: coefficient -0.4638, p=0.0014.
- `b2_term`: coefficient 0.3326, p<0.001.
- `b4_term`: coefficient -2.7383, p=0.2062.
- `b5_term`: coefficient 0.0187, p<0.001.
- `asym_term`: coefficient -0.1410, p=0.1588.

Interpretation:

- Robust evidence for liquidity and informational similarity channels.
- Robust negative direct weighted shock term.
- Weak evidence for arbitrage/mispricing once two-way clustering is used.
- Weak evidence for negative-shock asymmetry once two-way clustering is used.

## How to Continue

Best next steps:

1. Decide whether to add one extra theoretically motivated variable.
2. If adding a variable, update the panel construction, regression, robustness and documentation together.
3. Re-run `21_estimate_improved_model.py`, `22_run_robustness_checks.py`, and `23_run_ml_baselines.py` after any model change.
4. Update `README.md`, `docs/project_context/progress.md`, `docs/project_context/methodology.md`, `docs/project_context/architecture.md`, and this handoff document after any substantive code change.

