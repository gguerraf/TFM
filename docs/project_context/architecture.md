# System Architecture & Data Pipeline

> **Master's Thesis:** *Structural Dynamics and Contagion Mechanisms of Intra-ETF Shock Transmission*
> **Scripts Directory:** [`src`](../../src)
> **Status:** Current GitHub-style repository layout after reorganization

## 1. High-Level Workflow

The project is organized as a sequential research pipeline:

```text
Raw ETF holdings JSON files
        |
        v
src/pipeline/01_load_holdings.py
        |
        v
holdings/processed/{etf}_holdings.csv
        |
        v
Price, benchmark, GICS and volume downloads
        |
        v
holdings/processed/returns_clean.csv
holdings/processed/benchmarks.csv
holdings/processed/gics_data.csv
holdings/processed/amihud.csv
        |
        v
src/models/21_estimate_improved_model.py
        |
        v
holdings/results/panel_improved.csv
holdings/results/regression_improved.txt
        |
        +--> src/models/22_run_robustness_checks.py
        |
        +--> src/models/23_run_ml_baselines.py
```

The repository tracks code and documentation. The local `holdings/` folder contains raw data, processed data and generated outputs, and is intentionally ignored by Git.

## 2. Repository Structure

```text
src/
  pipeline/      Data ingestion, ticker repair, price downloads, benchmarks, GICS, volume and returns.
  models/        Baseline model, improved econometric model, robustness checks, ML and NN baselines.
  analysis/      Exploratory analysis and data integrity checks.
  utils/         Company universe extraction and deduplication utilities.

docs/
  project_context/  Methodology, architecture, progress notes, results interpretation and AI handoff.
  reports/          Progress review report files.
  data_quality/     Small data-quality workbooks.

data/            Placeholder for local/raw data notes. Large data is not committed.
outputs/         Placeholder for generated outputs. Large outputs are not committed.
holdings/        Local data folder expected by scripts. Ignored by Git.
```

## 3. Main Processed Inputs

The model expects the following local files under `holdings/processed/`:

| File | Role |
| --- | --- |
| `{etf}_holdings.csv` | Daily ETF holdings with constituent weights. |
| `returns_clean.csv` | Clean daily log returns for constituents and ETFs. |
| `benchmarks.csv` | Sector benchmark prices and benchmark returns. |
| `gics_data.csv` | Sector and industry labels used for `Similarity_ij`. |
| `amihud.csv` | Rolling Amihud illiquidity used for `Illiq_j`. |

## 4. Modeling Outputs

The main generated outputs are local and ignored by Git:

| Output | Producer | Purpose |
| --- | --- | --- |
| `holdings/results/panel_improved.csv` | `21_estimate_improved_model.py` | Receiver-event panel for the improved model. |
| `holdings/results/regression_improved.txt` | `21_estimate_improved_model.py` | Main one-way and two-way clustered regression results. |
| `holdings/results/shock_diagnostics.csv` | `21_estimate_improved_model.py` | Event-level shock diagnostics. |
| `holdings/results/robustness_summary.csv` | `22_run_robustness_checks.py` | Summary table for robustness checks. |
| `holdings/results/robustness_details.txt` | `22_run_robustness_checks.py` | Detailed robustness log. |
| `holdings/results/ml_results.txt` | `23_run_ml_baselines.py` | OLS, LightGBM and neural-network metrics. |
| `holdings/results/nn_grid_results.csv` | `23_run_ml_baselines.py` | Neural-network grid-search results. |
| `holdings/results/lgbm_feature_importance.csv` | `23_run_ml_baselines.py` | LightGBM feature importance table. |

## 5. Improved Model Architecture

`src/models/21_estimate_improved_model.py` is the current preferred econometric model.

It performs these steps:

1. Loads clean returns, benchmark returns, GICS labels and Amihud illiquidity.
2. Excludes `SPY` from the main analysis.
3. Uses `IXC` as the benchmark for `XLE`.
4. Identifies shocks using event-specific market-model residual volatility.
5. Builds the receiver-event panel `(event, stock_j)`.
6. Removes receiver observations contaminated by nearby receiver shocks.
7. Adds mechanism variables: `Illiq_j`, `Mispricing_k`, `Similarity_ij`, `w_j`, `Corr_ij_60d` and `HHI_etf_t`.
8. Adds interaction channels: `b1_term`, `b2_term`, `b4_term`, `b5_term`, `receiver_weight_term`, `corr_term`, `hhi_term` and `asym_term`.
9. Estimates the model with receiver-stock and year-quarter absorbed fixed effects using `AbsorbingLS`.
10. Reports one-way clustered and two-way clustered standard errors.

The script checks the cached panel columns before reusing `panel_improved.csv`. If the cached panel does not contain the latest variables, it rebuilds the panel.

## 6. Robustness Architecture

`src/models/22_run_robustness_checks.py` reuses the improved panel and estimates a battery of checks with absorbed fixed effects.

Current checks include:

- Baseline specification.
- Positive-shock and negative-shock subsamples.
- Pre-COVID and post-COVID subsamples.
- Per-ETF regressions.
- Top-weight exclusion.
- 1st-99th percentile trimming.
- Placebo event assignment within ETF.
- Randomized receiver membership.
- Shuffled receiver abnormal returns.
- No year-quarter fixed effects.

The corrected placebo recomputes all shock-based interaction terms after reassigning event-level shock variables.

## 7. ML and Neural Network Architecture

`src/models/23_run_ml_baselines.py` uses the improved panel and compares:

- Simple OLS without fixed effects.
- LightGBM.
- Feed-forward neural network.

The neural network follows supervisor guidance:

- Fixed `tanh` activation.
- Grid over hidden layers `[1, 2, 3]`.
- Grid over nodes per layer `[16, 32, 64]`.
- Fixed learning rate, dropout and batch size.

The latest run does not justify a graph neural network as a core thesis model, because predictive gains remain limited.

## 8. Implemented Methodological Extensions

The benchmark and model extensions have now been implemented and executed. They were kept as separate scripts so the main expanded model stays stable while each robustness layer can be reproduced independently.

The implemented order was:

1. Leave-one-out and leave-two-out synthetic ETF benchmark robustness.
2. Local Projections with horizons h=0,...,10.
3. Fama-French 5-factor plus momentum abnormal-return robustness.
4. Pooled ETF-specific interaction tests.
5. Quantile regression as a secondary tail diagnostic.
6. ETF-level result compilation for thesis interpretation.

The key reason for this order was methodological. Benchmark construction had to be checked before interpreting more complex models, because contamination in the benchmark could affect the sign and size of the direct spillover coefficient.


## 9. Extension Script Architecture

The methodological extensions are implemented as separate scripts instead of being merged into `21_estimate_improved_model.py`. This keeps the main model stable and makes each robustness layer reproducible.

| Script | Purpose | Main outputs |
| --- | --- | --- |
| `24_run_benchmark_robustness.py` | LOO/LTO synthetic benchmark robustness | `benchmark_robustness_summary.csv`, `benchmark_robustness_details.txt`, `benchmark_robustness_panel.csv` |
| `25_run_local_projections.py` | Dynamic horizon-by-horizon regressions | `local_projections_summary.csv`, `local_projections_details.txt`, `local_projection_irf.png` |
| `26_run_factor_robustness.py` | FF5+momentum abnormal return robustness | `factor_robustness_summary.csv`, `factor_robustness_details.txt`, `factor_robustness_panel.csv` |
| `27_run_etf_interactions.py` | Pooled ETF heterogeneity tests | `etf_interactions_summary.csv`, `etf_interactions_details.txt` |
| `28_run_quantile_regression.py` | Secondary tail diagnostic | `quantile_regression_summary.csv`, `quantile_regression_details.txt` |
| `29_compile_etf_results.py` | ETF-level summaries for thesis interpretation | `etf_baseline_summary.csv`, `etf_benchmark_robustness_summary.csv`, `etf_factor_robustness_summary.csv`, `etf_local_projections_summary.csv`, `etf_quantile_regression_summary.csv` |
| `30_run_specification_comparison.py` | Sequential specification comparison | `model_specification_comparison.csv`, `model_specification_comparison_details.txt` |
