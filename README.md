# TFM - Intra-ETF Shock Transmission

This repository contains the code and project documentation for a Master's Thesis on intra-ETF spillover effects. The project studies whether abnormal return shocks to one ETF constituent are transmitted to other stocks held by the same ETF.

## Repository Structure

```text
src/
  pipeline/      Data ingestion, ticker repair, price downloads, benchmarks, GICS, volume and returns.
  models/        Econometric models, robustness checks, LightGBM and neural-network baselines.
  analysis/      Exploratory data analysis and data-quality checks.
  utils/         Company universe extraction and deduplication utilities.

docs/
  reports/       Progress review report files.
  data_quality/  Small data-quality workbooks.
  project_context/ Methodology, architecture, progress notes and result interpretation.

data/            Placeholder for local/raw data notes. Large data is not committed.
outputs/         Placeholder for generated model outputs. Outputs are not committed.
holdings/        Local data folder expected by the scripts. This folder is ignored by Git.
```

## Data Policy

Large raw and processed data files are not committed to GitHub. The scripts expect a local `holdings/` folder with raw ETF JSON files and processed CSVs.

The main processed files used by the model are expected under:

```text
holdings/processed/
```

The model writes generated panels, regressions and plots under:

```text
holdings/results/
holdings/figures/
```

## Main Workflow

Run scripts from the repository root.

```powershell
python src/pipeline/00_validate_holdings.py
python src/pipeline/00_validate_duplicate_holdings.py
python src/pipeline/01_load_holdings.py
python src/utils/extract_company_universe.py
python src/utils/deduplicate_company_universe.py
python src/pipeline/02_download_prices.py
python src/pipeline/03_check_missing_downloads.py
python src/pipeline/04_retry_missing_tickers.py
python src/pipeline/05_apply_ticker_mappings.py
python src/pipeline/06_download_benchmarks.py
python src/pipeline/07_download_energy_benchmark.py
python src/pipeline/08_download_gics.py
python src/pipeline/09_add_manual_gics_labels.py
python src/pipeline/10_download_volume.py
python src/pipeline/11_compute_returns.py
python src/models/20_estimate_baseline_model.py
python src/models/21_estimate_improved_model.py
python src/models/22_run_robustness_checks.py
python src/models/23_run_ml_baselines.py
python src/analysis/30_exploratory_analysis.py
python src/analysis/31_raw_json_analysis.py
python src/analysis/32_integrity_checks.py
```

## Current Status

The improved econometric model has been executed and validated locally. The final improved panel contains 758,013 observations, with 744,666 observations used in the main regression after dropping missing model variables.

Main econometric findings:

- The liquidity channel is positive and statistically significant.
- The informational spillover channel is positive and statistically significant.
- The direct baseline propagation term is negative and remains significant with two-way clustered standard errors.
- The liquidity channel is positive and remains significant with two-way clustered standard errors.
- The informational spillover channel is positive and remains significant with two-way clustered standard errors.
- The arbitrage/mispricing channel and the negative-shock asymmetry term are weaker: they are marginal under one-way clustering but not significant under two-way clustering.

Machine-learning findings:

- OLS baseline test R2: 0.0002; directional accuracy: 0.5230.
- LightGBM test R2: 0.0163; directional accuracy: 0.5407.
- Tanh neural network test R2: 0.0146; directional accuracy: 0.5377.

LightGBM improves over the simple OLS predictive baseline, but the improvement is limited. The feed-forward neural network follows the supervisor's suggestion: fixed `tanh` activation and a modest grid over hidden layers and nodes per layer. Based on the current results, a graph neural network is not yet strongly justified unless a more explicit graph-learning research question is added.

## Notes on Robustness

The robustness battery includes sign splits, pre/post-COVID periods, per-ETF regressions, top-weight trimming, extreme-value trimming, placebo event assignment, shuffled outcomes and a no-time-fixed-effects specification. The corrected placebo event assignment breaks the link between event shocks and receiver outcomes while preserving ETF-level structure; in the current run the main placebo coefficients lose statistical significance.

See `docs/project_context/results_interpretation.md` for a fuller academic interpretation of the current results.

