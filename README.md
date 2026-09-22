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

The improved econometric model has been executed in two comparable stages.

### Previous Improved Specification

This was the improved model before adding receiver weight, pre-event return correlation and ETF concentration. The panel contained 758,013 observations, with 744,666 observations used in the main regression after dropping missing model variables.

Main two-way clustered findings in that version:

- `b1_term`: -0.4638, p=0.0014.
- `b2_term`: 0.3326, p<0.001.
- `b4_term`: -2.7383, p=0.2062.
- `b5_term`: 0.0187, p<0.001.
- `asym_term`: -0.1410, p=0.1588.

ML results in that version:

- OLS baseline test R2: 0.0002; directional accuracy: 0.5230.
- LightGBM test R2: 0.0163; directional accuracy: 0.5407.
- Tanh neural network test R2: 0.0146; directional accuracy: 0.5377.

### Expanded Specification

The latest model adds three additional academic variables: receiver portfolio weight (`w_j`), pre-event return correlation (`Corr_ij_60d`) and ETF concentration (`HHI_etf_t`). The rebuilt panel contains 757,368 observations, with 743,033 observations used in the main regression.

Main two-way clustered findings in the expanded version:

- `b1_term`: -1.2020, p<0.001.
- `b2_term`: 0.4854, p<0.001.
- `b4_term`: -3.1222, p=0.1463.
- `b5_term`: 0.0205, p<0.001.
- `receiver_weight_term`: 7.5488, p=0.0393.
- `corr_term`: 2.6345, p<0.001.
- `hhi_term`: -6.1627, p=0.0014.
- `asym_term`: -0.0996, p=0.3436.

ML results in the expanded version:

- OLS baseline test R2: 0.0004; directional accuracy: 0.5274.
- LightGBM test R2: -0.0122; directional accuracy: 0.5449.
- Tanh neural network test R2: 0.0146; directional accuracy: 0.5473.

The expanded variables improve the academic interpretation of the econometric model, especially through receiver weight, pre-event comovement and ETF concentration. However, they do not clearly improve out-of-sample predictive R2. LightGBM worsens in test R2, while the neural network remains similar in R2 and improves slightly in directional accuracy.

A graph neural network is still not strongly justified unless the thesis explicitly adds a graph-learning research question.

## Notes on Robustness

The robustness battery includes sign splits, pre/post-COVID periods, per-ETF regressions, top-weight trimming, extreme-value trimming, placebo event assignment, shuffled outcomes and a no-time-fixed-effects specification. The corrected placebo event assignment breaks the link between event shocks and receiver outcomes while preserving ETF-level structure. In the expanded run, the placebo coefficients lose statistical significance, including the new channels.

See `docs/project_context/results_interpretation.md` for a fuller academic interpretation and comparison between the previous and expanded specifications.
