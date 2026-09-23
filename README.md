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
python src/models/24_run_benchmark_robustness.py
python src/models/25_run_local_projections.py
python src/models/26_run_factor_robustness.py
python src/models/27_run_etf_interactions.py
python src/models/28_run_quantile_regression.py
python src/models/29_compile_etf_results.py
python src/models/30_run_specification_comparison.py
python src/analysis/40_exploratory_analysis.py
python src/analysis/41_raw_json_analysis.py
python src/analysis/42_integrity_checks.py
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
- LightGBM test R2: -0.0159; directional accuracy: 0.5444.
- Tanh neural network test R2: 0.0171; directional accuracy: 0.5395.

The expanded variables improve the academic interpretation of the econometric model, especially through receiver weight, pre-event comovement and ETF concentration. Predictive results remain modest, which is expected for daily abnormal returns. With fixed seed 24, LightGBM worsens in test R2 but keeps higher directional accuracy than OLS, while the tanh neural network reaches the best test R2 among the predictive baselines.

A graph neural network is still not strongly justified unless the thesis explicitly adds a graph-learning research question.

## Methodological Decisions Behind the Extensions

After reviewing the benchmark and model strategy, the next priority was not to add a more complex ML model immediately. The priority was to test whether the construction of abnormal returns was affecting the sign and size of the main econometric coefficients, especially the negative b1_term.

The current external benchmarks remain the main reference specification for now. However, the next robustness extension should add synthetic leave-one-out and leave-two-out ETF benchmarks:

- For shock identification of stock i, compute the benchmark as ETF(-i).
- For the receiver abnormal return of stock j, compute the benchmark ideally as ETF(-i,-j).
- If ETF(-i,-j) is too unstable for smaller ETFs, compare it with ETF(-j) and the current external benchmark.

This check is important because some external benchmarks still contain large ETF constituents. If the benchmark includes the shocked stock or the receiver stock, abnormal returns can be mechanically absorbed by the benchmark. This may help explain part of the negative b1_term.

The extensions were implemented in this order:

1. Local Projections following Jorda (2005), with horizons h=0,...,10, to show whether the transmission is temporary or persistent.
2. Fama-French / Carhart factor robustness, to verify that the results are not driven by the single-benchmark abnormal return model.
3. Pooled ETF-specific interaction terms, such as b1_term x ETF, to formally test heterogeneity across ETFs.
4. Quantile regression as a secondary extension to test whether spillovers are stronger in the tails.

Matched-control designs and Double/Debiased ML are lower priority. A GNN should remain future work unless the thesis explicitly becomes a graph-learning project.


## Implemented Methodological Extensions

The benchmark/model extensions discussed after the expanded model have now been implemented and executed. The new scripts are:

- `src/models/24_run_benchmark_robustness.py`: leave-one-out / leave-two-out benchmark robustness.
- `src/models/25_run_local_projections.py`: Local Projections for horizons `h=0,...,10`.
- `src/models/26_run_factor_robustness.py`: Fama-French 5-factor plus momentum robustness.
- `src/models/27_run_etf_interactions.py`: pooled ETF-specific interaction tests.
- `src/models/28_run_quantile_regression.py`: secondary quantile-regression diagnostic.
- `src/models/29_compile_etf_results.py`: ETF-level result summaries for thesis interpretation.
- `src/models/model_extension_utils.py`: shared utilities for the extension scripts.

`src/models/30_run_specification_comparison.py` has been added and executed for sequential specification comparison. It compares the core model, previous improved model, full expanded model and targeted variable-removal specifications. It records included variables, sample size, adjusted R2, main-channel coefficients, standard errors, p-values, significant channels, runtime and two-way clustering status.

Main new outputs are saved under `holdings/results/` and `holdings/figures/`, which remain local and ignored by Git.

Key results:

- LOO/LTO benchmark robustness does not remove the negative direct effect. With `ETF(-i,-j)` receiver benchmarks and surviving `ETF(-i)` origin shocks, `b1_term = -1.2369` with p<0.001. With `ETF(-j)` receiver benchmarks on the same surviving events, `b1_term = -2.2084` with p<0.001. This strengthens the interpretation that the negative direct term is not only a benchmark absorption artifact.
- Local Projections show a dynamic pattern: `b1_term` is already negative at `h=0` (-0.3908), reaches around -1.20 at `h=3`, and remains negative through `h=10` (-0.8513). The `corr_term` remains positive at every horizon.
- Fama-French / Carhart robustness is more mixed. On the same external-benchmark events, `b1_term` becomes positive and significant (`1.1575`). When only shocks that survive the factor-model threshold are kept, `b1_term` becomes small and not significant (`0.1675`, p=0.3646), while `corr_term` remains strongly positive. This means factor-model abnormal return construction materially affects the direct channel.
- Pooled ETF interactions confirm formal heterogeneity across funds, especially for the direct channel and concentration-related terms.
- Quantile regression is only a secondary diagnostic because it uses ETF and year-quarter controls rather than the full receiver fixed effects. It still shows `b1_term` negative across the 0.10, 0.25, 0.50, 0.75 and 0.90 quantiles, while `corr_term` remains positive across the distribution.

## Notes on Robustness

The robustness battery includes sign splits, pre/post-COVID periods, per-ETF regressions, top-weight trimming, extreme-value trimming, placebo event assignment, shuffled outcomes and a no-time-fixed-effects specification. The corrected placebo event assignment breaks the link between event shocks and receiver outcomes while preserving ETF-level structure. In the expanded run, the placebo coefficients lose statistical significance, including the new channels.

See docs/project_context/results_interpretation.md for a fuller academic interpretation and comparison between the previous and expanded specifications. See docs/project_context/etf_model_results_interpretation.md for the ETF-by-ETF interpretation of the baseline, benchmark robustness, factor robustness, Local Projections, pooled interactions and quantile diagnostics.

## Sequential Specification Comparison

The sequential specification comparison was executed with `src/models/30_run_specification_comparison.py`. This is an econometric specification table rather than a machine-learning ablation exercise. Its purpose is to show how the main coefficient and model fit change as theoretically motivated channels are added or removed.

The script estimates eleven specifications using the same absorbed fixed-effects framework and two-way clustered standard errors by event and receiver stock. It records included variables, number of observations, number of events, number of receivers, adjusted R2, main-channel coefficients, standard errors, p-values, number of significant channels, runtime and clustering status.

| Specification | Adj. R2 | b1 term | b1 p-value | Significant channels | Runtime sec. | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Core b1 | 0.0198 | 0.6041 | <0.001 | 1 | 8.61 | The direct weighted shock is positive when estimated alone, which means the simple bivariate channel is not enough to describe spillover structure. |
| Core + liquidity | 0.0209 | 0.1936 | 0.0631 | 1 | 8.02 | Adding liquidity absorbs part of the direct effect and makes b1 only marginally significant. |
| Core + liquidity + similarity | 0.0252 | -0.1985 | 0.0961 | 2 | 8.39 | Adding economic similarity changes the sign of b1, suggesting that relatedness among firms matters for identifying the direct channel. |
| Previous improved | 0.0309 | -0.4688 | 0.0013 | 3 | 11.72 | This reproduces the pre-expanded model and is the natural benchmark for the new variables. |
| Full expanded | 0.0355 | -1.2020 | <0.001 | 6 | 14.53 | The complete model has the best explanatory power and the largest number of significant channels. |
| Full without receiver weight | 0.0352 | -0.9280 | <0.001 | 5 | 13.49 | Receiver weight contributes to the full model but is not the main driver of b1 significance. |
| Full without correlation | 0.0323 | -0.4058 | 0.1070 | 4 | 13.38 | Removing the comovement channel makes b1 statistically insignificant, showing that correlation is central to the expanded specification. |
| Full without HHI | 0.0351 | -1.6396 | <0.001 | 4 | 13.60 | Concentration affects the scale of b1; without HHI, the direct channel becomes more negative. |
| Full without new variables | 0.0309 | -0.4688 | 0.0013 | 3 | 11.70 | This exactly reproduces the previous improved specification. |
| Full without main controls | 0.0305 | -0.7979 | <0.001 | 5 | 10.20 | Interactions alone still explain meaningful variation, but controls improve the full specification. |
| Full without asymmetry | 0.0345 | -0.8772 | <0.001 | 5 | 13.33 | The model remains stable without the negative-shock asymmetry block. |

The table gives three important conclusions.

First, the sign of `b1_term` depends strongly on the conditioning set. In the simplest model, `b1_term` is positive and significant. After adding liquidity and similarity, it becomes smaller and then negative. This means the direct channel cannot be interpreted in isolation. It is partly confounded with liquidity conditions and firm relatedness when the model is too small.

Second, the expanded variables improve the model in a meaningful but still realistic way. Adjusted R2 rises from 0.0309 in the previous improved specification to 0.0355 in the full expanded specification. This is not a large predictive jump, but for daily abnormal returns it is a relevant improvement. More importantly, the number of significant main channels rises from 3 to 6, which gives a richer and more interpretable transmission structure.

Third, the pre-event comovement channel is the key new variable. When `corr_term` and `Corr_ij_60d` are removed, `b1_term` becomes `-0.4058` and is no longer statistically significant at conventional levels. This is the strongest evidence that return comovement is not just an additional control. It changes the interpretation of the direct spillover coefficient and helps explain why the full model differs from the previous improved model.

The receiver-weight and concentration checks are also useful. Removing receiver weight keeps `b1_term` negative and significant, so receiver weight contributes to the model but is not the main source of the direct effect. Removing HHI makes `b1_term` more negative, which suggests that ETF concentration affects the scale of the direct channel. Removing asymmetry leaves the model stable, which confirms that negative-shock asymmetry is not central in the current specification.

For the thesis, this comparison should be described as a sequential specification comparison. The main narrative is that the full expanded model is preferred because it improves fit, increases the number of significant theoretically motivated channels and reveals that pre-event comovement is central to intra-ETF transmission.
