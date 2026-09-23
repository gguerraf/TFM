# Intra-ETF Shock Transmission

This repository contains the code for my Master's Thesis on intra-ETF shock transmission.

The project studies whether an abnormal shock to one ETF constituent is related to abnormal returns in other stocks held by the same ETF.

## Repository Structure

```text
src/
  pipeline/      Data loading, price downloads, benchmarks, volume and returns.
  models/        Econometric models, robustness checks and ML baselines.
  analysis/      Exploratory analysis and data-quality checks.
  utils/         Helper scripts.
```

## Main Workflow

Run the full main workflow from the repository root:

```powershell
python run_all.py
```

This runs the data pipeline and the model scripts in order. It checks the main Python dependencies first and stops if any script fails.

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

Useful options:

```text
python run_all.py --dry-run
python run_all.py --stage pipeline
python run_all.py --stage models
python run_all.py --include-analysis
python run_all.py --skip-dependency-check
```

Exploratory analysis scripts are not included by default. Use `--include-analysis` if you also want to run them.

## Main Model

The econometric model is implemented in:

```text
src/models/21_estimate_improved_model.py
```

The ETF universe is:

- `XME`
- `XLE`
- `IHE`
- `XLV`

`SPY` is excluded from the main analysis because it is a broad market ETF.

## Main Results

The model uses 743,033 observations in the main regression.

| Channel | Coef. | p-value |
| --- | ---: | ---: |
| Direct shock | -1.2020 | <0.001 |
| Liquidity | 0.4854 | <0.001 |
| Mispricing | -3.1222 | 0.146 |
| Similarity | 0.0205 | <0.001 |
| Receiver weight | 7.5488 | 0.039 |
| Return correlation | 2.6345 | <0.001 |
| ETF concentration | -6.1627 | 0.001 |
| Negative shock | -0.0996 | 0.344 |

The most stable positive mechanism is the pre-event return correlation channel.

## ML Baselines

The project also compares predictive models.

| Model | Test R2 | Test MAE | Directional accuracy |
| --- | ---: | ---: | ---: |
| OLS baseline | 0.0004 | 0.0221 | 0.5274 |
| LightGBM | -0.0159 | 0.0222 | 0.5444 |
| Tanh neural network | 0.0171 | 0.0220 | 0.5395 |

The neural network uses `tanh` activation and a small grid over hidden layers and nodes per layer.


## Sequential Specification Comparison

The specification comparison is produced by:

```text
src/models/30_run_specification_comparison.py
```

| Specification | Adj. R2 | b1 term | b1 p-value | Significant channels | Runtime sec. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core b1 | 0.0198 | 0.6041 | <0.001 | 1 | 8.61 |
| Core + liquidity | 0.0209 | 0.1936 | 0.0631 | 1 | 8.02 |
| Core + liquidity + similarity | 0.0252 | -0.1985 | 0.0961 | 2 | 8.39 |
| Previous improved | 0.0309 | -0.4688 | 0.0013 | 3 | 11.72 |
| Full expanded | 0.0355 | -1.2020 | <0.001 | 6 | 14.53 |
| Full without receiver weight | 0.0352 | -0.9280 | <0.001 | 5 | 13.49 |
| Full without correlation | 0.0323 | -0.4058 | 0.1070 | 4 | 13.38 |
| Full without HHI | 0.0351 | -1.6396 | <0.001 | 4 | 13.60 |
| Full without new variables | 0.0309 | -0.4688 | 0.0013 | 3 | 11.70 |
| Full without main controls | 0.0305 | -0.7979 | <0.001 | 5 | 10.20 |
| Full without asymmetry | 0.0345 | -0.8772 | <0.001 | 5 | 13.33 |

The full expanded model has the highest adjusted R2 and the largest number of significant channels.

Removing the correlation channel makes `b1_term` insignificant.