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

The model uses 793,933 observations in the main regression. Standard errors are clustered by event and by receiver.

| Channel | Coef. | p-value |
| --- | ---: | ---: |
| Direct shock | -1.2308 | <0.001 |
| Liquidity | 0.5277 | <0.001 |
| Mispricing | -2.5491 | 0.204 |
| Similarity | 0.0215 | <0.001 |
| Receiver weight | 7.8635 | 0.034 |
| Return correlation | 2.7541 | <0.001 |
| ETF concentration | -6.2183 | 0.001 |
| Negative shock | -0.1393 | 0.189 |

The most stable positive mechanism is the pre-event return correlation channel.

## ML Baselines

The project also compares predictive models on the test period (January 2024 to February 2026).

| Model | Test R2 | Test MAE | Directional accuracy |
| --- | ---: | ---: | ---: |
| OLS baseline | 0.0007 | 0.0224 | 0.5287 |
| LightGBM | -0.0158 | 0.0225 | 0.5342 |
| Tanh neural network | 0.0205 | 0.0223 | 0.5474 |

The neural network uses `tanh` activation and a small grid over hidden layers and nodes per layer. A simple rule that predicts the sign of the shock reaches a directional accuracy of 0.5412.

## Sequential Specification Comparison

The specification comparison is produced by:

```text
src/models/30_run_specification_comparison.py
```

| Specification | Adj. R2 | b1 term | b1 p-value | Significant channels |
| --- | ---: | ---: | ---: | ---: |
| Core b1 | 0.0205 | 0.6736 | <0.001 | 1 |
| Core + liquidity | 0.0218 | 0.1650 | 0.1267 | 1 |
| Core + liquidity + similarity | 0.0260 | -0.2356 | 0.0571 | 2 |
| Previous improved | 0.0313 | -0.4475 | 0.0025 | 3 |
| Full expanded | 0.0361 | -1.2308 | <0.001 | 6 |
| Full without receiver weight | 0.0358 | -0.9463 | <0.001 | 5 |
| Full without correlation | 0.0328 | -0.3861 | 0.1291 | 4 |
| Full without HHI | 0.0358 | -1.6709 | <0.001 | 4 |
| Full without new variables | 0.0313 | -0.4475 | 0.0025 | 3 |
| Full without main controls | 0.0315 | -0.8540 | <0.001 | 7 |
| Full without asymmetry | 0.0353 | -0.9728 | <0.001 | 6 |

The full expanded model has the highest adjusted R2. The version without the main controls has one more significant channel, but a clearly lower fit.

## Trading Simulation

The trading simulation is produced by:

```text
src/models/32_run_trading_simulation.py
```

It tests whether the transmission can be exploited with signals that use only the information available at the time of each trade. Positions are hedged with the ETF benchmark, aggregated in calendar-time portfolios and charged transaction costs. In the test period (January 2024 to February 2026), no strategy has a significant gross return, and the break-even costs are below 5 basis points per side.
