# Current Results Interpretation

This note summarizes the empirical results of the intra-ETF shock transmission project. It is written as thesis-oriented interpretation, not as a final results chapter.

The results should be read in two stages. The first stage is the previous improved specification, before adding receiver weight, pre-event return correlation and ETF concentration. The second stage is the expanded specification, which adds these variables to test whether basket structure and market connectedness explain part of the spillover mechanism.

## Previous Improved Specification

The previous improved model estimated whether abnormal return shocks in one ETF constituent were associated with abnormal returns in other constituents held by the same ETF. The final panel contained 758,013 observations. After removing rows with missing model variables, the main regression used 744,666 observations across 41,393 shock events, 208 receiver stocks and 45 year-quarter periods.

The specification included receiver-stock fixed effects and year-quarter fixed effects. The final inference used two-way clustered standard errors by event and receiver stock. The model excluded SPY from the main analysis because SPY is a broad market ETF, not a sector ETF, and it used IXC as the benchmark for XLE to avoid relying on the broad market index for the energy sector.

Two-way clustered results in the previous improved specification:

| Term | Coef. | SE | p-value |
| --- | ---: | ---: | ---: |
| `b1_term` | -0.4638 | 0.1454 | 0.0014 |
| `b2_term` | 0.3326 | 0.0540 | <0.001 |
| `b4_term` | -2.7383 | 2.1661 | 0.2062 |
| `b5_term` | 0.0187 | 0.0029 | <0.001 |
| `asym_term` | -0.1410 | 0.1001 | 0.1588 |

The direct propagation term, `b1_term = Shock_i x w_i`, was negative and statistically significant in the pooled model. This means that, on average, a larger constituent shock weighted by its ETF importance was associated with an opposite-sign abnormal reaction among receiver stocks. This should not be interpreted as the absence of spillovers. A plausible interpretation is that the direct effect captures substitution, relative-price adjustment, short-term reversal, or portfolio rebalancing forces after controlling for receiver fixed effects and time effects.

The liquidity channel, `b2_term = Shock_i x w_i x Illiq_j`, was positive and highly significant. This was one of the clearest findings. It suggested that receiver stocks with higher illiquidity were more exposed to intra-ETF transmission. This is consistent with the idea that less liquid stocks absorb order-flow pressure or common ETF-level trading effects less smoothly.

The arbitrage/mispricing channel, `b4_term = Shock_i x w_i x Mispricing_k`, was negative in the full model, but it was not statistically significant once two-way clustered standard errors were used. This means it should be treated as suggestive rather than as a core result.

The informational spillover channel, `b5_term = Shock_i x Similarity_ij`, was positive and highly significant. This was strong evidence for economically meaningful spillovers. Shocks to one constituent were more likely to be transmitted to receiver stocks that were economically similar, measured through sector and industry proximity.

The asymmetry term, `asym_term = Neg_e x Shock_i x w_i`, was negative but not statistically significant under two-way clustered standard errors. The evidence was not strong enough to make negative-shock asymmetry a main conclusion.

## Expanded Specification

The expanded model adds three variables:

- `w_j`: receiver stock weight in the ETF.
- `Corr_ij_60d`: 60-day pre-event return correlation between the shocked and receiver stock.
- `HHI_etf_t`: ETF concentration on the relevant holding date.

The rebuilt panel contains 757,368 observations. After removing rows with missing model variables, the main regression uses 743,033 observations across 41,349 shock events, 206 receiver stocks and 45 year-quarter periods.

Two-way clustered results in the expanded specification:

| Term | Coef. | SE | p-value |
| --- | ---: | ---: | ---: |
| `b1_term` | -1.2020 | 0.2161 | <0.001 |
| `b2_term` | 0.4854 | 0.0533 | <0.001 |
| `b4_term` | -3.1222 | 2.1493 | 0.1463 |
| `b5_term` | 0.0205 | 0.0028 | <0.001 |
| `receiver_weight_term` | 7.5488 | 3.6623 | 0.0393 |
| `corr_term` | 2.6345 | 0.2258 | <0.001 |
| `hhi_term` | -6.1627 | 1.9303 | 0.0014 |
| `asym_term` | -0.0996 | 0.1052 | 0.3436 |

The direct propagation term remains negative and statistically significant. Its magnitude becomes larger after adding the new channels. This suggests that the original direct term was partly mixed with basket-structure and connectedness effects. Once those channels are separated, the remaining direct effect looks more clearly like relative-price adjustment or rebalancing rather than simple same-direction contagion.

The liquidity channel remains positive and highly significant. Its coefficient increases from 0.3326 to 0.4854, which suggests that the liquidity mechanism is not weakened by the new variables. Instead, the liquidity interpretation becomes stronger because it survives a richer specification.

The informational similarity channel remains positive and highly significant. Its coefficient also increases slightly, from 0.0187 to 0.0205. This means that static economic similarity still matters even after controlling for dynamic return comovement.

The receiver-weight channel, `receiver_weight_term = Shock_i x w_i x w_j`, is positive and significant. This means that transmission is stronger when the receiver stock also has a larger weight in the ETF. This adds a useful basket-structure interpretation: both the origin weight and the receiver weight matter.

The return-comovement channel, `corr_term = Shock_i x w_i x Corr_ij_60d`, is positive and strongly significant. This is the most important new result. It shows that shocks transmit more strongly between stocks that already moved together before the event. This supports a connectedness interpretation that goes beyond static GICS similarity.

The ETF concentration channel, `hhi_term = Shock_i x w_i x HHI_etf_t`, is negative and significant. This suggests that concentration changes the way shocks propagate through the ETF basket. The negative sign should be interpreted carefully: in more concentrated ETFs, the marginal direct weighted shock may be partly absorbed or offset by dominant-name structure, sector-specific hedging, or relative-price adjustments.

The asymmetry term remains not significant. Therefore, the expanded specification does not change the conclusion that negative-shock asymmetry is not a main result at this stage.

## Effect of Adding the New Variables

The new variables do not overturn the previous results. They refine them.

The main robust findings from the previous model remain valid: liquidity and informational similarity are still significant, and the direct weighted shock term is still negative. The expanded model adds a more precise explanation of why transmission differs across receiver stocks and ETFs.

The key change is that the model now separates three mechanisms that were not explicit before:

1. Receiver importance in the ETF basket.
2. Dynamic market connectedness before the event.
3. ETF portfolio concentration.

This is useful for the thesis because it shows that intra-ETF transmission is not only about the shocked stock's weight or the receiver's liquidity. It also depends on how important the receiver is inside the ETF and how connected the two stocks already were in market returns.

## Heterogeneity Across ETFs

In the previous interpretation, the per-ETF regressions showed that the transmission mechanism was not homogeneous across funds. XLE, IHE and XLV showed strong negative direct propagation, while XME showed a positive direct propagation term.

In the expanded model, ETF-level heterogeneity remains important. The direct weighted shock term is negative in all ETF-specific regressions, but the size and channel behavior differ strongly across `XME`, `XLE`, `IHE` and `XLV`. The return-comovement channel is positive and significant across the ETF-specific regressions, which supports the idea that pre-event market connectedness is a stable transmission mechanism.

## Robustness Interpretation

The results are broadly robust to several checks, including positive/negative shock splits, pre/post-COVID periods, per-ETF regressions, excluding top-weight observations, trimming extreme values, shuffled outcomes and removing year-quarter fixed effects.

The corrected placebo event-assignment check is especially important. It randomly reassigns event-level shock variables within ETF and recomputes the interaction terms. In the expanded run, the main placebo coefficients lose statistical significance, including `receiver_weight_term`, `corr_term` and `hhi_term`. This supports the idea that the results are tied to the actual event timing and event-receiver structure.

The shuffled-outcome placebo also works as expected: after randomly permuting `AR_j`, the adjusted R2 collapses toward zero and the main coefficients lose significance.

## Machine Learning and Neural Network Results

The machine-learning script compares a simple OLS predictive baseline, LightGBM and a feed-forward neural network. The neural network follows the supervisor's suggestion: the activation function is fixed to `tanh`, and the hyperparameter grid is modest, varying the number of hidden layers and the number of nodes per layer.

ML comparison before and after the expanded variables:

| Specification | Model | Test R2 | Test MAE | Directional Accuracy |
| --- | --- | ---: | ---: | ---: |
| Previous improved | OLS baseline | 0.0002 | 0.0221 | 0.5230 |
| Previous improved | LightGBM | 0.0163 | 0.0220 | 0.5407 |
| Previous improved | Tanh neural network | 0.0146 | 0.0220 | 0.5377 |
| Expanded | OLS baseline | 0.0004 | 0.0221 | 0.5274 |
| Expanded, seed 24 | LightGBM | -0.0159 | 0.0222 | 0.5444 |
| Expanded, seed 24 | Tanh neural network | 0.0171 | 0.0220 | 0.5395 |

The expanded feature set does not improve predictive R2 for LightGBM. With seed 24, the tanh neural network gives the best test R2 among the predictive baselines, while LightGBM keeps higher directional accuracy than OLS. This supports the thesis interpretation that nonlinear ML can capture some weak signal, but not enough to replace the econometric specification as the main model.

## Open Interpretation Risk: Benchmark Absorption

The most important open interpretation issue is the negative b1_term. At the moment, the result can be read as evidence of relative-price adjustment, short-term reversal, substitution, or rebalancing after a shock. However, this interpretation should remain provisional until benchmark robustness is completed.

The reason is that some external benchmarks can still include the shocked stock or the receiver stock. If the shocked stock is inside the benchmark, part of the shock may be absorbed into the benchmark return. If the receiver stock is inside the benchmark, the receiver abnormal return may be mechanically compressed. This can affect both the magnitude and the sign of the estimated direct propagation term.

The agreed next diagnostic is therefore:

1. Use ETF(-i) for identifying shocks to stock i.
2. Use ETF(-i,-j) for receiver abnormal returns of stock j when feasible.
3. Compare these results with the current external-benchmark results.

If b1_term remains negative under the leave-two-out benchmark, the reversal/rebalancing interpretation becomes much stronger. If b1_term weakens or changes sign, the thesis should explain that benchmark absorption was an important driver of the earlier negative coefficient.


## Interpretation After Benchmark and Model Extensions

The new extensions give a clearer interpretation of the previous results.

The strongest update comes from LOO/LTO benchmark robustness. The concern was that the negative `b1_term` could be mechanically generated if the benchmark absorbed the shocked stock or the receiver stock. After recomputing shocks with `ETF(-i)` and receiver abnormal returns with `ETF(-i,-j)`, the direct term remains negative and highly significant. This means the reversal/rebalancing interpretation is stronger than before.

Local Projections also support this interpretation. The direct term is negative at all horizons from `h=0` to `h=10`, while the return-comovement channel remains positive. This suggests that the main result is not only a one-day artifact.

The Fama-French / Carhart check is the main caution. Under FF5+momentum abnormal returns, the direct channel is sensitive and can lose significance when only factor-surviving shocks are used. Therefore, the thesis should present the direct `b1` result as robust to synthetic ETF benchmark construction, but sensitive to the expected-return model. By contrast, `corr_term` is consistently positive and is now one of the most stable findings.

The quantile diagnostic shows that `b1_term` is negative across the distribution of receiver abnormal returns and that `corr_term` is positive across quantiles. Because this diagnostic does not include the full receiver fixed effects, it should be used as supporting evidence rather than as a main specification.

## Should We Build a Graph Neural Network?

A graph neural network is still not strongly justified as a core thesis model. The econometric model now already includes graph-like information through pre-event return correlation and economic similarity. The latest ML results do not show a large predictive gain that would justify the extra complexity of a GNN.

A GNN could be mentioned as a future extension if the thesis explicitly shifts toward network learning. A graph could connect stocks by ETF co-membership, holding-weight overlap, GICS similarity, return correlation or lead-lag relationships. For the current thesis scope, the priority should be to finalize the econometric specification and write a clear interpretation of the robust channels.

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
