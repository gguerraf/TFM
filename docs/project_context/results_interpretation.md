# Current Results Interpretation

This note summarizes the current empirical results of the intra-ETF shock transmission project. It is written as thesis-oriented interpretation, not as a final results chapter.

## Main Econometric Result

The improved model estimates whether abnormal return shocks in one ETF constituent are associated with abnormal returns in other constituents held by the same ETF. The final panel contains 758,013 observations. After removing rows with missing model variables, the main regression uses 744,666 observations across 41,393 shock events, 208 receiver stocks and 45 year-quarter periods.

The main specification includes receiver-stock fixed effects and year-quarter fixed effects. The final inference uses two-way clustered standard errors by event and receiver stock. The model excludes SPY from the main analysis because SPY is a broad market ETF, not a sector ETF, and it uses IXC as the benchmark for XLE to avoid relying on the broad market index for the energy sector.

## Interpretation of the Main Coefficients

The direct propagation term, `b1_term = Shock_i x w_i`, is negative and statistically significant in the pooled model. This means that, on average, a larger constituent shock weighted by its ETF importance is associated with an opposite-sign abnormal reaction among receiver stocks. This should not be interpreted as the absence of spillovers. A plausible interpretation is that the direct effect captures substitution, relative-price adjustment, short-term reversal, or portfolio rebalancing forces after controlling for receiver fixed effects and time effects.

The liquidity channel, `b2_term = Shock_i x w_i x Illiq_j`, is positive and highly significant. This is one of the clearest findings. It suggests that receiver stocks with higher illiquidity are more exposed to intra-ETF transmission. This is consistent with the idea that less liquid stocks absorb order-flow pressure or common ETF-level trading effects less smoothly.

The arbitrage/mispricing channel, `b4_term = Shock_i x w_i x Mispricing_k`, is negative in the full model, but it is not statistically significant once two-way clustered standard errors are used. This means it should be treated as suggestive rather than as a core result. The sign may reflect correction forces when the ETF has moved away from its benchmark over the pre-event window, but the evidence is weaker than for liquidity and informational spillovers.

The informational spillover channel, `b5_term = Shock_i x Similarity_ij`, is positive and highly significant. This is the strongest evidence for economically meaningful spillovers. Shocks to one constituent are more likely to be transmitted to receiver stocks that are economically similar, measured through sector and industry proximity.

The asymmetry term, `asym_term = Neg_e x Shock_i x w_i`, is negative but not statistically significant under two-way clustered standard errors. This suggests that negative shocks may propagate differently from positive shocks, but the current evidence is not strong enough to make asymmetry a main conclusion.

## Heterogeneity Across ETFs

The per-ETF regressions show that the transmission mechanism is not homogeneous across funds. XLE, IHE and XLV show strong negative direct propagation, while XME shows a positive direct propagation term. This matters for the thesis because it means that intra-ETF spillovers are not a single universal effect. The sector structure, constituent composition, liquidity and concentration of each ETF appear to influence the direction and magnitude of transmission.

The informational channel is positive and significant in all ETF-specific regressions. This supports the interpretation that economic relatedness between firms is a stable transmission mechanism. The liquidity and mispricing channels are more heterogeneous, which is reasonable because liquidity conditions and ETF arbitrage dynamics differ across sectors.

## Robustness Interpretation

The results are broadly robust to several checks, including positive/negative shock splits, pre/post-COVID periods, per-ETF regressions, excluding top-weight observations, trimming extreme values, shuffled outcomes and removing year-quarter fixed effects.

The shuffled-outcome placebo is especially useful: after randomly permuting `AR_j`, the main coefficients lose statistical significance and the adjusted R2 collapses toward zero. This indicates that the observed relationships are not purely mechanical artifacts of the design matrix.

The placebo event-assignment check should be interpreted as a falsification exercise. It breaks the link between the original shock event and the receiver outcome while preserving ETF-level structure. If this placebo produces much weaker effects than the baseline, it supports the idea that timing and event assignment matter.

## Machine Learning and Neural Network Results

The machine-learning script compares a simple OLS predictive baseline, LightGBM and a feed-forward neural network. The neural network follows the supervisor's suggestion: the activation function is fixed to `tanh`, and the hyperparameter grid is modest, varying the number of hidden layers and the number of nodes per layer. In the current run, the best configuration is 3 hidden layers with 16 nodes per layer.

The current test results are:

| Model | Test R2 | Test MAE | Directional Accuracy |
| --- | ---: | ---: | ---: |
| OLS baseline | 0.0002 | 0.0221 | 0.5230 |
| LightGBM | 0.0163 | 0.0220 | 0.5407 |
| Tanh neural network | 0.0146 | 0.0220 | 0.5377 |

LightGBM improves over OLS, but the improvement is limited. The neural network also improves over OLS, but it does not outperform LightGBM. This suggests that there is some non-linear predictive structure, but the current feature set does not contain enough graph-like information to clearly justify a more complex neural architecture.

## Should We Build a Graph Neural Network?

A graph neural network is not strongly justified yet. The current neural network and LightGBM results show only limited predictive gains, and LightGBM remains the best predictive model. A GNN would add implementation complexity and require a clear graph definition: nodes, edges, edge weights, temporal snapshots and target construction.

A GNN could become worthwhile if the thesis explicitly shifts toward network learning. For example, a graph could connect stocks by ETF co-membership, holding-weight overlap, GICS similarity, return correlation or lead-lag relationships. The model could then test whether graph structure improves prediction of receiver abnormal returns beyond the econometric variables already used.

For the current thesis scope, the better next step is not a GNN. The priority should be to finalize the econometric specification, validate robust standard errors, strengthen placebo tests and write a clear interpretation of the heterogeneous ETF results. A GNN can be mentioned as a possible extension rather than implemented as a core requirement.

