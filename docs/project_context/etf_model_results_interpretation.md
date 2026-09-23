# ETF-Level Model Results Interpretation

Last updated: 2026-09-23

This document summarizes the ETF-level empirical results after running the expanded model and the additional methodological extensions. It is written as a working interpretation note for the thesis, not as final thesis prose.

The main goal is to understand whether the spillover mechanism is homogeneous across ETFs or whether each ETF shows a different transmission pattern. The evidence shows clear heterogeneity. However, some patterns are stable across most specifications:

- The direct weighted shock channel, `b1_term`, is usually negative in the main event-study and ETF-benchmark specifications.
- The return-comovement channel, `corr_term`, is the most stable positive channel across ETFs and model variants.
- The informational similarity channel, `b5_term`, is generally positive, but its strength differs by ETF.
- The liquidity channel, `b2_term`, is less stable at ETF level than in the pooled model.
- The Fama-French / Carhart factor robustness is the main source of caution because it changes the direct `b1_term` in several ETFs.
- Quantile regression supports the main sign patterns, but it should be treated as a secondary diagnostic because it does not use the full receiver-stock fixed-effects structure.

The results use the following local output files:

- `holdings/results/etf_baseline_summary.csv`
- `holdings/results/etf_benchmark_robustness_summary.csv`
- `holdings/results/etf_factor_robustness_summary.csv`
- `holdings/results/etf_local_projections_summary.csv`
- `holdings/results/etf_quantile_regression_summary.csv`
- `holdings/results/etf_interactions_summary.csv`

## 1. Baseline Expanded Model by ETF

The expanded baseline model includes receiver weight, pre-event return correlation and ETF concentration. This is the closest ETF-level version of the main thesis model.

| ETF | N | Adj. R2 | b1 direct | p(b1) | b2 liquidity | p(b2) | b5 similarity | p(b5) | corr term | p(corr) | HHI term | p(HHI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XME | 82,502 | 0.0911 | -3.1019 | 0.00025 | -0.0082 | 0.9413 | 0.0067 | 0.5300 | 3.4192 | <0.001 | 60.2633 | 0.0115 |
| XLE | 58,693 | 0.0537 | -3.3336 | <0.001 | 0.2683 | 0.1305 | 0.0646 | <0.001 | 2.6396 | <0.001 | 12.2953 | 0.0169 |
| IHE | 142,443 | 0.0650 | -2.3690 | <0.001 | 0.0350 | 0.4543 | 0.0130 | <0.001 | 0.9372 | 0.0029 | 12.8341 | <0.001 |
| XLV | 459,395 | 0.0215 | -6.0862 | <0.001 | -0.3063 | 0.0863 | 0.0328 | <0.001 | 3.3612 | <0.001 | 104.4485 | <0.001 |

### Baseline interpretation

All four ETFs show a negative direct coefficient in the expanded baseline model. This means that, after controlling for the other channels and fixed effects, a larger weighted shock to the origin stock is associated with an opposite-sign abnormal response in receiver stocks. This pattern is strongest in XLV and also large in XLE and XME. IHE has a smaller but still strongly significant negative direct effect.

The return-comovement channel is positive and significant in all ETFs. This is one of the most important findings of the project. Stocks that were already moving together before the event tend to transmit shocks more strongly. This result appears in all sector groups, although it is weaker in IHE than in XME, XLE and XLV.

The informational similarity channel is positive and significant in XLE, IHE and XLV, but not in XME. This suggests that GICS-based similarity captures important transmission links in energy and healthcare-related ETFs, while metals and mining may be less well captured by static sector/industry classifications.

The liquidity channel is not robust at ETF level. It is not significant in XME, XLE or IHE, and it is only marginally negative in XLV. This does not fully contradict the pooled result, but it means the liquidity mechanism is less stable when each ETF is estimated separately.

ETF concentration is positive and significant in all ETFs in the baseline, but the interpretation should be cautious. HHI is an ETF-level variable and can partly proxy structural differences in the fund. It is especially large in XLV and XME.

## 2. LOO/LTO Benchmark Robustness by ETF

The LOO/LTO robustness checks whether the negative direct effect is caused by benchmark absorption. The key versions are:

- `LOO_LTO_all_events`: origin shock uses `ETF(-i)`, receiver abnormal return uses `ETF(-i,-j)`, but keeps the original external-benchmark event sample.
- `LOO_LTO_surviving_events`: same as above, but keeps only shocks that survive the `ETF(-i)` threshold.
- `LOO_receiver_surviving_events`: uses `ETF(-j)` for receiver abnormal returns and keeps only `ETF(-i)` surviving shocks.

| ETF | Specification | N | b1 direct | p(b1) | corr term | p(corr) | b5 similarity | p(b5) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XME | LOO_LTO_all_events | 82,502 | 0.2806 | 0.6983 | 4.3479 | <0.001 | 0.0278 | 0.0017 |
| XME | LOO_LTO_surviving_events | 56,952 | -1.1554 | 0.0840 | 4.2707 | <0.001 | 0.0163 | 0.0539 |
| XME | LOO_receiver_surviving_events | 56,952 | -2.5761 | 0.0003 | 3.7367 | <0.001 | 0.0206 | 0.0210 |
| XLE | LOO_LTO_all_events | 58,693 | -0.1589 | 0.7848 | 3.1617 | <0.001 | 0.1179 | <0.001 |
| XLE | LOO_LTO_surviving_events | 46,655 | -0.5459 | 0.3518 | 3.8620 | <0.001 | 0.0953 | <0.001 |
| XLE | LOO_receiver_surviving_events | 46,655 | -2.2756 | <0.001 | 3.2106 | <0.001 | 0.0938 | <0.001 |
| IHE | LOO_LTO_all_events | 142,443 | -1.6660 | <0.001 | 0.9727 | 0.0038 | 0.0179 | <0.001 |
| IHE | LOO_LTO_surviving_events | 121,190 | -2.0672 | <0.001 | 1.0350 | 0.0019 | 0.0134 | <0.001 |
| IHE | LOO_receiver_surviving_events | 121,190 | -3.9120 | <0.001 | 0.3269 | 0.3181 | 0.0135 | <0.001 |
| XLV | LOO_LTO_all_events | 459,395 | -4.5130 | <0.001 | 3.2012 | <0.001 | 0.0375 | <0.001 |
| XLV | LOO_LTO_surviving_events | 440,013 | -4.5114 | <0.001 | 3.1865 | <0.001 | 0.0352 | <0.001 |
| XLV | LOO_receiver_surviving_events | 440,013 | -5.7626 | <0.001 | 2.9003 | <0.001 | 0.0360 | <0.001 |

### Benchmark robustness interpretation

The LOO/LTO results are important because they test the main concern about benchmark absorption. If the negative direct channel disappeared under synthetic benchmarks, then the baseline `b1_term` would be much less convincing. This does not happen in the pooled model, and it mostly does not happen at ETF level.

XME is the most sensitive ETF. Under the strict LTO all-events version, the direct term becomes positive but not significant. Under surviving events, it becomes negative and marginally significant. Under the `ETF(-j)` receiver version, it becomes clearly negative and significant. This means the XME direct effect is real but more sensitive to the exact benchmark construction. XME is the smallest and most specialized ETF in the sample, so removing one or two stocks from the synthetic benchmark can change the benchmark more strongly than in broader ETFs.

XLE is also sensitive under strict LTO. The direct term is not significant in the LTO versions, but it becomes negative and significant in the `ETF(-j)` receiver version. This suggests that benchmark construction matters for energy, possibly because XLE is highly concentrated and large constituents can dominate the synthetic basket.

IHE is more robust. The direct term stays negative and significant in all LOO/LTO variants. The comovement channel is positive in the LTO variants but loses significance in the `ETF(-j)` receiver version. This suggests that the direct reversal is stable, while the dynamic connectedness channel is somewhat sensitive to how receiver abnormal returns are measured.

XLV is the most robust ETF under LOO/LTO. The direct term remains strongly negative and significant in every version. The similarity and comovement channels also remain positive and significant. This makes XLV the cleanest evidence that the negative direct channel is not simply an artifact of benchmark absorption.

## 3. Local Projections by ETF

Local Projections estimate the model separately at horizons `h=0,...,10`. They show whether the effect is immediate, persistent, or temporary.

### XME dynamics

In XME, the direct effect is weak at `h=0` and `h=1`, becomes statistically significant at `h=2`, reaches its strongest values around `h=3` to `h=8`, and fades by `h=10`.

| Horizon | b1 direct | p(b1) | corr term | p(corr) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -0.5599 | 0.2202 | 1.0739 | <0.001 |
| 1 | -1.0782 | 0.1222 | 1.8947 | <0.001 |
| 2 | -2.1615 | 0.0098 | 2.5868 | <0.001 |
| 3 | -3.6334 | <0.001 | 3.3364 | <0.001 |
| 5 | -4.0746 | <0.001 | 2.8180 | <0.001 |
| 8 | -4.1346 | 0.0011 | 2.8732 | <0.001 |
| 10 | -1.7622 | 0.2308 | 2.9316 | <0.001 |

This pattern suggests a delayed adjustment in XME. The direct channel is not immediate, but it becomes visible after a few days and then weakens again. The comovement channel is positive at all horizons, which is the more stable dynamic mechanism.

### XLE dynamics

XLE shows a clearer negative direct effect from the beginning. The direct coefficient is already significant at `h=0`, becomes stronger through `h=3`, and remains negative through `h=10`.

| Horizon | b1 direct | p(b1) | corr term | p(corr) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -0.7268 | 0.0078 | 0.5643 | <0.001 |
| 1 | -1.2454 | <0.001 | 1.1522 | <0.001 |
| 2 | -2.1826 | <0.001 | 1.8490 | <0.001 |
| 3 | -3.2392 | <0.001 | 2.5726 | <0.001 |
| 5 | -2.9006 | <0.001 | 2.1623 | <0.001 |
| 8 | -2.4557 | 0.0082 | 2.0340 | <0.001 |
| 10 | -2.6705 | 0.0140 | 2.4881 | <0.001 |

The dynamic evidence for XLE supports a persistent reversal or relative-price adjustment channel. The comovement channel is also consistently positive, which means connectedness matters in energy even when the direct effect is negative.

### IHE dynamics

IHE shows a very stable negative direct effect at every horizon. The magnitude increases until around `h=3` to `h=7` and remains significant at `h=10`.

| Horizon | b1 direct | p(b1) | corr term | p(corr) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -0.7835 | <0.001 | 0.2899 | 0.0261 |
| 1 | -1.3568 | <0.001 | 0.4899 | 0.0253 |
| 2 | -1.8385 | <0.001 | 0.7531 | 0.0063 |
| 3 | -2.3170 | <0.001 | 0.9266 | 0.0046 |
| 5 | -2.2913 | <0.001 | 0.8029 | 0.0116 |
| 8 | -2.1963 | <0.001 | 0.4538 | 0.2784 |
| 10 | -1.7006 | 0.0073 | 0.6266 | 0.2361 |

In IHE, the direct channel is stable, while the comovement channel is positive mainly in the short and medium horizons. This may indicate that pharmaceutical spillovers operate more through event-specific relative-price adjustments and fundamental similarity than through long-lasting return comovement.

### XLV dynamics

XLV shows the strongest and most persistent direct negative effect. The coefficient is already large at `h=0`, peaks around the original event window `h=3`, and remains significant through `h=10`.

| Horizon | b1 direct | p(b1) | corr term | p(corr) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -1.7600 | <0.001 | 0.9597 | <0.001 |
| 1 | -3.2063 | <0.001 | 1.7408 | <0.001 |
| 2 | -4.7952 | <0.001 | 2.5839 | <0.001 |
| 3 | -6.1142 | <0.001 | 3.3659 | <0.001 |
| 5 | -5.4949 | <0.001 | 3.3090 | <0.001 |
| 8 | -4.8718 | <0.001 | 3.2770 | <0.001 |
| 10 | -4.5769 | <0.001 | 3.3071 | <0.001 |

XLV is the strongest dynamic case. It supports the idea that broad health care ETF structure creates a persistent relative-price or rebalancing response after constituent shocks. It also shows the most persistent positive comovement channel.

## 4. Fama-French / Carhart Factor Robustness by ETF

The factor robustness uses FF5 plus momentum to compute abnormal returns. This check is not meant to replace the main ETF/sector benchmark design, but it tests whether the findings depend on the expected-return model.

| ETF | Specification | N | b1 direct | p(b1) | corr term | p(corr) | b5 similarity | p(b5) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XME | same events | 80,317 | 1.1260 | 0.2794 | 3.4861 | <0.001 | 0.0431 | 0.0006 |
| XME | factor-surviving events | 55,762 | -0.7561 | 0.4901 | 3.5681 | <0.001 | 0.0260 | 0.0180 |
| XLE | same events | 57,019 | -1.3140 | 0.3900 | 2.0228 | 0.0004 | 0.2843 | <0.001 |
| XLE | factor-surviving events | 30,858 | -4.9119 | 0.0024 | 3.1463 | <0.001 | 0.2152 | <0.001 |
| IHE | same events | 138,751 | 1.7572 | <0.001 | 0.1221 | 0.6995 | 0.0088 | 0.0009 |
| IHE | factor-surviving events | 97,395 | 0.5580 | 0.2116 | 0.2067 | 0.4989 | -0.0018 | 0.4598 |
| XLV | same events | 449,577 | 4.4020 | <0.001 | 2.1620 | <0.001 | 0.1081 | <0.001 |
| XLV | factor-surviving events | 299,711 | 1.8924 | 0.0170 | 2.3131 | <0.001 | 0.0527 | <0.001 |

### Factor robustness interpretation

The factor model is the strongest cautionary result. It shows that the direct `b1_term` is not invariant to how abnormal returns are defined.

XME does not show a significant direct channel under FF5+momentum. The direct coefficient changes sign across versions but remains insignificant. However, the comovement channel stays positive and significant. This means that for XME, factor robustness weakens the direct reversal story but supports connectedness.

XLE becomes negative and significant when only factor-surviving events are kept. This suggests that stronger energy shocks that remain idiosyncratic after FF5+momentum adjustment still produce negative direct spillovers. The similarity and comovement channels remain positive.

IHE is weakened by the factor model. The same-events version gives a positive direct coefficient, and the factor-surviving version is not significant. The comovement channel is not significant under FF5+momentum. Therefore, IHE's strongest evidence comes from the ETF/sector benchmark and LOO/LTO specifications, not from the factor model.

XLV flips positive under FF5+momentum and remains significant. This is important. It suggests that the broad health care direct reversal seen under ETF/sector benchmarks is sensitive to factor-model residualization. However, the comovement and similarity channels remain positive and significant. For XLV, the thesis should be careful: the direct channel is benchmark-model dependent, but connectedness and similarity remain robust.

## 5. Pooled ETF Interaction Model

The pooled ETF interaction model uses XME as the reference ETF and interacts the main channels with ETF dummies. It provides a formal test of heterogeneity.

Important findings from `etf_interactions_summary.csv`:

- The reference direct term for XME is negative but not significant in the pooled interaction model.
- The XLE direct interaction is negative and significant, meaning XLE's direct channel is significantly more negative than XME's.
- The XLV direct interaction is also negative and highly significant, meaning XLV has a much stronger direct reversal than XME.
- The IHE direct interaction is not statistically different from XME in the pooled interaction model.
- The base `corr_term` is positive and significant. The IHE interaction with `corr_term` is negative and significant, meaning the comovement channel is weaker in IHE than in XME. XLE and XLV do not differ as strongly from XME on this channel.
- The XLV concentration interaction is positive and significant, suggesting that concentration-related dynamics are especially important in XLV.

This pooled interaction model supports the view that ETF heterogeneity is real and should not be treated as a minor robustness detail. XLE and XLV behave differently from XME in the direct channel, and IHE has a weaker comovement profile.

## 6. Quantile Regression by ETF

Quantile regression is a secondary diagnostic. It uses ETF-level and year-quarter controls, not the full receiver-stock fixed effects of the main model. Therefore, it should support the interpretation but not replace the main results.

### Main quantile pattern

Across all ETFs, the direct `b1_term` is negative across all quantiles. This means the negative direct association is not only a mean effect. It appears in the lower tail, the median and the upper tail of receiver abnormal returns.

The `corr_term` is positive across all ETFs and all quantiles. This again supports return comovement as the most stable transmission mechanism.

### ETF-specific quantile interpretation

XME has negative `b1_term` across all quantiles, with coefficients around -2.85 to -4.68. The comovement channel is positive across the distribution. This supports the idea that, even though the strict LTO regression is more sensitive, the negative direct relationship appears throughout the conditional distribution.

XLE also has negative `b1_term` across all quantiles, from about -2.42 to -3.08. The similarity channel is strongly positive across quantiles. This suggests energy-sector receiver responses are shaped by both reversal and fundamental relatedness.

IHE has a stable negative direct term around -2.1 to -2.24 across quantiles. The comovement and similarity channels are positive, although smaller than in XLE and XLV. This is consistent with IHE showing a more moderate but stable spillover structure.

XLV has the strongest negative quantile coefficients, from about -4.70 to -6.45. The comovement and similarity channels are also positive. This supports the view that broad health care has the strongest distribution-wide spillover/reversal pattern under the ETF benchmark framework.

## 7. ETF-by-ETF Thesis Interpretation

### XME

XME is the smallest and most specialized ETF in the sample. Its results are more sensitive to benchmark construction than the other ETFs. In the expanded baseline, XME has a negative and significant direct channel, a strong positive comovement channel and a significant concentration channel. However, in the strict LTO benchmark robustness, the direct term becomes weaker and only becomes clearly significant again under the `ETF(-j)` receiver benchmark.

The Local Projections show that XME's direct effect is delayed. It is not significant at the first two horizons, becomes significant around `h=2`, peaks around `h=3` to `h=8`, and fades by `h=10`. This suggests that metals and mining spillovers may take a few days to be incorporated into co-constituent prices.

The most reliable XME result is the positive comovement channel. It is positive in the baseline, LOO/LTO, factor robustness, Local Projections and quantile regression. The direct reversal is present but more sensitive than in XLV and IHE.

### XLE

XLE is a highly concentrated energy ETF. The expanded baseline shows a strong negative direct channel, positive similarity and positive comovement. The LTO benchmark results are mixed: strict LTO weakens the direct term, but the `ETF(-j)` receiver version restores a strong negative direct coefficient. This suggests that benchmark construction matters in energy, probably because a few large firms dominate the ETF basket.

The Local Projections are very clear for XLE. The direct channel is negative from `h=0` and remains negative through `h=10`. This dynamic evidence supports a persistent reversal or relative-price adjustment mechanism.

The factor robustness is also important. Under factor-surviving events, XLE has a large negative and significant direct coefficient. This makes XLE one of the ETFs where the direct reversal survives the factor-model event filter.

Overall, XLE supports the thesis story, but with an important benchmark sensitivity warning. Energy-sector spillovers appear to be driven by a combination of direct reversal, similarity and return comovement.

### IHE

IHE is a pharmaceutical ETF. In the expanded baseline and LOO/LTO robustness, it shows a stable negative direct channel. The direct term remains negative and significant across the synthetic benchmark specifications. This suggests that the negative direct channel is not mainly a benchmark absorption artifact for IHE.

Local Projections also show a stable negative effect from `h=0` to `h=10`. However, the comovement channel weakens at longer horizons. This may mean that IHE transmission is more event-specific and less persistent through return-correlation networks than XME, XLE or XLV.

The factor robustness is the main caution for IHE. Under FF5+momentum, the direct channel becomes positive or insignificant, and the comovement channel is not significant. Therefore, IHE's evidence is strong under ETF/sector benchmark methods but weaker under factor-model residualization.

The thesis should present IHE as evidence of a stable ETF-benchmark spillover mechanism, but also as an example where factor-model robustness changes the interpretation.

### XLV

XLV is the broad health care ETF and provides the strongest evidence in the ETF-benchmark framework. The expanded baseline shows the largest negative direct coefficient, strong positive comovement, positive similarity and a large concentration term.

LOO/LTO robustness strongly supports the negative direct effect in XLV. The direct term remains negative and highly significant under all synthetic benchmark variants. This makes XLV the cleanest case against the idea that the negative `b1_term` is only benchmark absorption.

Local Projections also show the strongest dynamic pattern. The direct coefficient is negative and significant from `h=0` to `h=10`, peaks around the original event window and remains large afterwards. The comovement channel is positive and persistent.

The factor robustness gives a different view: under FF5+momentum, the direct term becomes positive and significant. This means that XLV's direct reversal is strongly tied to the ETF/sector abnormal-return definition. However, similarity and comovement remain positive under the factor model.

For the thesis, XLV should be presented as the strongest ETF-benchmark evidence of intra-ETF reversal and connectedness, but also as a case where expected-return modeling matters for the direct channel.

## 8. Main Cross-ETF Conclusions

1. The negative direct channel is not a simple benchmark artifact. LOO/LTO robustness supports it strongly in IHE and XLV, and partially in XME and XLE depending on the synthetic receiver benchmark.
2. Local Projections support a dynamic reversal pattern. The direct effect generally strengthens up to the original `h=3` window and then either persists or fades depending on the ETF.
3. Return comovement is the most stable channel. `corr_term` is positive in almost every model and ETF, including the factor robustness checks.
4. Static informational similarity is also important, especially in XLE, IHE and XLV. It is weaker in XME under the baseline but appears in some robustness checks.
5. Liquidity is less stable at ETF level. The pooled liquidity result should be discussed, but the ETF-level results show that it is not equally strong across funds.
6. Factor-model robustness is the main caution. It changes the direct channel in IHE and XLV and weakens it in the pooled factor-surviving specification. Therefore, the thesis should distinguish between robustness to ETF benchmark construction and robustness to expected-return model choice.
7. ETF heterogeneity is a substantive result. XLE and XLV have stronger direct reversal patterns than XME in the pooled interaction model, while IHE has weaker comovement dynamics.

## 9. Suggested Thesis Wording

A possible thesis interpretation is:

The empirical results show that intra-ETF shock transmission is not homogeneous across funds. Under the main ETF/sector benchmark framework, shocks to one constituent tend to be associated with opposite-sign abnormal returns in other constituents, especially in XLV, XLE and IHE. This negative direct channel is not eliminated by leave-one-out or leave-two-out synthetic ETF benchmarks, which suggests that it is not only a mechanical benchmark absorption effect. Instead, it is consistent with short-term relative-price adjustment, substitution, or ETF-level rebalancing after constituent shocks.

At the same time, the most stable positive mechanism is pre-event return comovement. Across ETFs, horizons and robustness checks, receiver stocks that moved more closely with the shocked stock before the event tend to show stronger abnormal responses. This supports a connectedness-based interpretation of intra-ETF transmission.

The results also show that the direct channel is sensitive to the expected-return model. When abnormal returns are recomputed using Fama-French factors and momentum, the direct coefficient changes in several ETFs, especially IHE and XLV. Therefore, the direct reversal should be presented as robust to ETF benchmark absorption checks but not fully invariant to all abnormal-return specifications. By contrast, the comovement channel is more stable and should be treated as one of the strongest findings of the thesis.
