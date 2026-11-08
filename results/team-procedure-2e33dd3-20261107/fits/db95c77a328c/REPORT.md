# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 3 | 0.7598 +/- 0.0530 | 0.7453 +/- 0.0624 |
| email-Enron | order | 3 | allset_cpa | 3 | 0.7554 +/- 0.0215 | 0.7664 +/- 0.0182 |
| email-Enron | order | 3 | allset_cpa_both | 3 | 0.7569 +/- 0.0154 | 0.7602 +/- 0.0310 |
| email-Enron | order | 3 | allset_cpa_encoder | 3 | 0.7478 +/- 0.0073 | 0.7561 +/- 0.0188 |
| tags-ask-ubuntu | order | 3 | allset | 3 | 0.8742 +/- 0.0033 | 0.8912 +/- 0.0036 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | 0.8722 +/- 0.0064 | 0.8911 +/- 0.0028 |
| tags-ask-ubuntu | order | 3 | allset_cpa_both | 3 | 0.8823 +/- 0.0014 | 0.8945 +/- 0.0013 |
| tags-ask-ubuntu | order | 3 | allset_cpa_encoder | 3 | 0.8838 +/- 0.0009 | 0.8962 +/- 0.0011 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 0.5341 | 0.6881 | 71.7 | 0/3 |
| email-Enron | order | 3 | allset_cpa | 0.5163 | 0.6982 | 103.3 | 0/3 |
| email-Enron | order | 3 | allset_cpa_both | 0.5054 | 0.7135 | 40.0 | 0/3 |
| email-Enron | order | 3 | allset_cpa_encoder | 0.5065 | 0.7109 | 53.3 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset | 0.4781 | 0.8712 | 146.7 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.4894 | 0.8705 | 150.0 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa_both | 0.5319 | 0.8835 | 55.0 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa_encoder | 0.6614 | 0.8833 | 55.0 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset_cpa | 3 | -0.0044 +/- 0.0315 | 0.0211 +/- 0.0594 | 2/3 |
| email-Enron | order | 3 | allset_cpa_encoder | 3 | -0.0121 +/- 0.0461 | 0.0108 +/- 0.0537 | 2/3 |
| email-Enron | order | 3 | allset_cpa_both | 3 | -0.0029 +/- 0.0501 | 0.0149 +/- 0.0392 | 2/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | -0.0019 +/- 0.0073 | -0.0001 +/- 0.0044 | 1/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa_encoder | 3 | 0.0096 +/- 0.0036 | 0.0049 +/- 0.0034 | 3/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa_both | 3 | 0.0081 +/- 0.0024 | 0.0033 +/- 0.0022 | 3/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 0.7598 | 0.6830 (0.6843) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 3 | allset_cpa | 0.7554 | 0.2313 (0.6843) | 7.0000 | 0.1419 | 0.0819 | -0.0020 |
| email-Enron | order | 3 | allset_cpa_both | 0.7569 | 0.2186 (0.6843) | 7.0000 | 0.1853 | 0.0242 | 0.0036 |
| email-Enron | order | 3 | allset_cpa_encoder | 0.7478 | 0.6543 (0.6843) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | allset | 0.8742 | 0.3871 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.8722 | 0.1168 (0.4581) | 7.0000 | 0.2989 | 0.2316 | -0.0000 |
| tags-ask-ubuntu | order | 3 | allset_cpa_both | 0.8823 | 0.1216 (0.4581) | 7.0000 | 0.2365 | 0.0889 | -0.0001 |
| tags-ask-ubuntu | order | 3 | allset_cpa_encoder | 0.8838 | 0.4002 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
