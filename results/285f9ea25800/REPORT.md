# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_all | 2 | allset | 10 | 0.7153 +/- 0.0351 | 0.7035 +/- 0.0499 |
| email-Enron | matched_all | 2 | allset_cpa | 10 | 0.7369 +/- 0.0503 | 0.7440 +/- 0.0744 |
| email-Enron | matched_all | 2 | allset_cpa_encoder | 10 | 0.7347 +/- 0.0490 | 0.7373 +/- 0.0589 |
| email-Enron | matched_all | 2 | allset_size | 10 | 0.7347 +/- 0.0539 | 0.7367 +/- 0.0668 |
| email-Enron | matched_all | 2 | deepsets | 10 | 0.6596 +/- 0.0140 | 0.6750 +/- 0.0177 |
| email-Enron | matched_all | 2 | gcn | 10 | 0.7521 +/- 0.0681 | 0.7557 +/- 0.0784 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 2 | allset | 0.5093 | 0.6837 | 46.0 | 0/10 |
| email-Enron | matched_all | 2 | allset_cpa | 0.5110 | 0.6934 | 78.0 | 0/10 |
| email-Enron | matched_all | 2 | allset_cpa_encoder | 0.4964 | 0.6940 | 45.5 | 0/10 |
| email-Enron | matched_all | 2 | allset_size | 0.5041 | 0.6964 | 56.5 | 0/10 |
| email-Enron | matched_all | 2 | deepsets | 0.5194 | 0.6258 | 22.0 | 0/10 |
| email-Enron | matched_all | 2 | gcn | 0.4755 | 0.7081 | 65.5 | 0/10 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 2 | allset_cpa | 10 | 0.0216 +/- 0.0551 | 0.0405 +/- 0.0691 | 6/10 |
| email-Enron | matched_all | 2 | allset_cpa_encoder | 10 | 0.0194 +/- 0.0395 | 0.0338 +/- 0.0546 | 6/10 |
| email-Enron | matched_all | 2 | allset_size | 10 | 0.0194 +/- 0.0381 | 0.0332 +/- 0.0438 | 6/10 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 2 | allset | 0.7153 | 0.3066 (0.3306) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 2 | allset_cpa | 0.7369 | 0.0928 (0.3306) | 7.0000 | 0.1547 | 0.0480 | 0.0063 |
| email-Enron | matched_all | 2 | allset_cpa_encoder | 0.7347 | 0.2968 (0.3306) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 2 | allset_size | 0.7347 | 0.3018 (0.3306) | 0.0000 | 0.1227 | 0.0064 | 0.0025 |
| email-Enron | matched_all | 2 | deepsets | 0.6596 | 0.0959 (0.3306) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 2 | gcn | 0.7521 | 0.3101 (0.3306) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
