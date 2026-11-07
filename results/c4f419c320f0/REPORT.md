# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 3 | allset | 10 | 0.7156 +/- 0.0477 | 0.7144 +/- 0.0550 |
| email-Enron | matched_unseen | 3 | allset_cpa | 10 | 0.7188 +/- 0.0443 | 0.7200 +/- 0.0475 |
| email-Enron | matched_unseen | 3 | allset_cpa_encoder | 10 | 0.7211 +/- 0.0346 | 0.7149 +/- 0.0440 |
| email-Enron | matched_unseen | 3 | allset_size | 10 | 0.6964 +/- 0.0265 | 0.6978 +/- 0.0408 |
| email-Enron | matched_unseen | 3 | deepsets | 10 | 0.6378 +/- 0.0302 | 0.6545 +/- 0.0173 |
| email-Enron | matched_unseen | 3 | gcn | 10 | 0.7122 +/- 0.0459 | 0.7142 +/- 0.0625 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 3 | allset | 0.5024 | 0.6898 | 65.5 | 0/10 |
| email-Enron | matched_unseen | 3 | allset_cpa | 0.5027 | 0.6871 | 57.5 | 0/10 |
| email-Enron | matched_unseen | 3 | allset_cpa_encoder | 0.4804 | 0.6946 | 50.0 | 0/10 |
| email-Enron | matched_unseen | 3 | allset_size | 0.5046 | 0.6798 | 76.0 | 0/10 |
| email-Enron | matched_unseen | 3 | deepsets | 0.4970 | 0.6415 | 36.0 | 0/10 |
| email-Enron | matched_unseen | 3 | gcn | 0.4801 | 0.7084 | 46.0 | 0/10 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 3 | allset_cpa | 10 | 0.0032 +/- 0.0171 | 0.0057 +/- 0.0368 | 4/10 |
| email-Enron | matched_unseen | 3 | allset_cpa_encoder | 10 | 0.0056 +/- 0.0383 | 0.0005 +/- 0.0306 | 8/10 |
| email-Enron | matched_unseen | 3 | allset_size | 10 | -0.0192 +/- 0.0337 | -0.0166 +/- 0.0260 | 2/10 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 3 | allset | 0.7156 | 0.6537 (0.6825) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 3 | allset_cpa | 0.7188 | 0.2531 (0.6825) | 7.0000 | 0.1471 | 0.0451 | 0.0020 |
| email-Enron | matched_unseen | 3 | allset_cpa_encoder | 0.7211 | 0.6613 (0.6825) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 3 | allset_size | 0.6964 | 0.6602 (0.6825) | 0.0000 | 0.2276 | 0.0232 | 0.0019 |
| email-Enron | matched_unseen | 3 | deepsets | 0.6378 | 0.3136 (0.6825) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 3 | gcn | 0.7122 | 0.6613 (0.6825) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
