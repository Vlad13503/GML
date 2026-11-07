# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_all | 3 | allset | 10 | 0.7816 +/- 0.0405 | 0.7910 +/- 0.0525 |
| email-Enron | matched_all | 3 | allset_cpa | 10 | 0.7646 +/- 0.0517 | 0.7739 +/- 0.0544 |
| email-Enron | matched_all | 3 | allset_cpa_encoder | 10 | 0.7811 +/- 0.0376 | 0.7945 +/- 0.0377 |
| email-Enron | matched_all | 3 | allset_size | 10 | 0.7988 +/- 0.0483 | 0.8173 +/- 0.0589 |
| email-Enron | matched_all | 3 | deepsets | 10 | 0.6734 +/- 0.0329 | 0.6926 +/- 0.0332 |
| email-Enron | matched_all | 3 | gcn | 10 | 0.7716 +/- 0.0643 | 0.7873 +/- 0.0867 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 3 | allset | 0.5044 | 0.7166 | 80.0 | 0/10 |
| email-Enron | matched_all | 3 | allset_cpa | 0.5032 | 0.7186 | 91.0 | 0/10 |
| email-Enron | matched_all | 3 | allset_cpa_encoder | 0.4757 | 0.7169 | 68.5 | 0/10 |
| email-Enron | matched_all | 3 | allset_size | 0.5059 | 0.7307 | 89.5 | 0/10 |
| email-Enron | matched_all | 3 | deepsets | 0.4970 | 0.6371 | 25.5 | 0/10 |
| email-Enron | matched_all | 3 | gcn | 0.4775 | 0.7245 | 66.0 | 0/10 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 3 | allset_cpa | 10 | -0.0170 +/- 0.0532 | -0.0170 +/- 0.0674 | 3/10 |
| email-Enron | matched_all | 3 | allset_cpa_encoder | 10 | -0.0005 +/- 0.0341 | 0.0035 +/- 0.0456 | 6/10 |
| email-Enron | matched_all | 3 | allset_size | 10 | 0.0172 +/- 0.0481 | 0.0264 +/- 0.0645 | 6/10 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 3 | allset | 0.7816 | 0.4524 (0.5298) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 3 | allset_cpa | 0.7646 | 0.1322 (0.5298) | 7.0000 | 0.1741 | 0.0904 | 0.0043 |
| email-Enron | matched_all | 3 | allset_cpa_encoder | 0.7811 | 0.4195 (0.5298) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 3 | allset_size | 0.7988 | 0.4565 (0.5298) | 0.0000 | 0.1130 | 0.0259 | 0.0012 |
| email-Enron | matched_all | 3 | deepsets | 0.6734 | 0.1213 (0.5298) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 3 | gcn | 0.7716 | 0.4794 (0.5298) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
