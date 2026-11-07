# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 3 | allset | 10 | 0.7644 +/- 0.0470 | 0.7763 +/- 0.0530 |
| email-Enron | matched_targets | 3 | allset_cpa | 10 | 0.7869 +/- 0.0449 | 0.8007 +/- 0.0485 |
| email-Enron | matched_targets | 3 | allset_cpa_encoder | 10 | 0.7845 +/- 0.0587 | 0.8046 +/- 0.0572 |
| email-Enron | matched_targets | 3 | allset_size | 10 | 0.7613 +/- 0.0270 | 0.7801 +/- 0.0326 |
| email-Enron | matched_targets | 3 | deepsets | 10 | 0.7044 +/- 0.0198 | 0.7175 +/- 0.0227 |
| email-Enron | matched_targets | 3 | gcn | 10 | 0.7367 +/- 0.0428 | 0.7444 +/- 0.0470 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 3 | allset | 0.5024 | 0.6972 | 71.0 | 0/10 |
| email-Enron | matched_targets | 3 | allset_cpa | 0.5027 | 0.7030 | 84.0 | 0/10 |
| email-Enron | matched_targets | 3 | allset_cpa_encoder | 0.4804 | 0.7136 | 73.5 | 0/10 |
| email-Enron | matched_targets | 3 | allset_size | 0.5063 | 0.6880 | 74.5 | 0/10 |
| email-Enron | matched_targets | 3 | deepsets | 0.4970 | 0.6500 | 25.5 | 0/10 |
| email-Enron | matched_targets | 3 | gcn | 0.4801 | 0.6987 | 55.0 | 0/10 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 3 | allset_cpa | 10 | 0.0224 +/- 0.0181 | 0.0243 +/- 0.0281 | 9/10 |
| email-Enron | matched_targets | 3 | allset_cpa_encoder | 10 | 0.0201 +/- 0.0329 | 0.0283 +/- 0.0454 | 8/10 |
| email-Enron | matched_targets | 3 | allset_size | 10 | -0.0031 +/- 0.0368 | 0.0037 +/- 0.0401 | 5/10 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 3 | allset | 0.7644 | 0.3334 (0.3714) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 3 | allset_cpa | 0.7869 | 0.0973 (0.3714) | 7.0000 | 0.1779 | 0.0489 | -0.0026 |
| email-Enron | matched_targets | 3 | allset_cpa_encoder | 0.7845 | 0.3291 (0.3714) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 3 | allset_size | 0.7613 | 0.3257 (0.3714) | 0.0000 | 0.1102 | 0.0068 | -0.0001 |
| email-Enron | matched_targets | 3 | deepsets | 0.7044 | 0.1056 (0.3714) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 3 | gcn | 0.7367 | 0.3413 (0.3714) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
