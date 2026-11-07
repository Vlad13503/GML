# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 2 | allset | 10 | 0.6867 +/- 0.0328 | 0.7075 +/- 0.0303 |
| email-Enron | matched_targets | 2 | allset_cpa | 10 | 0.6598 +/- 0.0352 | 0.6785 +/- 0.0381 |
| email-Enron | matched_targets | 2 | allset_cpa_encoder | 10 | 0.6853 +/- 0.0272 | 0.6881 +/- 0.0302 |
| email-Enron | matched_targets | 2 | allset_size | 10 | 0.6598 +/- 0.0480 | 0.6864 +/- 0.0487 |
| email-Enron | matched_targets | 2 | deepsets | 10 | 0.6582 +/- 0.0144 | 0.6740 +/- 0.0176 |
| email-Enron | matched_targets | 2 | gcn | 10 | 0.6943 +/- 0.0313 | 0.6922 +/- 0.0435 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 2 | allset | 0.5048 | 0.6396 | 57.0 | 0/10 |
| email-Enron | matched_targets | 2 | allset_cpa | 0.5076 | 0.6420 | 60.0 | 0/10 |
| email-Enron | matched_targets | 2 | allset_cpa_encoder | 0.4727 | 0.6452 | 29.5 | 0/10 |
| email-Enron | matched_targets | 2 | allset_size | 0.5014 | 0.6412 | 63.0 | 0/10 |
| email-Enron | matched_targets | 2 | deepsets | 0.5194 | 0.6182 | 28.0 | 0/10 |
| email-Enron | matched_targets | 2 | gcn | 0.4863 | 0.6714 | 46.5 | 0/10 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 2 | allset_cpa | 10 | -0.0269 +/- 0.0299 | -0.0290 +/- 0.0313 | 2/10 |
| email-Enron | matched_targets | 2 | allset_cpa_encoder | 10 | -0.0014 +/- 0.0263 | -0.0194 +/- 0.0168 | 6/10 |
| email-Enron | matched_targets | 2 | allset_size | 10 | -0.0269 +/- 0.0371 | -0.0211 +/- 0.0394 | 1/10 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 2 | allset | 0.6867 | 0.2676 (0.2766) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 2 | allset_cpa | 0.6598 | 0.0765 (0.2766) | 7.0000 | 0.1815 | 0.0228 | 0.0119 |
| email-Enron | matched_targets | 2 | allset_cpa_encoder | 0.6853 | 0.2758 (0.2766) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 2 | allset_size | 0.6598 | 0.2749 (0.2766) | 0.0000 | 0.1186 | 0.0055 | 0.0158 |
| email-Enron | matched_targets | 2 | deepsets | 0.6582 | 0.0903 (0.2766) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 2 | gcn | 0.6943 | 0.2801 (0.2766) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
