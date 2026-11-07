# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 3 | allset | 3 | 0.7451 +/- 0.0182 | 0.7897 +/- 0.0373 |
| email-Enron | matched_targets | 3 | allset_cpa | 3 | 0.7559 +/- 0.0643 | 0.7859 +/- 0.0397 |
| email-Enron | matched_targets | 3 | allset_cpa_encoder | 3 | 0.7194 +/- 0.0087 | 0.7295 +/- 0.0103 |
| email-Enron | matched_targets | 3 | allset_size | 3 | 0.7314 +/- 0.0184 | 0.7665 +/- 0.0251 |
| email-Enron | matched_targets | 3 | deepsets | 3 | 0.6890 +/- 0.0269 | 0.7313 +/- 0.0280 |
| email-Enron | matched_targets | 3 | gcn | 3 | 0.7343 +/- 0.0337 | 0.7451 +/- 0.0505 |
| tags-ask-ubuntu | matched_targets | 3 | allset | 3 | 0.8770 +/- 0.0016 | 0.8970 +/- 0.0010 |
| tags-ask-ubuntu | matched_targets | 3 | allset_cpa | 3 | 0.8713 +/- 0.0058 | 0.8957 +/- 0.0040 |
| tags-ask-ubuntu | matched_targets | 3 | allset_cpa_encoder | 3 | 0.8880 +/- 0.0017 | 0.8996 +/- 0.0016 |
| tags-ask-ubuntu | matched_targets | 3 | allset_size | 3 | 0.8757 +/- 0.0016 | 0.8962 +/- 0.0006 |
| tags-ask-ubuntu | matched_targets | 3 | deepsets | 3 | 0.8619 +/- 0.0018 | 0.8806 +/- 0.0036 |
| tags-ask-ubuntu | matched_targets | 3 | gcn | 3 | 0.8853 +/- 0.0011 | 0.8981 +/- 0.0011 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 3 | allset | 0.5120 | 0.6825 | 96.7 | 0/3 |
| email-Enron | matched_targets | 3 | allset_cpa | 0.5096 | 0.6977 | 108.3 | 0/3 |
| email-Enron | matched_targets | 3 | allset_cpa_encoder | 0.4348 | 0.6767 | 23.3 | 0/3 |
| email-Enron | matched_targets | 3 | allset_size | 0.5100 | 0.6729 | 71.7 | 0/3 |
| email-Enron | matched_targets | 3 | deepsets | 0.5010 | 0.6465 | 13.3 | 0/3 |
| email-Enron | matched_targets | 3 | gcn | 0.4995 | 0.6923 | 61.7 | 0/3 |
| tags-ask-ubuntu | matched_targets | 3 | allset | 0.4780 | 0.8702 | 150.0 | 0/3 |
| tags-ask-ubuntu | matched_targets | 3 | allset_cpa | 0.4898 | 0.8650 | 150.0 | 0/3 |
| tags-ask-ubuntu | matched_targets | 3 | allset_cpa_encoder | 0.6635 | 0.8850 | 53.3 | 0/3 |
| tags-ask-ubuntu | matched_targets | 3 | allset_size | 0.4855 | 0.8679 | 150.0 | 0/3 |
| tags-ask-ubuntu | matched_targets | 3 | deepsets | 0.4976 | 0.8617 | 10.0 | 0/3 |
| tags-ask-ubuntu | matched_targets | 3 | gcn | 0.4702 | 0.8862 | 80.0 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 3 | allset_cpa | 3 | 0.0108 +/- 0.0530 | -0.0037 +/- 0.0391 | 2/3 |
| email-Enron | matched_targets | 3 | allset_cpa_encoder | 3 | -0.0257 +/- 0.0127 | -0.0601 +/- 0.0271 | 0/3 |
| email-Enron | matched_targets | 3 | allset_size | 3 | -0.0137 +/- 0.0086 | -0.0232 +/- 0.0153 | 0/3 |
| tags-ask-ubuntu | matched_targets | 3 | allset_cpa | 3 | -0.0057 +/- 0.0059 | -0.0013 +/- 0.0032 | 0/3 |
| tags-ask-ubuntu | matched_targets | 3 | allset_cpa_encoder | 3 | 0.0111 +/- 0.0028 | 0.0025 +/- 0.0015 | 3/3 |
| tags-ask-ubuntu | matched_targets | 3 | allset_size | 3 | -0.0013 +/- 0.0026 | -0.0008 +/- 0.0004 | 1/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 3 | allset | 0.7451 | 0.3092 (0.3598) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 3 | allset_cpa | 0.7559 | 0.1201 (0.3598) | 7.0000 | 0.1680 | 0.0616 | -0.0042 |
| email-Enron | matched_targets | 3 | allset_cpa_encoder | 0.7194 | 0.3250 (0.3598) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 3 | allset_size | 0.7314 | 0.3164 (0.3598) | 0.0000 | 0.1091 | 0.0044 | 0.0016 |
| email-Enron | matched_targets | 3 | deepsets | 0.6890 | 0.1005 (0.3598) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 3 | gcn | 0.7343 | 0.3454 (0.3598) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 3 | allset | 0.8770 | 0.1774 (0.2291) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 3 | allset_cpa | 0.8713 | 0.0283 (0.2291) | 7.0000 | 0.3138 | 0.1370 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 3 | allset_cpa_encoder | 0.8880 | 0.1728 (0.2291) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 3 | allset_size | 0.8757 | 0.1757 (0.2291) | 0.0000 | 0.1227 | 0.0052 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 3 | deepsets | 0.8619 | 0.0236 (0.2291) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 3 | gcn | 0.8853 | 0.1696 (0.2291) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
