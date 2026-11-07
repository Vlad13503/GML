# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_all | 2 | allset | 3 | 0.7195 +/- 0.0383 | 0.7502 +/- 0.0275 |
| email-Enron | matched_all | 2 | allset_cpa | 3 | 0.6903 +/- 0.0346 | 0.7004 +/- 0.0563 |
| email-Enron | matched_all | 2 | allset_cpa_encoder | 3 | 0.7288 +/- 0.0769 | 0.7489 +/- 0.0691 |
| email-Enron | matched_all | 2 | allset_size | 3 | 0.6608 +/- 0.1136 | 0.6834 +/- 0.1122 |
| email-Enron | matched_all | 2 | deepsets | 3 | 0.6568 +/- 0.0380 | 0.6780 +/- 0.0337 |
| email-Enron | matched_all | 2 | gcn | 3 | 0.7538 +/- 0.0262 | 0.7737 +/- 0.0419 |
| tags-ask-ubuntu | matched_all | 2 | allset | 3 | 0.8624 +/- 0.0131 | 0.8865 +/- 0.0070 |
| tags-ask-ubuntu | matched_all | 2 | allset_cpa | 3 | 0.8647 +/- 0.0093 | 0.8893 +/- 0.0062 |
| tags-ask-ubuntu | matched_all | 2 | allset_cpa_encoder | 3 | 0.8821 +/- 0.0012 | 0.8935 +/- 0.0013 |
| tags-ask-ubuntu | matched_all | 2 | allset_size | 3 | 0.8675 +/- 0.0054 | 0.8886 +/- 0.0030 |
| tags-ask-ubuntu | matched_all | 2 | deepsets | 3 | 0.8553 +/- 0.0093 | 0.8807 +/- 0.0064 |
| tags-ask-ubuntu | matched_all | 2 | gcn | 3 | 0.8660 +/- 0.0018 | 0.8795 +/- 0.0019 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 2 | allset | 0.5172 | 0.6582 | 88.3 | 0/3 |
| email-Enron | matched_all | 2 | allset_cpa | 0.5174 | 0.6418 | 63.3 | 0/3 |
| email-Enron | matched_all | 2 | allset_cpa_encoder | 0.4665 | 0.6608 | 93.3 | 0/3 |
| email-Enron | matched_all | 2 | allset_size | 0.5176 | 0.6395 | 56.7 | 0/3 |
| email-Enron | matched_all | 2 | deepsets | 0.5181 | 0.6175 | 48.3 | 0/3 |
| email-Enron | matched_all | 2 | gcn | 0.5091 | 0.6810 | 88.3 | 0/3 |
| tags-ask-ubuntu | matched_all | 2 | allset | 0.4760 | 0.8351 | 146.7 | 0/3 |
| tags-ask-ubuntu | matched_all | 2 | allset_cpa | 0.4775 | 0.8350 | 150.0 | 0/3 |
| tags-ask-ubuntu | matched_all | 2 | allset_cpa_encoder | 0.5907 | 0.8657 | 60.0 | 0/3 |
| tags-ask-ubuntu | matched_all | 2 | allset_size | 0.4755 | 0.8433 | 150.0 | 0/3 |
| tags-ask-ubuntu | matched_all | 2 | deepsets | 0.4962 | 0.8489 | 45.0 | 0/3 |
| tags-ask-ubuntu | matched_all | 2 | gcn | 0.4780 | 0.8719 | 51.7 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 2 | allset_cpa | 3 | -0.0293 +/- 0.0690 | -0.0498 +/- 0.0729 | 2/3 |
| email-Enron | matched_all | 2 | allset_cpa_encoder | 3 | 0.0092 +/- 0.1136 | -0.0013 +/- 0.0928 | 1/3 |
| email-Enron | matched_all | 2 | allset_size | 3 | -0.0587 +/- 0.1510 | -0.0668 +/- 0.1275 | 1/3 |
| tags-ask-ubuntu | matched_all | 2 | allset_cpa | 3 | 0.0024 +/- 0.0085 | 0.0028 +/- 0.0043 | 2/3 |
| tags-ask-ubuntu | matched_all | 2 | allset_cpa_encoder | 3 | 0.0197 +/- 0.0138 | 0.0070 +/- 0.0079 | 3/3 |
| tags-ask-ubuntu | matched_all | 2 | allset_size | 3 | 0.0051 +/- 0.0111 | 0.0021 +/- 0.0068 | 1/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 2 | allset | 0.7195 | 0.2723 (0.3025) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 2 | allset_cpa | 0.6903 | 0.0852 (0.3025) | 7.0000 | 0.1467 | 0.0293 | 0.0203 |
| email-Enron | matched_all | 2 | allset_cpa_encoder | 0.7288 | 0.2838 (0.3025) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 2 | allset_size | 0.6608 | 0.2944 (0.3025) | 0.0000 | 0.1129 | 0.0069 | 0.0532 |
| email-Enron | matched_all | 2 | deepsets | 0.6568 | 0.0965 (0.3025) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 2 | gcn | 0.7538 | 0.2819 (0.3025) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_all | 2 | allset | 0.8624 | 0.1543 (0.1811) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_all | 2 | allset_cpa | 0.8647 | 0.0303 (0.1811) | 7.0000 | 0.2797 | 0.0773 | -0.0001 |
| tags-ask-ubuntu | matched_all | 2 | allset_cpa_encoder | 0.8821 | 0.1505 (0.1811) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_all | 2 | allset_size | 0.8675 | 0.1569 (0.1811) | 0.0000 | 0.1223 | 0.0047 | 0.0000 |
| tags-ask-ubuntu | matched_all | 2 | deepsets | 0.8553 | 0.0256 (0.1811) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_all | 2 | gcn | 0.8660 | 0.1714 (0.1811) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
