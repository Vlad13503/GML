# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 2 | allset | 3 | 0.6760 +/- 0.0460 | 0.7039 +/- 0.0243 |
| email-Enron | matched_targets | 2 | allset_cpa | 3 | 0.6615 +/- 0.0216 | 0.6973 +/- 0.0172 |
| email-Enron | matched_targets | 2 | allset_cpa_encoder | 3 | 0.6690 +/- 0.0363 | 0.6777 +/- 0.0207 |
| email-Enron | matched_targets | 2 | allset_size | 3 | 0.6631 +/- 0.0327 | 0.6727 +/- 0.0214 |
| email-Enron | matched_targets | 2 | deepsets | 3 | 0.6609 +/- 0.0114 | 0.6729 +/- 0.0138 |
| email-Enron | matched_targets | 2 | gcn | 3 | 0.6716 +/- 0.0212 | 0.6815 +/- 0.0201 |
| tags-ask-ubuntu | matched_targets | 2 | allset | 3 | 0.8370 +/- 0.0052 | 0.8606 +/- 0.0022 |
| tags-ask-ubuntu | matched_targets | 2 | allset_cpa | 3 | 0.8427 +/- 0.0019 | 0.8644 +/- 0.0018 |
| tags-ask-ubuntu | matched_targets | 2 | allset_cpa_encoder | 3 | 0.8692 +/- 0.0016 | 0.8801 +/- 0.0011 |
| tags-ask-ubuntu | matched_targets | 2 | allset_size | 3 | 0.8393 +/- 0.0023 | 0.8624 +/- 0.0007 |
| tags-ask-ubuntu | matched_targets | 2 | deepsets | 3 | 0.8365 +/- 0.0038 | 0.8632 +/- 0.0011 |
| tags-ask-ubuntu | matched_targets | 2 | gcn | 3 | 0.8551 +/- 0.0023 | 0.8686 +/- 0.0025 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 2 | allset | 0.5099 | 0.6192 | 61.7 | 0/3 |
| email-Enron | matched_targets | 2 | allset_cpa | 0.5079 | 0.6203 | 46.7 | 0/3 |
| email-Enron | matched_targets | 2 | allset_cpa_encoder | 0.4598 | 0.6189 | 50.0 | 0/3 |
| email-Enron | matched_targets | 2 | allset_size | 0.5106 | 0.6154 | 48.3 | 0/3 |
| email-Enron | matched_targets | 2 | deepsets | 0.5181 | 0.5965 | 18.3 | 0/3 |
| email-Enron | matched_targets | 2 | gcn | 0.5176 | 0.6301 | 30.0 | 0/3 |
| tags-ask-ubuntu | matched_targets | 2 | allset | 0.4779 | 0.8164 | 105.0 | 0/3 |
| tags-ask-ubuntu | matched_targets | 2 | allset_cpa | 0.4779 | 0.8259 | 128.3 | 0/3 |
| tags-ask-ubuntu | matched_targets | 2 | allset_cpa_encoder | 0.7819 | 0.8600 | 15.0 | 0/3 |
| tags-ask-ubuntu | matched_targets | 2 | allset_size | 0.4773 | 0.8200 | 113.3 | 0/3 |
| tags-ask-ubuntu | matched_targets | 2 | deepsets | 0.4962 | 0.8285 | 10.0 | 0/3 |
| tags-ask-ubuntu | matched_targets | 2 | gcn | 0.4893 | 0.8653 | 53.3 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 2 | allset_cpa | 3 | -0.0145 +/- 0.0324 | -0.0065 +/- 0.0164 | 1/3 |
| email-Enron | matched_targets | 2 | allset_cpa_encoder | 3 | -0.0070 +/- 0.0175 | -0.0262 +/- 0.0365 | 1/3 |
| email-Enron | matched_targets | 2 | allset_size | 3 | -0.0128 +/- 0.0211 | -0.0312 +/- 0.0426 | 1/3 |
| tags-ask-ubuntu | matched_targets | 2 | allset_cpa | 3 | 0.0057 +/- 0.0058 | 0.0038 +/- 0.0027 | 3/3 |
| tags-ask-ubuntu | matched_targets | 2 | allset_cpa_encoder | 3 | 0.0322 +/- 0.0046 | 0.0195 +/- 0.0029 | 3/3 |
| tags-ask-ubuntu | matched_targets | 2 | allset_size | 3 | 0.0023 +/- 0.0029 | 0.0018 +/- 0.0015 | 2/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_targets | 2 | allset | 0.6760 | 0.2628 (0.2687) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 2 | allset_cpa | 0.6615 | 0.0810 (0.2687) | 7.0000 | 0.2079 | 0.0304 | 0.0043 |
| email-Enron | matched_targets | 2 | allset_cpa_encoder | 0.6690 | 0.2689 (0.2687) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 2 | allset_size | 0.6631 | 0.2785 (0.2687) | 0.0000 | 0.1194 | 0.0024 | 0.0016 |
| email-Enron | matched_targets | 2 | deepsets | 0.6609 | 0.0849 (0.2687) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_targets | 2 | gcn | 0.6716 | 0.2628 (0.2687) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 2 | allset | 0.8370 | 0.1583 (0.1811) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 2 | allset_cpa | 0.8427 | 0.0407 (0.1811) | 7.0000 | 0.2378 | 0.2148 | 0.0007 |
| tags-ask-ubuntu | matched_targets | 2 | allset_cpa_encoder | 0.8692 | 0.1752 (0.1811) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 2 | allset_size | 0.8393 | 0.1743 (0.1811) | 0.0000 | 0.1232 | 0.0100 | -0.0000 |
| tags-ask-ubuntu | matched_targets | 2 | deepsets | 0.8365 | 0.0257 (0.1811) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_targets | 2 | gcn | 0.8551 | 0.1735 (0.1811) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
