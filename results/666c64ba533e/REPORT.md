# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 3 | allset | 3 | 0.7325 +/- 0.0241 | 0.7351 +/- 0.0328 |
| email-Enron | matched_unseen | 3 | allset_cpa | 3 | 0.7090 +/- 0.0187 | 0.7076 +/- 0.0357 |
| email-Enron | matched_unseen | 3 | allset_cpa_encoder | 3 | 0.7173 +/- 0.0238 | 0.7293 +/- 0.0431 |
| email-Enron | matched_unseen | 3 | allset_size | 3 | 0.7322 +/- 0.0144 | 0.7360 +/- 0.0189 |
| email-Enron | matched_unseen | 3 | deepsets | 3 | 0.6562 +/- 0.0107 | 0.6779 +/- 0.0176 |
| email-Enron | matched_unseen | 3 | gcn | 3 | 0.7190 +/- 0.0123 | 0.7205 +/- 0.0066 |
| tags-ask-ubuntu | matched_unseen | 3 | allset | 3 | 0.8726 +/- 0.0029 | 0.8899 +/- 0.0021 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_cpa | 3 | 0.8726 +/- 0.0036 | 0.8910 +/- 0.0023 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_cpa_encoder | 3 | 0.8834 +/- 0.0003 | 0.8947 +/- 0.0003 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_size | 3 | 0.8725 +/- 0.0011 | 0.8896 +/- 0.0013 |
| tags-ask-ubuntu | matched_unseen | 3 | deepsets | 3 | 0.8650 +/- 0.0017 | 0.8866 +/- 0.0004 |
| tags-ask-ubuntu | matched_unseen | 3 | gcn | 3 | 0.8815 +/- 0.0008 | 0.8944 +/- 0.0007 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 3 | allset | 0.5120 | 0.6902 | 70.0 | 0/3 |
| email-Enron | matched_unseen | 3 | allset_cpa | 0.5096 | 0.6875 | 50.0 | 0/3 |
| email-Enron | matched_unseen | 3 | allset_cpa_encoder | 0.4348 | 0.6906 | 38.3 | 0/3 |
| email-Enron | matched_unseen | 3 | allset_size | 0.5103 | 0.6809 | 63.3 | 0/3 |
| email-Enron | matched_unseen | 3 | deepsets | 0.5010 | 0.6484 | 26.7 | 0/3 |
| email-Enron | matched_unseen | 3 | gcn | 0.4995 | 0.6801 | 40.0 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 3 | allset | 0.4780 | 0.8728 | 148.3 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_cpa | 0.4898 | 0.8730 | 143.3 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_cpa_encoder | 0.6635 | 0.8859 | 50.0 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_size | 0.4872 | 0.8738 | 148.3 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 3 | deepsets | 0.4976 | 0.8675 | 56.7 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 3 | gcn | 0.4702 | 0.8861 | 105.0 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 3 | allset_cpa | 3 | -0.0235 +/- 0.0260 | -0.0274 +/- 0.0468 | 1/3 |
| email-Enron | matched_unseen | 3 | allset_cpa_encoder | 3 | -0.0152 +/- 0.0208 | -0.0058 +/- 0.0500 | 1/3 |
| email-Enron | matched_unseen | 3 | allset_size | 3 | -0.0003 +/- 0.0282 | 0.0009 +/- 0.0458 | 1/3 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_cpa | 3 | -0.0000 +/- 0.0023 | 0.0011 +/- 0.0011 | 2/3 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_cpa_encoder | 3 | 0.0109 +/- 0.0032 | 0.0048 +/- 0.0018 | 3/3 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_size | 3 | -0.0001 +/- 0.0034 | -0.0003 +/- 0.0030 | 2/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 3 | allset | 0.7325 | 0.6178 (0.6812) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 3 | allset_cpa | 0.7090 | 0.2574 (0.6812) | 7.0000 | 0.1398 | 0.0381 | 0.0059 |
| email-Enron | matched_unseen | 3 | allset_cpa_encoder | 0.7173 | 0.6717 (0.6812) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 3 | allset_size | 0.7322 | 0.6423 (0.6812) | 0.0000 | 0.2325 | 0.0154 | 0.0011 |
| email-Enron | matched_unseen | 3 | deepsets | 0.6562 | 0.3111 (0.6812) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 3 | gcn | 0.7190 | 0.6799 (0.6812) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 3 | allset | 0.8726 | 0.3682 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_cpa | 0.8726 | 0.1090 (0.4581) | 7.0000 | 0.3066 | 0.1912 | -0.0000 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_cpa_encoder | 0.8834 | 0.3988 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 3 | allset_size | 0.8725 | 0.3740 (0.4581) | 0.0000 | 0.1581 | 0.0118 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 3 | deepsets | 0.8650 | 0.1152 (0.4581) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 3 | gcn | 0.8815 | 0.4120 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
