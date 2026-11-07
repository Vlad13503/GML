# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_all | 3 | allset | 3 | 0.7532 +/- 0.0224 | 0.7532 +/- 0.0303 |
| email-Enron | matched_all | 3 | allset_cpa | 3 | 0.7927 +/- 0.0496 | 0.8154 +/- 0.0451 |
| email-Enron | matched_all | 3 | allset_cpa_encoder | 3 | 0.7727 +/- 0.0426 | 0.7817 +/- 0.0655 |
| email-Enron | matched_all | 3 | allset_size | 3 | 0.7403 +/- 0.0207 | 0.7403 +/- 0.0277 |
| email-Enron | matched_all | 3 | deepsets | 3 | 0.6805 +/- 0.0185 | 0.7063 +/- 0.0060 |
| email-Enron | matched_all | 3 | gcn | 3 | 0.7144 +/- 0.0053 | 0.7220 +/- 0.0367 |
| tags-ask-ubuntu | matched_all | 3 | allset | 3 | 0.8775 +/- 0.0095 | 0.8991 +/- 0.0053 |
| tags-ask-ubuntu | matched_all | 3 | allset_cpa | 3 | 0.8754 +/- 0.0089 | 0.8998 +/- 0.0050 |
| tags-ask-ubuntu | matched_all | 3 | allset_cpa_encoder | 3 | 0.8908 +/- 0.0027 | 0.9026 +/- 0.0027 |
| tags-ask-ubuntu | matched_all | 3 | allset_size | 3 | 0.8760 +/- 0.0010 | 0.8990 +/- 0.0006 |
| tags-ask-ubuntu | matched_all | 3 | deepsets | 3 | 0.8747 +/- 0.0037 | 0.8958 +/- 0.0034 |
| tags-ask-ubuntu | matched_all | 3 | gcn | 3 | 0.8965 +/- 0.0154 | 0.9093 +/- 0.0142 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 3 | allset | 0.5177 | 0.6987 | 63.3 | 0/3 |
| email-Enron | matched_all | 3 | allset_cpa | 0.5124 | 0.7302 | 86.7 | 0/3 |
| email-Enron | matched_all | 3 | allset_cpa_encoder | 0.4653 | 0.7394 | 90.0 | 0/3 |
| email-Enron | matched_all | 3 | allset_size | 0.5133 | 0.6861 | 61.7 | 0/3 |
| email-Enron | matched_all | 3 | deepsets | 0.5010 | 0.6581 | 43.3 | 0/3 |
| email-Enron | matched_all | 3 | gcn | 0.5074 | 0.7088 | 78.3 | 0/3 |
| tags-ask-ubuntu | matched_all | 3 | allset | 0.4787 | 0.8686 | 148.3 | 0/3 |
| tags-ask-ubuntu | matched_all | 3 | allset_cpa | 0.4909 | 0.8658 | 148.3 | 0/3 |
| tags-ask-ubuntu | matched_all | 3 | allset_cpa_encoder | 0.6203 | 0.8877 | 61.7 | 0/3 |
| tags-ask-ubuntu | matched_all | 3 | allset_size | 0.4866 | 0.8684 | 150.0 | 0/3 |
| tags-ask-ubuntu | matched_all | 3 | deepsets | 0.4976 | 0.8734 | 66.7 | 0/3 |
| tags-ask-ubuntu | matched_all | 3 | gcn | 0.4685 | 0.8935 | 138.3 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 3 | allset_cpa | 3 | 0.0395 +/- 0.0720 | 0.0623 +/- 0.0724 | 2/3 |
| email-Enron | matched_all | 3 | allset_cpa_encoder | 3 | 0.0196 +/- 0.0307 | 0.0285 +/- 0.0683 | 2/3 |
| email-Enron | matched_all | 3 | allset_size | 3 | -0.0129 +/- 0.0389 | -0.0129 +/- 0.0529 | 1/3 |
| tags-ask-ubuntu | matched_all | 3 | allset_cpa | 3 | -0.0022 +/- 0.0068 | 0.0007 +/- 0.0033 | 1/3 |
| tags-ask-ubuntu | matched_all | 3 | allset_cpa_encoder | 3 | 0.0133 +/- 0.0122 | 0.0035 +/- 0.0080 | 3/3 |
| tags-ask-ubuntu | matched_all | 3 | allset_size | 3 | -0.0016 +/- 0.0090 | -0.0001 +/- 0.0051 | 1/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_all | 3 | allset | 0.7532 | 0.4277 (0.5272) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 3 | allset_cpa | 0.7927 | 0.1377 (0.5272) | 7.0000 | 0.1082 | 0.1346 | -0.0073 |
| email-Enron | matched_all | 3 | allset_cpa_encoder | 0.7727 | 0.4904 (0.5272) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 3 | allset_size | 0.7403 | 0.4380 (0.5272) | 0.0000 | 0.1119 | 0.0086 | 0.0090 |
| email-Enron | matched_all | 3 | deepsets | 0.6805 | 0.1208 (0.5272) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_all | 3 | gcn | 0.7144 | 0.4830 (0.5272) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_all | 3 | allset | 0.8775 | 0.1751 (0.2291) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_all | 3 | allset_cpa | 0.8754 | 0.0268 (0.2291) | 7.0000 | 0.3137 | 0.1559 | -0.0004 |
| tags-ask-ubuntu | matched_all | 3 | allset_cpa_encoder | 0.8908 | 0.1745 (0.2291) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_all | 3 | allset_size | 0.8760 | 0.1774 (0.2291) | 0.0000 | 0.1228 | 0.0100 | -0.0000 |
| tags-ask-ubuntu | matched_all | 3 | deepsets | 0.8747 | 0.0236 (0.2291) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_all | 3 | gcn | 0.8965 | 0.1688 (0.2291) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
