# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | order | 4 | allset | 3 | 0.7871 +/- 0.0602 | 0.7934 +/- 0.0536 |
| email-Enron | order | 4 | allset_cpa | 3 | 0.7732 +/- 0.0759 | 0.7790 +/- 0.0738 |
| email-Enron | order | 4 | allset_size | 3 | 0.7605 +/- 0.0463 | 0.7752 +/- 0.0571 |
| email-Enron | order | 4 | deepsets | 3 | 0.6928 +/- 0.0407 | 0.6982 +/- 0.0315 |
| email-Enron | order | 4 | gcn | 3 | 0.7623 +/- 0.0172 | 0.7656 +/- 0.0045 |
| tags-ask-ubuntu | order | 4 | allset | 3 | 0.8857 +/- 0.0028 | 0.9045 +/- 0.0013 |
| tags-ask-ubuntu | order | 4 | allset_cpa | 3 | 0.8819 +/- 0.0053 | 0.9029 +/- 0.0041 |
| tags-ask-ubuntu | order | 4 | allset_size | 3 | 0.8764 +/- 0.0028 | 0.9007 +/- 0.0008 |
| tags-ask-ubuntu | order | 4 | deepsets | 3 | 0.8836 +/- 0.0059 | 0.9024 +/- 0.0058 |
| tags-ask-ubuntu | order | 4 | gcn | 3 | 0.8995 +/- 0.0155 | 0.9118 +/- 0.0134 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 4 | allset | 0.5321 | 0.7355 | 116.7 | 0/3 |
| email-Enron | order | 4 | allset_cpa | 0.5123 | 0.7099 | 81.7 | 0/3 |
| email-Enron | order | 4 | allset_size | 0.5149 | 0.7071 | 110.0 | 0/3 |
| email-Enron | order | 4 | deepsets | 0.4921 | 0.6387 | 51.7 | 0/3 |
| email-Enron | order | 4 | gcn | 0.4866 | 0.7119 | 56.7 | 0/3 |
| tags-ask-ubuntu | order | 4 | allset | 0.4756 | 0.8786 | 150.0 | 0/3 |
| tags-ask-ubuntu | order | 4 | allset_cpa | 0.4934 | 0.8771 | 148.3 | 0/3 |
| tags-ask-ubuntu | order | 4 | allset_size | 0.4905 | 0.8720 | 148.3 | 0/3 |
| tags-ask-ubuntu | order | 4 | deepsets | 0.4970 | 0.8756 | 80.0 | 0/3 |
| tags-ask-ubuntu | order | 4 | gcn | 0.4661 | 0.8957 | 150.0 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 4 | allset_cpa | 3 | -0.0138 +/- 0.0162 | -0.0144 +/- 0.0297 | 1/3 |
| email-Enron | order | 4 | allset_size | 3 | -0.0265 +/- 0.0322 | -0.0182 +/- 0.0176 | 1/3 |
| tags-ask-ubuntu | order | 4 | allset_cpa | 3 | -0.0038 +/- 0.0079 | -0.0016 +/- 0.0048 | 1/3 |
| tags-ask-ubuntu | order | 4 | allset_size | 3 | -0.0092 +/- 0.0017 | -0.0038 +/- 0.0021 | 0/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | order | 4 | allset | 0.7871 | 0.7201 (0.8121) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 4 | allset_cpa | 0.7732 | 0.3035 (0.8121) | 7.0000 | 0.1536 | 0.1529 | -0.0004 |
| email-Enron | order | 4 | allset_size | 0.7605 | 0.7610 (0.8121) | 0.0000 | 0.1964 | 0.0152 | -0.0047 |
| email-Enron | order | 4 | deepsets | 0.6928 | 0.3969 (0.8121) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 4 | gcn | 0.7623 | 0.7686 (0.8121) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 4 | allset | 0.8857 | 0.3350 (0.4270) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 4 | allset_cpa | 0.8819 | 0.0783 (0.4270) | 7.0000 | 0.3385 | 0.2215 | -0.0001 |
| tags-ask-ubuntu | order | 4 | allset_size | 0.8764 | 0.3258 (0.4270) | 0.0000 | 0.1364 | 0.0114 | 0.0000 |
| tags-ask-ubuntu | order | 4 | deepsets | 0.8836 | 0.0812 (0.4270) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 4 | gcn | 0.8995 | 0.3296 (0.4270) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
