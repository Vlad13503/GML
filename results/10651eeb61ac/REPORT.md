# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 3 | 0.8426 +/- 0.0153 | 0.8320 +/- 0.0406 |
| email-Enron | order | 3 | allset_cpa | 3 | 0.8441 +/- 0.0220 | 0.8226 +/- 0.0401 |
| tags-ask-ubuntu | order | 3 | allset | 3 | 0.9934 +/- 0.0009 | 0.9925 +/- 0.0013 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | 0.9932 +/- 0.0007 | 0.9921 +/- 0.0011 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 0.5257 | 0.7253 | 73.3 | 0/3 |
| email-Enron | order | 3 | allset_cpa | 0.5029 | 0.7339 | 95.0 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset | 0.4735 | 0.9677 | 143.3 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.4870 | 0.9690 | 146.7 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset_cpa | 3 | 0.0015 +/- 0.0114 | -0.0095 +/- 0.0265 | 2/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | -0.0002 +/- 0.0003 | -0.0004 +/- 0.0005 | 1/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 0.8426 | 0.6562 (0.6843) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 3 | allset_cpa | 0.8441 | 0.2347 (0.6843) | 7.0000 | 0.3172 | 0.1801 | 0.0006 |
| tags-ask-ubuntu | order | 3 | allset | 0.9934 | 0.4197 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.9932 | 0.1120 (0.4581) | 7.0000 | 0.3154 | 0.1954 | -0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
