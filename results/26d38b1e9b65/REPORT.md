# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 3 | 0.6013 +/- 0.0203 | 0.5997 +/- 0.0177 |
| email-Enron | order | 3 | allset_cpa | 3 | 0.6092 +/- 0.0315 | 0.5990 +/- 0.0333 |
| tags-ask-ubuntu | order | 3 | allset | 3 | 0.7895 +/- 0.0069 | 0.8052 +/- 0.0017 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | 0.8026 +/- 0.0009 | 0.8137 +/- 0.0021 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 0.5104 | 0.6276 | 50.0 | 0/3 |
| email-Enron | order | 3 | allset_cpa | 0.5013 | 0.6193 | 66.7 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset | 0.4834 | 0.8004 | 150.0 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.4926 | 0.8122 | 150.0 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset_cpa | 3 | 0.0079 +/- 0.0258 | -0.0006 +/- 0.0211 | 2/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | 0.0131 +/- 0.0074 | 0.0085 +/- 0.0030 | 3/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 0.6013 | 0.6645 (0.6843) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 3 | allset_cpa | 0.6092 | 0.2415 (0.6843) | 7.0000 | 0.1929 | 0.0365 | 0.0037 |
| tags-ask-ubuntu | order | 3 | allset | 0.7895 | 0.3860 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.8026 | 0.1245 (0.4581) | 7.0000 | 0.2200 | 0.1058 | -0.0005 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
