# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 3 | 0.7436 +/- 0.0534 | 0.7546 +/- 0.0320 |
| email-Enron | order | 3 | allset_cpa | 3 | 0.7242 +/- 0.0189 | 0.7481 +/- 0.0523 |
| email-Enron | order | 3 | allset_cpa_both | 3 | 0.7286 +/- 0.0107 | 0.7408 +/- 0.0171 |
| email-Enron | order | 3 | allset_cpa_encoder | 3 | 0.7249 +/- 0.0219 | 0.7210 +/- 0.0385 |
| tags-ask-ubuntu | order | 3 | allset | 3 | 0.8738 +/- 0.0014 | 0.8911 +/- 0.0019 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | 0.8742 +/- 0.0025 | 0.8921 +/- 0.0020 |
| tags-ask-ubuntu | order | 3 | allset_cpa_both | 3 | 0.8823 +/- 0.0006 | 0.8947 +/- 0.0011 |
| tags-ask-ubuntu | order | 3 | allset_cpa_encoder | 3 | 0.8839 +/- 0.0010 | 0.8960 +/- 0.0012 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 0.5027 | 0.6924 | 90.0 | 0/3 |
| email-Enron | order | 3 | allset_cpa | 0.5012 | 0.6813 | 75.0 | 0/3 |
| email-Enron | order | 3 | allset_cpa_both | 0.4624 | 0.6926 | 60.0 | 0/3 |
| email-Enron | order | 3 | allset_cpa_encoder | 0.4313 | 0.6883 | 53.3 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset | 0.4781 | 0.8711 | 150.0 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.4894 | 0.8729 | 148.3 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa_both | 0.5319 | 0.8836 | 55.0 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa_encoder | 0.6614 | 0.8832 | 51.7 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset_cpa | 3 | -0.0194 +/- 0.0371 | -0.0066 +/- 0.0441 | 1/3 |
| email-Enron | order | 3 | allset_cpa_encoder | 3 | -0.0188 +/- 0.0341 | -0.0336 +/- 0.0112 | 1/3 |
| email-Enron | order | 3 | allset_cpa_both | 3 | -0.0151 +/- 0.0431 | -0.0139 +/- 0.0163 | 1/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | 0.0005 +/- 0.0026 | 0.0011 +/- 0.0008 | 2/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa_encoder | 3 | 0.0102 +/- 0.0017 | 0.0050 +/- 0.0012 | 3/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa_both | 3 | 0.0085 +/- 0.0009 | 0.0036 +/- 0.0012 | 3/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | order | 3 | allset | 0.7436 | 0.6315 (0.6843) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 3 | allset_cpa | 0.7242 | 0.2249 (0.6843) | 7.0000 | 0.2851 | 0.0353 | 0.0001 |
| email-Enron | order | 3 | allset_cpa_both | 0.7286 | 0.2588 (0.6843) | 7.0000 | 0.2141 | 0.0785 | 0.0035 |
| email-Enron | order | 3 | allset_cpa_encoder | 0.7249 | 0.6545 (0.6843) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | allset | 0.8738 | 0.3647 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.8742 | 0.1215 (0.4581) | 7.0000 | 0.2677 | 0.1925 | -0.0000 |
| tags-ask-ubuntu | order | 3 | allset_cpa_both | 0.8823 | 0.1225 (0.4581) | 7.0000 | 0.2390 | 0.0883 | -0.0001 |
| tags-ask-ubuntu | order | 3 | allset_cpa_encoder | 0.8839 | 0.4035 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
