# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 2 | allset | 10 | 0.6330 +/- 0.0252 | 0.6333 +/- 0.0335 |
| email-Enron | matched_unseen | 2 | allset_cpa | 10 | 0.6306 +/- 0.0180 | 0.6347 +/- 0.0361 |
| email-Enron | matched_unseen | 2 | allset_cpa_encoder | 10 | 0.6518 +/- 0.0241 | 0.6497 +/- 0.0306 |
| email-Enron | matched_unseen | 2 | allset_size | 10 | 0.6190 +/- 0.0287 | 0.6140 +/- 0.0363 |
| email-Enron | matched_unseen | 2 | deepsets | 10 | 0.6041 +/- 0.0141 | 0.6221 +/- 0.0193 |
| email-Enron | matched_unseen | 2 | gcn | 10 | 0.6589 +/- 0.0144 | 0.6464 +/- 0.0397 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 2 | allset | 0.5048 | 0.6243 | 57.5 | 0/10 |
| email-Enron | matched_unseen | 2 | allset_cpa | 0.5076 | 0.6225 | 77.5 | 0/10 |
| email-Enron | matched_unseen | 2 | allset_cpa_encoder | 0.4727 | 0.6413 | 37.5 | 0/10 |
| email-Enron | matched_unseen | 2 | allset_size | 0.4984 | 0.6156 | 47.5 | 0/10 |
| email-Enron | matched_unseen | 2 | deepsets | 0.5194 | 0.6064 | 47.5 | 0/10 |
| email-Enron | matched_unseen | 2 | gcn | 0.4863 | 0.6584 | 42.5 | 0/10 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 2 | allset_cpa | 10 | -0.0024 +/- 0.0292 | 0.0014 +/- 0.0416 | 4/10 |
| email-Enron | matched_unseen | 2 | allset_cpa_encoder | 10 | 0.0188 +/- 0.0441 | 0.0164 +/- 0.0490 | 6/10 |
| email-Enron | matched_unseen | 2 | allset_size | 10 | -0.0140 +/- 0.0289 | -0.0193 +/- 0.0204 | 3/10 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 2 | allset | 0.6330 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 2 | allset_cpa | 0.6306 | n/a (n/a) | 7.0000 | 0.2700 | 0.0664 | 0.0050 |
| email-Enron | matched_unseen | 2 | allset_cpa_encoder | 0.6518 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 2 | allset_size | 0.6190 | n/a (n/a) | 0.0000 | 0.2859 | 0.0123 | 0.0081 |
| email-Enron | matched_unseen | 2 | deepsets | 0.6041 | n/a (n/a) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 2 | gcn | 0.6589 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
