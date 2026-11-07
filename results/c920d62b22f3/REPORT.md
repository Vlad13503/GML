# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 2 | allset | 3 | 0.6078 +/- 0.0194 | 0.6253 +/- 0.0409 |
| email-Enron | matched_unseen | 2 | allset_cpa | 3 | 0.6181 +/- 0.0362 | 0.6142 +/- 0.0730 |
| email-Enron | matched_unseen | 2 | allset_cpa_encoder | 3 | 0.6460 +/- 0.0052 | 0.6490 +/- 0.0266 |
| email-Enron | matched_unseen | 2 | allset_size | 3 | 0.5996 +/- 0.0155 | 0.6073 +/- 0.0185 |
| email-Enron | matched_unseen | 2 | deepsets | 3 | 0.5995 +/- 0.0173 | 0.6024 +/- 0.0133 |
| email-Enron | matched_unseen | 2 | gcn | 3 | 0.6474 +/- 0.0361 | 0.6526 +/- 0.0268 |
| tags-ask-ubuntu | matched_unseen | 2 | allset | 3 | 0.8435 +/- 0.0031 | 0.8595 +/- 0.0023 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_cpa | 3 | 0.8424 +/- 0.0021 | 0.8598 +/- 0.0008 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_cpa_encoder | 3 | 0.8686 +/- 0.0012 | 0.8794 +/- 0.0012 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_size | 3 | 0.8430 +/- 0.0017 | 0.8591 +/- 0.0027 |
| tags-ask-ubuntu | matched_unseen | 2 | deepsets | 3 | 0.8174 +/- 0.0026 | 0.8499 +/- 0.0012 |
| tags-ask-ubuntu | matched_unseen | 2 | gcn | 3 | 0.8578 +/- 0.0006 | 0.8710 +/- 0.0007 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 2 | allset | 0.5099 | 0.5943 | 53.3 | 0/3 |
| email-Enron | matched_unseen | 2 | allset_cpa | 0.5079 | 0.6064 | 30.0 | 0/3 |
| email-Enron | matched_unseen | 2 | allset_cpa_encoder | 0.4598 | 0.6176 | 11.7 | 0/3 |
| email-Enron | matched_unseen | 2 | allset_size | 0.5171 | 0.5924 | 26.7 | 0/3 |
| email-Enron | matched_unseen | 2 | deepsets | 0.5181 | 0.5816 | 53.3 | 0/3 |
| email-Enron | matched_unseen | 2 | gcn | 0.5176 | 0.6188 | 10.0 | 1/3 |
| tags-ask-ubuntu | matched_unseen | 2 | allset | 0.4779 | 0.8346 | 118.3 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_cpa | 0.4779 | 0.8381 | 111.7 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_cpa_encoder | 0.7819 | 0.8608 | 23.3 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_size | 0.4775 | 0.8336 | 128.3 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 2 | deepsets | 0.4962 | 0.8353 | 25.0 | 0/3 |
| tags-ask-ubuntu | matched_unseen | 2 | gcn | 0.4893 | 0.8641 | 66.7 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 2 | allset_cpa | 3 | 0.0103 +/- 0.0207 | -0.0111 +/- 0.0388 | 2/3 |
| email-Enron | matched_unseen | 2 | allset_cpa_encoder | 3 | 0.0382 +/- 0.0201 | 0.0237 +/- 0.0213 | 3/3 |
| email-Enron | matched_unseen | 2 | allset_size | 3 | -0.0082 +/- 0.0288 | -0.0180 +/- 0.0437 | 2/3 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_cpa | 3 | -0.0011 +/- 0.0046 | 0.0004 +/- 0.0026 | 2/3 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_cpa_encoder | 3 | 0.0251 +/- 0.0042 | 0.0200 +/- 0.0035 | 3/3 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_size | 3 | -0.0005 +/- 0.0033 | -0.0003 +/- 0.0014 | 1/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | matched_unseen | 2 | allset | 0.6078 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 2 | allset_cpa | 0.6181 | n/a (n/a) | 7.0000 | 0.1028 | 0.0067 | 0.0138 |
| email-Enron | matched_unseen | 2 | allset_cpa_encoder | 0.6460 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 2 | allset_size | 0.5996 | n/a (n/a) | 0.0000 | 0.2864 | 0.0090 | 0.0257 |
| email-Enron | matched_unseen | 2 | deepsets | 0.5995 | n/a (n/a) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | matched_unseen | 2 | gcn | 0.6474 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 2 | allset | 0.8435 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_cpa | 0.8424 | n/a (n/a) | 7.0000 | 0.2332 | 0.1366 | 0.0004 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_cpa_encoder | 0.8686 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 2 | allset_size | 0.8430 | n/a (n/a) | 0.0000 | 0.1993 | 0.0079 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 2 | deepsets | 0.8174 | n/a (n/a) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | matched_unseen | 2 | gcn | 0.8578 | n/a (n/a) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
