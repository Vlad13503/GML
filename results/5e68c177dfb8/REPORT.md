# Experiment results

Mean and sample standard deviation across split/training seeds.
One-seed runs are validation runs, not evidence of a stable effect.

| Dataset | Protocol | k | Model | Seeds | Test AP | Test AUROC |
|---|---|---|---|---|---|---|
| email-Enron | interpolation | 3 | allset | 3 | 0.7441 +/- 0.0575 | 0.7436 +/- 0.0864 |
| email-Enron | interpolation | 3 | allset_cpa | 3 | 0.7138 +/- 0.0308 | 0.7425 +/- 0.0133 |
| email-Enron | interpolation | 3 | allset_size | 3 | 0.7216 +/- 0.0160 | 0.7189 +/- 0.0442 |
| email-Enron | interpolation | 3 | deepsets | 3 | 0.6926 +/- 0.0152 | 0.6941 +/- 0.0154 |
| email-Enron | interpolation | 3 | gcn | 3 | 0.7251 +/- 0.0298 | 0.7202 +/- 0.0389 |
| email-Enron | order | 3 | allset | 3 | 0.7436 +/- 0.0534 | 0.7546 +/- 0.0320 |
| email-Enron | order | 3 | allset_cpa | 3 | 0.7242 +/- 0.0189 | 0.7481 +/- 0.0523 |
| email-Enron | order | 3 | allset_size | 3 | 0.7489 +/- 0.0169 | 0.7746 +/- 0.0106 |
| email-Enron | order | 3 | deepsets | 3 | 0.6521 +/- 0.0274 | 0.6881 +/- 0.0392 |
| email-Enron | order | 3 | gcn | 3 | 0.7307 +/- 0.0135 | 0.7339 +/- 0.0394 |
| email-Enron | random | 3 | allset | 3 | 0.7010 +/- 0.0340 | 0.7067 +/- 0.0235 |
| email-Enron | random | 3 | allset_cpa | 3 | 0.6971 +/- 0.0467 | 0.7093 +/- 0.0271 |
| email-Enron | random | 3 | allset_size | 3 | 0.7229 +/- 0.0259 | 0.7216 +/- 0.0456 |
| email-Enron | random | 3 | deepsets | 3 | 0.6497 +/- 0.0315 | 0.6600 +/- 0.0106 |
| email-Enron | random | 3 | gcn | 3 | 0.7131 +/- 0.0326 | 0.7131 +/- 0.0308 |
| tags-ask-ubuntu | interpolation | 3 | allset | 3 | 0.8736 +/- 0.0053 | 0.8915 +/- 0.0055 |
| tags-ask-ubuntu | interpolation | 3 | allset_cpa | 3 | 0.8703 +/- 0.0138 | 0.8886 +/- 0.0094 |
| tags-ask-ubuntu | interpolation | 3 | allset_size | 3 | 0.8691 +/- 0.0015 | 0.8864 +/- 0.0024 |
| tags-ask-ubuntu | interpolation | 3 | deepsets | 3 | 0.8499 +/- 0.0023 | 0.8723 +/- 0.0023 |
| tags-ask-ubuntu | interpolation | 3 | gcn | 3 | 0.8915 +/- 0.0128 | 0.9002 +/- 0.0120 |
| tags-ask-ubuntu | order | 3 | allset | 3 | 0.8738 +/- 0.0014 | 0.8911 +/- 0.0019 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | 0.8742 +/- 0.0025 | 0.8921 +/- 0.0020 |
| tags-ask-ubuntu | order | 3 | allset_size | 3 | 0.8734 +/- 0.0029 | 0.8911 +/- 0.0017 |
| tags-ask-ubuntu | order | 3 | deepsets | 3 | 0.8668 +/- 0.0012 | 0.8877 +/- 0.0003 |
| tags-ask-ubuntu | order | 3 | gcn | 3 | 0.8883 +/- 0.0112 | 0.9008 +/- 0.0097 |
| tags-ask-ubuntu | random | 3 | allset | 3 | 0.8732 +/- 0.0153 | 0.8982 +/- 0.0104 |
| tags-ask-ubuntu | random | 3 | allset_cpa | 3 | 0.8638 +/- 0.0161 | 0.8915 +/- 0.0103 |
| tags-ask-ubuntu | random | 3 | allset_size | 3 | 0.8745 +/- 0.0112 | 0.8975 +/- 0.0087 |
| tags-ask-ubuntu | random | 3 | deepsets | 3 | 0.8742 +/- 0.0050 | 0.8943 +/- 0.0036 |
| tags-ask-ubuntu | random | 3 | gcn | 3 | 0.8972 +/- 0.0116 | 0.9074 +/- 0.0111 |

## Convergence

| Dataset | Protocol | k | Model | Epoch-0 val AP | Best val AP | Mean best epoch | Seeds never improved |
|---|---|---|---|---|---|---|---|
| email-Enron | interpolation | 3 | allset | 0.5145 | 0.6851 | 85.0 | 0/3 |
| email-Enron | interpolation | 3 | allset_cpa | 0.5041 | 0.6850 | 68.3 | 0/3 |
| email-Enron | interpolation | 3 | allset_size | 0.5105 | 0.6775 | 41.7 | 0/3 |
| email-Enron | interpolation | 3 | deepsets | 0.5057 | 0.6599 | 21.7 | 0/3 |
| email-Enron | interpolation | 3 | gcn | 0.5094 | 0.6965 | 35.0 | 0/3 |
| email-Enron | order | 3 | allset | 0.5027 | 0.6924 | 90.0 | 0/3 |
| email-Enron | order | 3 | allset_cpa | 0.5012 | 0.6813 | 75.0 | 0/3 |
| email-Enron | order | 3 | allset_size | 0.5008 | 0.6924 | 106.7 | 0/3 |
| email-Enron | order | 3 | deepsets | 0.4921 | 0.6366 | 50.0 | 0/3 |
| email-Enron | order | 3 | gcn | 0.4890 | 0.6938 | 80.0 | 0/3 |
| email-Enron | random | 3 | allset | 0.5446 | 0.7164 | 60.0 | 0/3 |
| email-Enron | random | 3 | allset_cpa | 0.5102 | 0.7293 | 100.0 | 0/3 |
| email-Enron | random | 3 | allset_size | 0.5247 | 0.7354 | 90.0 | 0/3 |
| email-Enron | random | 3 | deepsets | 0.4907 | 0.6546 | 45.0 | 0/3 |
| email-Enron | random | 3 | gcn | 0.4943 | 0.7230 | 75.0 | 0/3 |
| tags-ask-ubuntu | interpolation | 3 | allset | 0.4724 | 0.8583 | 150.0 | 0/3 |
| tags-ask-ubuntu | interpolation | 3 | allset_cpa | 0.4837 | 0.8545 | 150.0 | 0/3 |
| tags-ask-ubuntu | interpolation | 3 | allset_size | 0.4835 | 0.8549 | 148.3 | 0/3 |
| tags-ask-ubuntu | interpolation | 3 | deepsets | 0.4975 | 0.8358 | 60.0 | 0/3 |
| tags-ask-ubuntu | interpolation | 3 | gcn | 0.4718 | 0.8807 | 146.7 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset | 0.4781 | 0.8711 | 150.0 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.4894 | 0.8729 | 148.3 | 0/3 |
| tags-ask-ubuntu | order | 3 | allset_size | 0.4861 | 0.8705 | 145.0 | 0/3 |
| tags-ask-ubuntu | order | 3 | deepsets | 0.4987 | 0.8671 | 56.7 | 0/3 |
| tags-ask-ubuntu | order | 3 | gcn | 0.4703 | 0.8884 | 101.7 | 0/3 |
| tags-ask-ubuntu | random | 3 | allset | 0.4787 | 0.8772 | 150.0 | 0/3 |
| tags-ask-ubuntu | random | 3 | allset_cpa | 0.4956 | 0.8683 | 128.3 | 0/3 |
| tags-ask-ubuntu | random | 3 | allset_size | 0.4936 | 0.8794 | 146.7 | 0/3 |
| tags-ask-ubuntu | random | 3 | deepsets | 0.4997 | 0.8769 | 81.7 | 0/3 |
| tags-ask-ubuntu | random | 3 | gcn | 0.4666 | 0.8991 | 150.0 | 0/3 |

## Paired variant changes against vanilla AllSet (test, all sizes)

| Dataset | Protocol | k | Variant | Seeds | Delta AP | Delta AUROC | Seeds AP improved |
|---|---|---|---|---|---|---|---|
| email-Enron | interpolation | 3 | allset_cpa | 3 | -0.0303 +/- 0.0361 | -0.0011 +/- 0.0734 | 0/3 |
| email-Enron | interpolation | 3 | allset_size | 3 | -0.0225 +/- 0.0428 | -0.0247 +/- 0.0464 | 1/3 |
| email-Enron | order | 3 | allset_cpa | 3 | -0.0194 +/- 0.0371 | -0.0066 +/- 0.0441 | 1/3 |
| email-Enron | order | 3 | allset_size | 3 | 0.0053 +/- 0.0546 | 0.0200 +/- 0.0361 | 2/3 |
| email-Enron | random | 3 | allset_cpa | 3 | -0.0039 +/- 0.0301 | 0.0026 +/- 0.0042 | 2/3 |
| email-Enron | random | 3 | allset_size | 3 | 0.0219 +/- 0.0417 | 0.0149 +/- 0.0367 | 2/3 |
| tags-ask-ubuntu | interpolation | 3 | allset_cpa | 3 | -0.0033 +/- 0.0100 | -0.0029 +/- 0.0083 | 1/3 |
| tags-ask-ubuntu | interpolation | 3 | allset_size | 3 | -0.0045 +/- 0.0069 | -0.0051 +/- 0.0057 | 1/3 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 3 | 0.0005 +/- 0.0026 | 0.0011 +/- 0.0008 | 2/3 |
| tags-ask-ubuntu | order | 3 | allset_size | 3 | -0.0004 +/- 0.0021 | 0.0000 +/- 0.0020 | 1/3 |
| tags-ask-ubuntu | random | 3 | allset_cpa | 3 | -0.0094 +/- 0.0189 | -0.0067 +/- 0.0108 | 1/3 |
| tags-ask-ubuntu | random | 3 | allset_size | 3 | 0.0013 +/- 0.0047 | -0.0007 +/- 0.0024 | 1/3 |

## Interpretability against performance (test)

Probe: test MAE of log(1 + size), with the constant baseline in brackets. Replication: relative change at factor 8. Neutralized: AP change when the decoder size is fixed.

| Dataset | Protocol | k | Model | Test AP | Probe MAE | Replication (modified) | Replication (decoder) | Logit change | Neutralized AP change |
|---|---|---|---|---|---|---|---|---|---|
| email-Enron | interpolation | 3 | allset | 0.7441 | 0.1632 (0.1373) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | interpolation | 3 | allset_cpa | 0.7138 | 0.1375 (0.1373) | 7.0000 | 0.1727 | 0.0397 | 0.0108 |
| email-Enron | interpolation | 3 | allset_size | 0.7216 | 0.1476 (0.1373) | 0.0000 | 0.1202 | 0.0023 | 0.0008 |
| email-Enron | interpolation | 3 | deepsets | 0.6926 | 0.1052 (0.1373) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | interpolation | 3 | gcn | 0.7251 | 0.1506 (0.1373) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 3 | allset | 0.7436 | 0.6315 (0.6843) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 3 | allset_cpa | 0.7242 | 0.2249 (0.6843) | 7.0000 | 0.2851 | 0.0353 | 0.0001 |
| email-Enron | order | 3 | allset_size | 0.7489 | 0.6669 (0.6843) | 0.0000 | 0.2329 | 0.0385 | 0.0007 |
| email-Enron | order | 3 | deepsets | 0.6521 | 0.3189 (0.6843) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | order | 3 | gcn | 0.7307 | 0.6818 (0.6843) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | random | 3 | allset | 0.7010 | 0.2431 (0.2804) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| email-Enron | random | 3 | allset_cpa | 0.6971 | 0.0763 (0.2804) | 7.0000 | 0.1573 | 0.0249 | -0.0001 |
| email-Enron | random | 3 | allset_size | 0.7229 | 0.2500 (0.2804) | 0.0000 | 0.1132 | 0.0039 | 0.0003 |
| email-Enron | random | 3 | deepsets | 0.6497 | 0.0695 (0.2804) | n/a | 7.0000 | 0.0000 | 0.0000 |
| email-Enron | random | 3 | gcn | 0.7131 | 0.2565 (0.2804) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | interpolation | 3 | allset | 0.8736 | 0.1469 (0.1116) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | interpolation | 3 | allset_cpa | 0.8703 | 0.0689 (0.1116) | 7.0000 | 0.2949 | 0.3185 | -0.0020 |
| tags-ask-ubuntu | interpolation | 3 | allset_size | 0.8691 | 0.1441 (0.1116) | 0.0000 | 0.1232 | 0.0130 | -0.0000 |
| tags-ask-ubuntu | interpolation | 3 | deepsets | 0.8499 | 0.0527 (0.1116) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | interpolation | 3 | gcn | 0.8915 | 0.1555 (0.1116) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | allset | 0.8738 | 0.3647 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | allset_cpa | 0.8742 | 0.1215 (0.4581) | 7.0000 | 0.2677 | 0.1925 | -0.0000 |
| tags-ask-ubuntu | order | 3 | allset_size | 0.8734 | 0.3784 (0.4581) | 0.0000 | 0.1577 | 0.0101 | -0.0000 |
| tags-ask-ubuntu | order | 3 | deepsets | 0.8668 | 0.1148 (0.4581) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | order | 3 | gcn | 0.8883 | 0.4000 (0.4581) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | random | 3 | allset | 0.8732 | 0.1804 (0.2291) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | random | 3 | allset_cpa | 0.8638 | 0.0279 (0.2291) | 7.0000 | 0.3198 | 0.1184 | -0.0006 |
| tags-ask-ubuntu | random | 3 | allset_size | 0.8745 | 0.1825 (0.2291) | 0.0000 | 0.1227 | 0.0059 | -0.0000 |
| tags-ask-ubuntu | random | 3 | deepsets | 0.8742 | 0.0262 (0.2291) | n/a | 7.0000 | 0.0000 | 0.0000 |
| tags-ask-ubuntu | random | 3 | gcn | 0.8972 | 0.1667 (0.2291) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

AP depends on the configured negative prevalence. Compare only matched protocols/candidates.
Per-size bins with fewer than 20 positives in any seed are flagged as sparse; seed SD is not a confidence interval.
Interpretability diagnostics and computational interventions are in each model's interpretability.json.
See paired_cpa_summary.csv for within-seed changes, size_response.csv for per-size size dependence, and convergence.csv for training curves.
